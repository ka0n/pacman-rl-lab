# SPDX-License-Identifier: GPL-3.0-or-later
"""Apple Silicon diagnostics and bounded smoke tests, never a full study.

python -m pacman_lab.macos doctor
python -m pacman_lab.macos smoke --device mps --engine both
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np


def runtime_info():
    import torch
    from .native import _load_library
    cpu = platform.processor()
    if platform.system() == 'Darwin':
        result = subprocess.run(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True, check=False)
        cpu = result.stdout.strip() or cpu
    try:
        _load_library()
        native = True
        native_error = None
    except (FileNotFoundError, OSError) as exc:
        native = False
        native_error = str(exc)
    return {'platform': platform.platform(), 'machine': platform.machine(), 'cpu': cpu,
            'python': platform.python_version(), 'torch': torch.__version__,
            'mps_built': torch.backends.mps.is_built(),
            'mps_available': torch.backends.mps.is_available(),
            'rust_library_available': native, 'rust_error': native_error,
            'note': 'MPS requires Metal GPU access. A restricted process may report unavailable.'}


def gradient_probe(architecture, device_name):
    """One real Adam update; compare initial device Q-values with CPU first."""
    import torch
    from .cnn_models import SPECS, make_model
    from .cnn_motion import observer_for
    from .cnn_train import minibatch_loss
    from .core import PacmanGame
    from .dqn import select_device

    device = select_device(device_name)
    torch.manual_seed(710091)
    spec = SPECS[architecture]
    game = PacmanGame('small', ghosts=0, max_steps=4, seed=710091)
    observer = observer_for(architecture, spec.frames)
    state = observer.reset(game.observe())
    trajectory = []
    for _ in range(4):
        result = game.step(3)
        ns = observer.step(result.observation)
        trajectory.append((state.copy(), 3, result.reward, ns.copy(),
                           result.terminated or result.truncated))
        state = ns
    cpu_model = make_model(architecture, tuple(state.shape)).eval()
    online = deepcopy(cpu_model).to(device).train()
    target = deepcopy(online).eval()
    x = torch.as_tensor(np.stack([entry[0] for entry in trajectory]), dtype=torch.float32)
    if spec.recurrent:
        x = x[None]
    with torch.no_grad():
        cpu_q = cpu_model(x)
        device_q = online(x.to(device))
        if spec.recurrent:
            cpu_q, device_q = cpu_q[0], device_q[0]
        torch.testing.assert_close(device_q.cpu(), cpu_q, rtol=1e-3, atol=1e-4)
    samples = [trajectory] if spec.recurrent else trajectory
    loss = minibatch_loss(online, target, samples, architecture, device, burnin=1)
    if not torch.isfinite(loss):
        raise AssertionError(f'{architecture}: nonfinite loss')
    optimizer = torch.optim.Adam(online.parameters(), lr=0.00015)
    before = next(online.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in online.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise AssertionError(f'{architecture}: invalid gradients')
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.)
    optimizer.step()
    if torch.equal(before, next(online.parameters()).detach()):
        raise AssertionError(f'{architecture}: optimizer did not change weights')
    if device.type == 'mps':
        torch.mps.synchronize()
    return {'architecture': architecture, 'device': str(device), 'loss': loss.item(),
            'optimizer_updates': 1, 'cpu_q_parity': True, 'finite_gradients': True}


def run_smoke(*, output: Path, device_name='mps', engine='both', steps=64):
    import torch
    from .cnn_batch import run_batched
    from .cnn_models import SPECS
    from .dqn import select_device
    from .marathon_train import checkpoint_read
    from .native_bench import verify

    if engine not in ('python', 'rust', 'both') or not 16 <= steps <= 256:
        raise ValueError('Smoke requires python/rust/both and 16..256 transitions per run')
    device = select_device(device_name)  # Fail explicitly if MPS was requested but unavailable.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    engines = ('python', 'rust') if engine == 'both' else (engine,)
    if 'rust' in engines:
        verify(episodes=2, seed=710091)
    probes = [gradient_probe(arch, str(device)) for arch in SPECS]
    runs = []
    for selected_engine in engines:
        for architecture in ('cnn_single', 'cnn_motion'):
            folder = output / f'{architecture}_{selected_engine}'
            run_batched(architecture=architecture, steps=steps, seed=710091, output=folder,
                        worlds=4, device_name=str(device), engine=selected_engine,
                        maze='small', warmup=16, batch_size=4, replay_capacity=256,
                        train_every=8, target_every=16, eval_every=steps,
                        save_every=steps, eval_episodes=2, death_penalty=-20.,
                        matched_scratch_init=architecture == 'cnn_motion')
            checkpoint = checkpoint_read(folder / 'last.pt', torch.device('cpu'))
            with (folder / 'metrics.csv').open(newline='', encoding='utf-8') as fp:
                loss = float(list(csv.DictReader(fp))[-1]['train_loss'])
            if checkpoint['step'] != steps or loss is None or not np.isfinite(loss):
                raise AssertionError(f'{folder}: incomplete smoke or invalid loss')
            updates = max(int(state['step'].item()) for state in checkpoint['optimizer']['state'].values())
            runs.append({'architecture': architecture, 'engine': selected_engine,
                         'device': str(device), 'transitions': steps,
                         'optimizer_updates': updates, 'loss': loss,
                         'checkpoint_roundtrip': True})
    report = {'runtime': runtime_info(), 'gradient_probes': probes, 'runs': runs,
              'caveat': 'Smoke tests on fresh random models; no evidence about learning quality. '
                        'No original checkpoint or historical test seed was used.'}
    (output / 'smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Print runtime, MPS and native-library availability')
    smoke = commands.add_parser('smoke', help='Bounded gradient, replay, engine and checkpoint smoke')
    smoke.add_argument('--device', choices=('cpu', 'mps', 'auto'), default='mps')
    smoke.add_argument('--engine', choices=('python', 'rust', 'both'), default='both')
    smoke.add_argument('--steps', type=int, default=64)
    smoke.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        print(json.dumps(runtime_info(), indent=2))
    else:
        if args.output is None:
            # Reserve a unique parent; run_smoke refuses to overwrite its result directory.
            parent = Path(tempfile.mkdtemp(prefix='pacman-macos-smoke-'))
            args.output = parent / 'results'
        result = run_smoke(output=args.output, device_name=args.device,
                           engine=args.engine, steps=args.steps)
        print(json.dumps(result, indent=2))
        print(f'Smoke results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
