# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone CPU/GPU training-pipeline benchmark for Pac-Man RL Lab v0.7.

No checkpoints are modified.  This benchmark uses real environment transitions,
model inference, replay samples, loss and optimizer updates, but a deliberately
short/fixed epsilon schedule; it measures throughput, not learning quality.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np

from .cnn_models import SPECS, make_model, parameters_count
from .cnn_train import EpisodeReplay, minibatch_loss
from .core import PacmanGame
from .dqn import select_device
from .cnn_motion import observer_for


def run_profile(*, architecture: str, engine: str = "python", device_name: str = "auto",
                maze: str = "large", steps: int = 3000, warmup: int = 800,
                batch_size: int = 64, replay_capacity: int = 10000,
                train_every: int = 8, target_every: int = 2000, epsilon: float = .2,
                seed: int = 999, burnin: int = 4, unroll: int = 8,
                output: str | Path | None = None, sync_gpu: bool = True):
    import torch

    if architecture not in SPECS:
        raise ValueError(f"Unknown architecture {architecture}")
    if engine not in ("python", "rust"):
        raise ValueError("engine must be python or rust")
    if steps <= 0 or warmup < 0 or batch_size <= 0 or replay_capacity < batch_size or train_every <= 0:
        raise ValueError("Invalid step, warmup or batch configuration")
    if target_every < 1 or burnin < 0 or unroll < 1 or not 0 <= epsilon <= 1:
        raise ValueError("Invalid target, sequence or epsilon setting")

    device = select_device(device_name)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    spec = SPECS[architecture]
    if engine == "rust":
        from .native import NativePacmanGame
        game = NativePacmanGame(maze=maze, ghosts=2)
    else:
        game = PacmanGame(maze=maze, ghosts=2)

    def sync():
        if sync_gpu and device.type == "cuda":
            torch.cuda.synchronize(device)
        elif sync_gpu and device.type == "mps":
            torch.mps.synchronize()

    timings = {k: 0.0 for k in ("policy_forward", "game_step_and_reset", "replay_write",
                                "replay_sample", "loss_and_transfer", "backward", "optimizer_step",
                                "target_copy", "setup")}
    counts = {"episodes": 0, "policy_calls": 0, "updates": 0,
              "replay_samples": 0, "random_actions": 0}
    @contextmanager
    def meter(key):
        sync()
        before = time.perf_counter()
        try:
            yield
        finally:
            sync()
            timings[key] += time.perf_counter() - before

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        with meter("setup"):
            history = observer_for(architecture, spec.frames)
            state = history.reset(game.reset(seed=1_000_000 + seed)[0])
            shape = tuple(state.shape)
            online = make_model(architecture, shape).to(device)
            target = make_model(architecture, shape).to(device)
            target.load_state_dict(online.state_dict())
            target.eval()
            optimizer = torch.optim.Adam(online.parameters(), lr=0.00015)
            replay = (EpisodeReplay(replay_capacity, burnin + unroll)
                      if spec.recurrent else deque(maxlen=replay_capacity))
        hidden = None
        for t in range(1, steps + 1):
            # Match the actual trainer: recurrent nets need an inference pass at
            # every step to update their persistent hidden state.
            if spec.recurrent:
                with meter("policy_forward"):
                    with torch.inference_mode():
                        v = torch.as_tensor(state, device=device, dtype=torch.float32)[None, None]
                        q, hidden = online(v, hidden)
                        proposed_action = int(q[0, -1].argmax().item())
                    counts["policy_calls"] += 1
                if rng.random() < epsilon:
                    counts["random_actions"] += 1
                    action = rng.randrange(4)
                else:
                    action = proposed_action
            elif rng.random() < epsilon:
                counts["random_actions"] += 1
                action = rng.randrange(4)
            else:
                with meter("policy_forward"):
                    with torch.inference_mode():
                        v = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                        action = int(online(v).argmax(1).item())
                    counts["policy_calls"] += 1

            with meter("game_step_and_reset"):
                result = game.step(action)
                done = bool(result.terminated or result.truncated)
                next_state = history.step(result.observation)
                if done:
                    counts["episodes"] += 1
                    reset_obs = game.reset(seed=1_000_000 + seed + counts["episodes"])[0]
            with meter("replay_write"):
                item = (state.copy(), action, float(result.reward), next_state.copy(), done)
                if spec.recurrent:
                    replay.append(item, done)
                else:
                    replay.append(item)

            if done:
                state = history.reset(reset_obs)
                hidden = None
            else:
                state = next_state
            if t % train_every == 0 and t >= warmup:
                if spec.recurrent:
                    ready = replay.total >= max(warmup, batch_size) and bool(replay.eligible)
                else:
                    ready = len(replay) >= max(warmup, batch_size)
                if ready:
                    with meter("replay_sample"):
                        samples = (replay.sample(rng, batch_size) if spec.recurrent
                                   else rng.sample(replay, batch_size))
                        counts["replay_samples"] += 1
                    with meter("loss_and_transfer"):
                        loss = minibatch_loss(online, target, samples, architecture, device, burnin)
                    with meter("backward"):
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    with meter("optimizer_step"):
                        optimizer.step()
                    counts["updates"] += 1
            if t % target_every == 0:
                with meter("target_copy"):
                    target.load_state_dict(online.state_dict())
        sync()
    finally:
        if engine == "rust":
            game.close()

    elapsed = time.perf_counter() - start
    measured = sum(timings.values())
    timings["other_python_and_profiling_overhead"] = max(0.0, elapsed - measured)
    result = {
        "benchmark": "pacman_rl_lab_v0.7_pipeline_profile_v1",
        "settings": {
            "architecture": architecture, "engine": engine, "device": str(device),
            "maze": maze, "steps": steps, "warmup": warmup, "batch_size": batch_size,
            "train_every": train_every, "target_every": target_every,
            "epsilon": epsilon, "seed": seed, "burnin": burnin, "unroll": unroll,
            "replay_capacity": replay_capacity,
            "gpu_sync_each_section": sync_gpu and device.type in ("cuda", "mps"),
            "cuda_sync_each_section": sync_gpu and device.type == "cuda",
        },
        "parameters": parameters_count(online),
        "torch_version": torch.__version__,
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else
                     "Apple Metal (MPS)" if device.type == "mps" else None),
        "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
                                 if device.type == "cuda" else None,
        "counts": counts,
        "elapsed_seconds": round(elapsed, 4),
        "steps_per_second": round(steps / elapsed, 2),
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        "timings_percent": {k: round(100.0 * v / elapsed, 2) for k, v in timings.items()},
        "caveat": "Synchronizing individual GPU sections adds diagnostic overhead; results are not production throughput. No checkpoint was loaded or saved. Exploration is constant, not a learning experiment."
    }
    print(f"{architecture} | {engine} | {device} | {parameters_count(online):,} parameters")
    print(f"{steps:,} env steps / {counts['updates']:,} gradient updates / {counts['policy_calls']:,} network policy calls")
    print(f"Overall: {elapsed:.2f}s, {steps / elapsed:,.1f} steps/s (profiled)")
    for key, seconds in sorted(timings.items(), key=lambda entry: -entry[1]):
        print(f"  {key:<33} {seconds:>8.2f}s ({100*seconds/elapsed:5.1f}%)")
    if output is not None:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved profile: {p}")
    return result


def main():
    p = argparse.ArgumentParser(description="Real CNN/GRU pipeline profile; no checkpoints changed")
    p.add_argument("--architecture", choices=list(SPECS), default="cnn_single")
    p.add_argument("--engine", choices=["python", "rust"], default="rust")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    p.add_argument("--maze", choices=["small", "large", "xlarge"], default="large")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=10000)
    p.add_argument("--train-every", type=int, default=8)
    p.add_argument("--target-every", type=int, default=2000)
    p.add_argument("--epsilon", type=float, default=.2)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--burnin", type=int, default=4)
    p.add_argument("--unroll", type=int, default=8)
    p.add_argument("--output", type=Path)
    p.add_argument("--no-gpu-sync", action="store_true", help="less overhead; timings may not reflect async GPU work")
    a = p.parse_args()
    run_profile(architecture=a.architecture, engine=a.engine, device_name=a.device,
                maze=a.maze, steps=a.steps, warmup=a.warmup, batch_size=a.batch_size,
                replay_capacity=a.replay_capacity, train_every=a.train_every,
                target_every=a.target_every, epsilon=a.epsilon, seed=a.seed,
                burnin=a.burnin, unroll=a.unroll, output=a.output,
                sync_gpu=not a.no_gpu_sync)


if __name__ == "__main__":
    main()
