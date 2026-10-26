# SPDX-License-Identifier: GPL-3.0-or-later
"""Safe v0.9 checkpoint evaluation, fine-tuning and warm resume for v0.8 models.

Usage: python -m pacman_lab.cnn_v09 {evaluate,finetune,resume} --help
Does not modify source checkpoints or an existing marathon manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .cnn_batch import eval_batched, run_batched
from .cnn_models import SPECS, make_model
from .dqn import select_device
from .marathon_train import checkpoint_read


DEFAULT_MAZES = ("large", "large_unseen_a", "large_unseen_b")


def evaluate_checkpoints(checkpoints, *, output: Path, episodes: int = 1000,
                         mazes=DEFAULT_MAZES, seed: int = 70000,
                         engine: str = "rust", device_name: str = "auto",
                         worlds: int = 16):
    """Independent audit; common held-out episode seeds for paired comparison.

    This is NOT a new selection-validation set; repeated tuning against these
    results would invalidate the claim that they are held out.
    """
    if episodes < 1 or worlds < 1 or seed < 0:
        raise ValueError("episodes and worlds must be >=1; seed >=0")
    if not checkpoints or not mazes:
        raise ValueError("Need checkpoint(s) and maze(s)")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior audit: {output}")
    import torch
    from .core import MAPS

    device = select_device(device_name)
    data = {"version": "0.9.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_per_maze": episodes, "seed_base": seed,
            "engine": engine, "device": str(device), "worlds": worlds,
            "policy": "Greedy (no exploratory actions)",
            "note": "Independent held-out evaluation; do not optimize hyperparameters on these test seeds.",
            "results": []}
    for cp in checkpoints:
        cp = Path(cp).resolve(strict=True)
        payload = checkpoint_read(cp, device)
        arch = payload.get("architecture")
        if arch not in SPECS:
            raise ValueError(f"Unsupported checkpoint architecture {arch} in {cp}")
        shape = tuple(payload["input_shape"])
        model = make_model(arch, shape).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        item = {"checkpoint": str(cp), "sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
                "architecture": arch, "source_steps": int(payload["step"]),
                "source_seed": payload.get("config", {}).get("seed"), "metrics": {}}
        for maze in mazes:
            if maze not in MAPS:
                raise ValueError(f"Unknown maze {maze}")
            h, w = len(MAPS[maze]), len(MAPS[maze][0])
            if (h, w) != shape[1:]:
                raise ValueError(f"Maze {maze} shape {(h,w)} does not match source {shape}")
            metrics = eval_batched(model, arch, device, maze=maze, episodes=episodes,
                                   seed=seed, engine=engine, worlds=worlds)
            item["metrics"][maze] = metrics
            print(f"AUDIT {cp.parent.name}/{cp.name} {maze}: "
                  f"{metrics['win_percent']:.2f}% wins, {metrics['pellet_percent']:.2f}% pellets",flush=True)
        data["results"].append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _config_kwargs(payload):
    cfg = payload["config"]
    keys = ("architecture", "steps", "seed", "maze", "worlds", "eval_every", "save_every",
            "eval_episodes", "replay_capacity", "batch_size", "warmup", "train_every",
            "target_every", "burnin", "unroll", "learning_rate", "epsilon_start",
            "epsilon_end", "epsilon_decay_fraction", "engine")
    out = {k: cfg[k] for k in keys}
    for k in ('death_penalty', 'risk_shaping_weight', 'fresh_optimizer', 'matched_scratch_init'):
        if k in cfg:
            out[k] = cfg[k]
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate/fine-tune v0.8 CNN checkpoints without modifying source models")
    sub = p.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="Independent audit of best.pt checkpoints")
    ev.add_argument("--suite", type=Path, help="v0.8 run suite; selects all matching best.pt files")
    ev.add_argument("--checkpoint", type=Path, action="append", default=[], help="Repeat for multiple checkpoints")
    ev.add_argument("--architecture", choices=("cnn_single", "cnn_history4", "mlp_wide", "cnn_gru"),
                    action="append", default=[], help="Only when using --suite; default CNN and CNN+History")
    ev.add_argument("--episodes", type=int, default=1000)
    ev.add_argument("--maze", choices=DEFAULT_MAZES, action="append", default=[])
    ev.add_argument("--test-seed", type=int, default=70000)
    ev.add_argument("--worlds", type=int, default=16)
    ev.add_argument("--engine", choices=("rust", "python"), default="rust")
    ev.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ev.add_argument("--output", required=True, type=Path)

    ft = sub.add_parser("finetune", help="New run from a source best.pt; never writes into source")
    ft.add_argument("--checkpoint", required=True, type=Path)
    ft.add_argument("--architecture", choices=tuple(SPECS), default=None,
                    help="Target architecture; cnn_motion accepts cnn_single checkpoint")
    ft.add_argument("--output", required=True, type=Path)
    ft.add_argument("--steps", type=int, default=500000, help="Additional transitions")
    ft.add_argument("--learning-rate", type=float, default=0.00003)
    ft.add_argument("--epsilon-start", type=float, default=0.10)
    ft.add_argument("--epsilon-end", type=float, default=0.05)
    ft.add_argument("--epsilon-decay-fraction", type=float, default=0.70)
    ft.add_argument("--eval-every", type=int, default=25000)
    ft.add_argument("--save-every", type=int, default=25000)
    ft.add_argument("--eval-episodes", type=int, default=100)
    ft.add_argument("--worlds", type=int, default=16)
    ft.add_argument("--engine", choices=("rust", "python"), default="rust")
    ft.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ft.add_argument("--fresh-optimizer", action="store_true",
                    help="Do not restore Adam states, for fair control vs cnn_motion transfer")
    ft.add_argument("--death-penalty", type=float, default=None,
                    help="Override environment's -5 death reward (e.g. -20); training only")
    ft.add_argument("--risk-shaping-weight", type=float, default=0.0,
                    help="Optional potential-based topological risk shaping (try 0.5), off by default")
    ft.add_argument("--seed", type=int, default=None, help="New RNG seed; default source seed + 5,000,000")

    rs = sub.add_parser("resume", help="Resume an unfinished fine-tune run with original configuration")
    rs.add_argument("--checkpoint", required=True, type=Path, help="Fine-tune run's last.pt")
    rs.add_argument("--output", required=True, type=Path, help="Same fine-tune output directory")
    rs.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")

    a = p.parse_args()
    if a.command == "evaluate":
        paths = list(a.checkpoint)
        if a.suite:
            selected = set(a.architecture or ("cnn_single", "cnn_history4"))
            if not a.suite.is_dir():
                p.error(f"Suite directory missing: {a.suite}")
            paths += sorted(c for c in a.suite.glob("*_seed*/best.pt")
                            if c.parent.name.split("_seed")[0] in selected)
        if not paths:
            p.error("No checkpoints found. Pass --checkpoint and/or --suite")
        evaluate_checkpoints(paths, output=a.output, episodes=a.episodes,
                             mazes=a.maze or DEFAULT_MAZES, seed=a.test_seed,
                             engine=a.engine, device_name=a.device, worlds=a.worlds)
    elif a.command == "finetune":
        device = select_device(a.device)
        source = checkpoint_read(a.checkpoint, device)
        arch = a.architecture or source.get("architecture")
        if arch not in SPECS:
            p.error("Only v0.8 CNN/GRU/MLP checkpoints are supported")
        cfg = source["config"]
        seed = a.seed if a.seed is not None else int(cfg["seed"]) + 5_000_000
        if a.architecture and a.architecture != source.get("architecture") and not (
                a.architecture == "cnn_motion" and source.get("architecture") == "cnn_single"):
            p.error("Only cnn_single -> cnn_motion architecture transfer is supported")
        if a.output.resolve() == a.checkpoint.parent.resolve():
            p.error("Fine-tune output cannot be the source checkpoint folder")
        run_batched(architecture=arch, steps=a.steps, seed=seed, output=a.output,
                    worlds=a.worlds, device_name=a.device, eval_every=a.eval_every,
                    save_every=a.save_every, eval_episodes=a.eval_episodes,
                    replay_capacity=int(cfg["replay_capacity"]),
                    batch_size=int(cfg["batch_size"]), warmup=int(cfg["warmup"]),
                    train_every=int(cfg["train_every"]), target_every=int(cfg["target_every"]),
                    burnin=int(cfg["burnin"]), unroll=int(cfg["unroll"]),
                    learning_rate=a.learning_rate, epsilon_start=a.epsilon_start,
                    epsilon_end=a.epsilon_end, epsilon_decay_fraction=a.epsilon_decay_fraction,
                    init_checkpoint=str(a.checkpoint), maze=cfg["maze"], engine=a.engine,
                    death_penalty=a.death_penalty,
                    risk_shaping_weight=a.risk_shaping_weight,
                    fresh_optimizer=a.fresh_optimizer)
    elif a.command == "resume":
        device = select_device(a.device)
        source = checkpoint_read(a.checkpoint, device)
        if not source.get("fine_tune"):
            p.error("Expected fine-tune last.pt with provenance; use legacy resume for old runs")
        if a.output.resolve() != a.checkpoint.parent.resolve():
            p.error("Resume output must be same directory as last.pt")
        kwargs = _config_kwargs(source)
        run_batched(output=a.output, device_name=a.device,
                    resume=str(a.checkpoint), **kwargs)


if __name__ == "__main__":
    main()
