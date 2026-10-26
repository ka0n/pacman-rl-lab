# SPDX-License-Identifier: GPL-3.0-or-later
"""Held-out evaluation of one or more checkpoints against a random baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import evaluate_random
from .compare import load_model
from .core import PacmanGame
from .dqn import select_device
from .train import evaluate


def benchmark(checkpoints: list[str], episodes: int = 100, seed: int = 20000,
              device_name: str = "auto") -> dict:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed must be nonnegative")
    device = select_device(device_name)
    models = [load_model(path, device) for path in checkpoints]
    mazes = {payload["maze"] for _, payload in models}
    if len(mazes) != 1:
        raise ValueError("All checkpoints must use the same maze")
    maze = mazes.pop()
    expected = PacmanGame(maze=maze, ghosts=2).observe().shape
    if any(tuple(payload["input_shape"]) != expected for _, payload in models):
        raise ValueError("Checkpoint observation shape does not match game")
    results = {"maze": maze, "episodes": episodes, "seed_start": seed,
               "random": evaluate_random(episodes=episodes, maze=maze, ghosts=2, seed=seed),
               "agents": []}
    for path, (model, payload) in zip(checkpoints, models):
        results["agents"].append({
            "checkpoint": str(path),
            "training_steps": payload.get("steps"),
            "metrics": evaluate(model, device, maze, episodes=episodes, seed=seed),
        })
    return results


def main():
    p = argparse.ArgumentParser(description="Compare DQN checkpoints on held-out seeds")
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Can specify repeatedly to compare models")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    result = benchmark(args.checkpoint, args.episodes, args.seed, args.device)
    print(f"Unseen evaluation: {args.episodes} games, seeds "
          f"{args.seed}..{args.seed + args.episodes - 1}, {result['maze']}, 2 ghosts")
    print(f"{'Agent':<40} {'Pellets %':>10} {'Wins %':>8} {'Wall/ep':>9} {'Still %':>9}")

    def row(label, metrics):
        pellets = metrics.get("pellet_percent", metrics.get("avg_pellet_percent"))
        print(f"{label:<40} {pellets:>10.2f} {metrics['win_percent']:>8.2f} "
              f"{metrics['wall_bumps_per_episode']:>9.1f} "
              f"{metrics['stationary_percent']:>9.1f}")

    row("random", result["random"])
    for i, agent in enumerate(result["agents"], 1):
        path = Path(agent["checkpoint"])
        row(f"{i}. {path.parent.name}/{path.name}"[:40], agent["metrics"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
