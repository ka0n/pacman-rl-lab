# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare pre-trained agents and random baseline on identical test seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import evaluate_random
from .core import PacmanGame
from .dqn import make_qnet, select_device
from .train import evaluate


def load_model(checkpoint: str | Path, device):
    import torch
    # Checkpoints must be local/trusted; loads only safe tensors/primitive dictionaries.
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = make_qnet(tuple(payload["input_shape"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def run_comparison(direct: str, curriculum: str, episodes: int = 100,
                   seed: int = 20000, device_name: str = "auto") -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    device = select_device(device_name)
    direct_model, direct_payload = load_model(direct, device)
    curriculum_model, curriculum_payload = load_model(curriculum, device)
    if direct_payload["maze"] != curriculum_payload["maze"]:
        raise ValueError("Models use different mazes; comparison not meaningful")
    maze = direct_payload["maze"]
    expected = PacmanGame(maze=maze, ghosts=2).observe().shape
    if tuple(direct_payload["input_shape"]) != expected or tuple(curriculum_payload["input_shape"]) != expected:
        raise ValueError("Incompatible input shape")
    return {
        "maze": maze,
        "episodes": episodes,
        "seed_start": seed,
        "random": evaluate_random(episodes=episodes, maze=maze, ghosts=2, seed=seed),
        "direct": evaluate(direct_model, device, maze, episodes=episodes, seed=seed),
        "curriculum": evaluate(curriculum_model, device, maze, episodes=episodes, seed=seed),
    }


def main():
    parser = argparse.ArgumentParser(description="Fair 100-episode seeded evaluation")
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result file")
    args = parser.parse_args()
    results = run_comparison(str(args.direct), str(args.curriculum), args.episodes, args.seed, args.device)
    print(f"Identical seeds: {args.seed}..{args.seed + args.episodes - 1} | "
          f"maze={results['maze']} | 2 ghosts")
    print(f"{'Agent':<12} {'Pellets %':>10} {'Wins %':>9} {'Wall/ep':>9} {'Still %':>9}")
    for name in ("random", "direct", "curriculum"):
        x = results[name]
        pct = x.get("pellet_percent", x.get("avg_pellet_percent", 0.0))
        wall = x.get("wall_bumps_per_episode", float("nan"))
        still = x.get("stationary_percent", float("nan"))
        print(f"{name:<12} {pct:>10.2f} {x['win_percent']:>9.2f} {wall:>9.1f} {still:>9.1f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
