# SPDX-License-Identifier: GPL-3.0-or-later
"""Zero-shot test of a saved checkpoint on same-size unseen maze topologies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import evaluate_random
from .compare import load_model
from .core import MAPS, PacmanGame
from .dqn import select_device
from .train import evaluate

UNSEEN_MAPS = ("unseen_a", "unseen_b")


def run_transfer(checkpoint: str, episodes: int = 100, seed: int = 40000,
                 device_name: str = "auto", maps: tuple[str, ...] | None = None) -> dict:
    if episodes < 1 or seed < 0:
        raise ValueError("episodes >= 1 and seed >= 0 required")
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    chosen = (payload["maze"], *UNSEEN_MAPS) if maps is None else maps
    if not chosen:
        raise ValueError("At least one maze required")
    for maze in chosen:
        if maze not in MAPS:
            raise ValueError(f"Unknown maze: {maze}")
        expected = PacmanGame(maze=maze, ghosts=2).observe().shape
        if tuple(payload["input_shape"]) != expected:
            raise ValueError(f"Map {maze!r} has shape {expected}, checkpoint needs {payload['input_shape']}")
    result = {"checkpoint": checkpoint, "trained_maze": payload["maze"],
              "training_steps": payload.get("steps"), "episodes": episodes,
              "seed_start": seed, "maps": []}
    for maze in chosen:
        random_metrics = evaluate_random(episodes=episodes, maze=maze, ghosts=2, seed=seed)
        agent_metrics = evaluate(model, device, maze, episodes=episodes, seed=seed)
        result["maps"].append({"maze": maze, "trained_map": maze == payload["maze"],
                               "pellets_initial": PacmanGame(maze=maze, ghosts=2).initial_pellet_count,
                               "random": random_metrics, "agent": agent_metrics})
    return result


def main():
    p = argparse.ArgumentParser(description="Zero-shot maze transfer, NO additional training")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=40000)
    p.add_argument("--maze", action="append", choices=tuple(MAPS),
                   help="Repeated to choose maps. Default: training map + unseen_a + unseen_b")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    result = run_transfer(a.checkpoint, a.episodes, a.seed, a.device,
                          tuple(a.maze) if a.maze is not None else None)
    print(f"Zero-shot transfer: {a.episodes} episodes per map; same-size mazes, "
          f"{result['trained_maze']} checkpoint, 2 ghosts; greedy policy")
    print(f"{'Maze':<14} {'Agent pellets':>14} {'Agent wins':>11} {'Random pellets':>15} {'Random wins':>12}")
    for row in result["maps"]:
        agent, random = row["agent"], row["random"]
        print(f"{row['maze']:<14} {agent['pellet_percent']:>13.2f}% {agent['win_percent']:>10.2f}% "
              f"{random['avg_pellet_percent']:>14.2f}% {random['win_percent']:>11.2f}%")
    print("NOTE: Maps differ in layout/difficulty; compare to each map's random baseline. "
          "New ghost seeds on the original maze are NOT a new topology.")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved results to: {a.output.resolve()}")


if __name__ == "__main__":
    main()
