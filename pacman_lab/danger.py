# SPDX-License-Identifier: GPL-3.0-or-later
"""Controlled, synthetic one-step ghost-risk probe: not a survival benchmark."""
from __future__ import annotations

import argparse
import random

import numpy as np

from .compare import load_model
from .core import DIRECTIONS, PacmanGame
from .dqn import select_device


def _threatened_state(game: PacmanGame, rng: random.Random):
    """Make an observation with one ghost in an adjacent legal tile.

    The second ghost is not adjacent. All remaining planes stay unmodified.
    Only positions with at least two legal choices are accepted.
    """
    legal = game.legal_destinations(game.player)
    if len(legal) < 2:
        return None
    threatened_action, first_ghost = rng.choice(legal)
    far_tiles = [(r, c) for r in range(game.height) for c in range(game.width)
                 if (r, c) not in game.walls and
                 abs(r - game.player[0]) + abs(c - game.player[1]) > 1]
    if not far_tiles:
        return None
    second_ghost = rng.choice(far_tiles)
    observation = game.observe().copy()
    observation[3, first_ghost[0], first_ghost[1]] = 1
    observation[3, second_ghost[0], second_ghost[1]] = 1
    return observation, threatened_action, {a for a, _ in legal}


def run_danger_probe(checkpoint: str, samples: int = 200, seed: int = 31000,
                     device_name: str = "auto") -> dict:
    """Does a greedy agent immediately walk onto an adjacent stationary ghost?

    This measures only *one-step* choice in synthetic observations; actual ghosts
    move and can collide after Pac-Man moves, so this is NOT a win-rate estimate.
    """
    import torch

    if samples < 1 or seed < 0:
        raise ValueError("samples must be positive and seed nonnegative")
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    game = PacmanGame(maze=payload["maze"], ghosts=0)
    rng = random.Random(seed)
    dangerous, blocked, safe = 0, 0, 0
    random_expected_safe = 0.0
    with torch.inference_mode():
        for i in range(samples):
            game.reset(seed=seed + i)
            # Generate pellet/player variations with a random legal walk.
            for _ in range(rng.randrange(0, 60)):
                if game.done:
                    break
                legal = game.legal_destinations(game.player)
                if not legal:
                    break
                game.step(rng.choice(legal)[0])
            candidate = _threatened_state(game, rng)
            # If player ended in a dead end, reset to known open initial cell.
            if candidate is None:
                game.reset(seed=seed + i)
                candidate = _threatened_state(game, rng)
            if candidate is None:
                raise RuntimeError("Maze has no valid threatened state")
            observation, threatened_action, legal_actions = candidate
            x = torch.as_tensor(observation, dtype=torch.float32,
                                device=device).unsqueeze(0)
            action = int(model(x).argmax(dim=1).item())
            dangerous += int(action == threatened_action)
            blocked += int(action not in legal_actions)
            safe += int(action in legal_actions and action != threatened_action)
            random_expected_safe += (len(legal_actions) - 1) / 4
    return {"samples": samples,
            "ghost_step_percent": 100 * dangerous / samples,
            "wall_step_percent": 100 * blocked / samples,
            "safe_legal_step_percent": 100 * safe / samples,
            "random_expected_safe_legal_percent": 100 * random_expected_safe / samples,
            "random_expected_ghost_step_percent": 25.0}


def main():
    p = argparse.ArgumentParser(description="Probe immediate danger reaction, not actual survival")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=31000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = p.parse_args()
    result = run_danger_probe(args.checkpoint, args.samples, args.seed, args.device)
    print(f"Synthetic immediate-danger probe: {result['samples']} states")
    print(f"Agent: safe legal {result['safe_legal_step_percent']:.1f}% | "
          f"direct collision {result['ghost_step_percent']:.1f}% | "
          f"wall {result['wall_step_percent']:.1f}%")
    print(f"Uniform-random expectation: safe legal "
          f"{result['random_expected_safe_legal_percent']:.1f}% | "
          f"direct collision {result['random_expected_ghost_step_percent']:.1f}%")
    print("NOTE: This tests one-step choices in synthetic states only; moving ghosts, "
          "full-game survival and long-term strategy are not measured.")


if __name__ == "__main__":
    main()
