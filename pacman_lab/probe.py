# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired ghost-position intervention: does the policy react to ghost inputs?"""
from __future__ import annotations

import argparse
import random

import numpy as np

from .compare import load_model
from .core import PacmanGame
from .dqn import select_device


def moved_ghost_observation(game: PacmanGame, rng: random.Random):
    """Relocate one ghost in an observation while freezing all other inputs.

    This is a sensitivity test, *not* a measure of correct/optimal ghost evasion.
    """
    if not game.ghosts:
        return None
    moved = rng.randrange(len(game.ghosts))
    old = game.ghosts[moved]
    busy = set(game.ghosts) | {game.player}
    candidates = [(r, c) for r in range(game.height) for c in range(game.width)
                  if (r, c) not in game.walls and (r, c) not in busy]
    if not candidates:
        return None
    # Prefer a distant alternative location to test a meaningful change.
    destination = max(candidates, key=lambda p: (abs(p[0]-old[0])+abs(p[1]-old[1]),
                                                 rng.random()))
    obs = game.observe().copy()
    obs[3, old[0], old[1]] = 0
    obs[3, destination[0], destination[1]] = 1
    return obs, old, destination


def run_probe(checkpoint: str, samples: int = 200, seed: int = 30000,
              device_name: str = "auto") -> dict:
    import torch
    if samples <= 0:
        raise ValueError("samples must be positive")
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    game = PacmanGame(maze=payload["maze"], ghosts=2, seed=seed)
    rng = random.Random(seed)
    changes = 0
    deltas = []
    examples = []
    episodes = 0
    with torch.inference_mode():
        while len(deltas) < samples:
            if game.done:
                episodes += 1
                game.reset(seed=seed + episodes)
            altered = moved_ghost_observation(game, rng)
            if altered is None:
                raise RuntimeError("Cannot find alternative ghost position")
            other_obs, old, new = altered
            batch = np.stack((game.observe(), other_obs))
            q_values = model(torch.as_tensor(batch, dtype=torch.float32, device=device)).cpu().numpy()
            a, b = int(q_values[0].argmax()), int(q_values[1].argmax())
            delta = float(np.max(np.abs(q_values[0] - q_values[1])))
            changes += int(a != b)
            deltas.append(delta)
            if len(examples) < 4 and a != b:
                examples.append({"ghost_from": old, "ghost_to": new,
                                 "action_before": a, "action_after": b,
                                 "max_abs_q_delta": round(delta, 3)})
            game.step(a)
    return {"samples": samples, "episodes": episodes + 1,
            "action_change_percent": 100 * changes / samples,
            "mean_max_abs_q_delta": float(np.mean(deltas)),
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description="Measure policy sensitivity to moved ghost positions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    result = run_probe(args.checkpoint, samples=args.samples, seed=args.seed, device_name=args.device)
    print(f"Ghost-position intervention: {result['samples']} paired states")
    print(f"Action changed in {result['action_change_percent']:.1f}% of pairs")
    print(f"Mean max |Q-value change| = {result['mean_max_abs_q_delta']:.4f}")
    for row in result["examples"]:
        print("Example:", row)
    print("NOTE: Sensitivity alone does not establish optimal ghost avoidance.")


if __name__ == "__main__":
    main()
