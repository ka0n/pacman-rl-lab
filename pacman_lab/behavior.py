# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyze real greedy trajectories plus a controlled ghost-position intervention.

Counterfactual is a *single-observation* probe, not an altered rollout.
The agent only observes ghost POSITION, not direction/history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compare import load_model
from .core import PacmanGame
from .dqn import select_device

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_nearest_ghost_far(game: PacmanGame, observation: np.ndarray):
    """Relocate one ghost as far as possible, keeping other channels untouched.

    Preserves ghost count. Counterfactual positions are physically valid cells,
    but may not be reachable in one ghost move. Not a dynamics intervention.
    """
    if not game.ghosts:
        return None
    source = min(game.ghosts, key=lambda g: (manhattan(game.player, g), g))
    occupied = set(game.ghosts) - {source}
    positions = [(r, c) for r in range(game.height) for c in range(game.width)
                 if (r, c) not in game.walls and (r, c) not in occupied
                 and (r, c) != game.player and (r, c) != source]
    if not positions:
        return None
    destination = max(positions, key=lambda p: (manhattan(game.player, p), p))
    if manhattan(game.player, destination) <= manhattan(game.player, source):
        return None
    altered = observation.copy()
    altered[3, source[0], source[1]] = 0
    altered[3, destination[0], destination[1]] = 1
    return altered, source, destination


def run_behavior(checkpoint: str, episodes: int = 20, seed: int = 40000,
                 maze: str | None = None, examples: int = 8, device_name: str = "auto") -> dict:
    """Greedy behavior on real game rollouts, with ghost relocation Q comparison."""
    import torch

    if episodes < 1 or seed < 0 or examples < 0:
        raise ValueError("episodes >= 1, seed >= 0 and examples >= 0 required")
    device = select_device(device_name)
    model, payload = load_model(checkpoint, device)
    active_maze = maze or payload["maze"]
    game = PacmanGame(maze=active_maze, ghosts=2)
    if tuple(payload["input_shape"]) != game.observe().shape:
        raise ValueError("Unseen maze has incompatible observation shape")

    totals = {"near_steps": 0, "far_steps": 0, "near_reversals": 0,
              "far_reversals": 0, "near_interventions": 0,
              "far_interventions": 0, "near_action_changes": 0,
              "far_action_changes": 0, "wall_bumps": 0, "reversals": 0,
              "episodes_won": 0, "total_steps": 0}
    fractions = []
    examples_seen = []
    q_deltas = {"near": [], "far": []}

    with torch.inference_mode():
        for ep in range(episodes):
            game.reset(seed=seed + ep, ghosts=2, ghost_every=1)
            while not game.done:
                observation = game.observe()
                nearest_distance = min(manhattan(game.player, ghost) for ghost in game.ghosts)
                category = "near" if nearest_distance <= 2 else "far"
                totals[f"{category}_steps"] += 1
                relocated = move_nearest_ghost_far(game, observation)
                if relocated is not None:
                    altered, before, after = relocated
                    inputs = np.stack((observation, altered))
                else:
                    before = after = None
                    inputs = observation[None, ...]
                x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
                q_batch = model(x).detach().cpu().numpy()
                q = q_batch[0]
                action = int(q.argmax())
                cf_action = None
                delta = None
                changed = False
                if relocated is not None:
                    cf_q = q_batch[1]
                    cf_action = int(cf_q.argmax())
                    changed = action != cf_action
                    delta = float(np.max(np.abs(q - cf_q)))
                    totals[f"{category}_interventions"] += 1
                    totals[f"{category}_action_changes"] += int(changed)
                    q_deltas[category].append(delta)
                previous_reversals = game.reversals
                old_pos = game.player
                game.step(action)
                reversal = game.reversals > previous_reversals
                totals[f"{category}_reversals"] += int(reversal)
                if (reversal or changed) and len(examples_seen) < max(examples * 5, 20):
                    examples_seen.append({
                        "episode": ep, "step": game.steps, "player": list(old_pos),
                        "nearest_ghost_distance": nearest_distance,
                        "ghost_from": list(before) if before else None,
                        "ghost_to": list(after) if after else None,
                        "chosen": ACTIONS[action], "counterfactual_chosen":
                            ACTIONS[cf_action] if cf_action is not None else None,
                        "action_changed": changed, "reversal": reversal,
                        "q_values": [round(float(v), 3) for v in q],
                        "max_abs_q_delta": round(delta, 3) if delta is not None else None,
                    })
            totals["episodes_won"] += int(game.won)
            totals["wall_bumps"] += game.wall_bumps
            totals["reversals"] += game.reversals
            totals["total_steps"] += game.steps
            fractions.append(game.info()["collected_fraction"])

    def percentage(num: int, den: int) -> float | None:
        return (100 * num / den) if den else None

    prioritized = sorted(examples_seen, key=lambda x: (
        x["nearest_ghost_distance"] > 2, not x["action_changed"], not x["reversal"], x["episode"], x["step"]))
    result = {
        "checkpoint": checkpoint, "maze": active_maze, "episodes": episodes,
        "seed_start": seed, "mean_pellets_percent": float(100 * np.mean(fractions)),
        "win_percent": percentage(totals["episodes_won"], episodes),
        "wall_bumps_per_episode": totals["wall_bumps"] / episodes,
        "reversals_per_episode": totals["reversals"] / episodes,
        "near_ghost_distance": "Manhattan <= 2 cells (descriptive, not necessarily hazardous)",
        "near_steps": totals["near_steps"], "far_steps": totals["far_steps"],
        "near_reversal_percent": percentage(totals["near_reversals"], totals["near_steps"]),
        "far_reversal_percent": percentage(totals["far_reversals"], totals["far_steps"]),
        "near_cf_pairs": totals["near_interventions"],
        "far_cf_pairs": totals["far_interventions"],
        "near_cf_action_change_percent": percentage(totals["near_action_changes"], totals["near_interventions"]),
        "far_cf_action_change_percent": percentage(totals["far_action_changes"], totals["far_interventions"]),
        "near_mean_max_abs_q_delta": float(np.mean(q_deltas["near"])) if q_deltas["near"] else None,
        "far_mean_max_abs_q_delta": float(np.mean(q_deltas["far"])) if q_deltas["far"] else None,
        "examples": prioritized[:examples],
        "limitations": (
            "Descriptive association, not proof that reversals intentionally wait for ghosts. "
            "Counterfactual moves nearest ghost to distant free cell in ONE observation "
            "without running new ghost physics; position changes can be off-distribution. "
            "Network does not observe ghost direction/history."
        ),
    }
    return result


def fmt_percent(x):
    return "n/a" if x is None else f"{x:.1f}%"


def main():
    p = argparse.ArgumentParser(description="Inspect reversals and sensitivity to moving a nearby ghost")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=40000)
    p.add_argument("--maze", default=None)
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    result = run_behavior(a.checkpoint, a.episodes, a.seed, a.maze, a.examples, a.device)
    print(f"Real trajectories: {a.episodes} seeded games, maze={result['maze']}, "
          f"pellets={result['mean_pellets_percent']:.1f}% wins={result['win_percent']:.1f}%")
    print(f"Reversals/ep: {result['reversals_per_episode']:.1f} | wall bumps/ep: {result['wall_bumps_per_episode']:.1f}")
    print(f"Ghost <=2 tiles: {result['near_steps']} states | reversal {fmt_percent(result['near_reversal_percent'])} "
          f"| action changed if ghost relocated {fmt_percent(result['near_cf_action_change_percent'])}")
    print(f"Ghost >2 tiles:  {result['far_steps']} states | reversal {fmt_percent(result['far_reversal_percent'])} "
          f"| action changed if ghost relocated {fmt_percent(result['far_cf_action_change_percent'])}")
    for row in result["examples"]:
        print("Example:", row)
    print("LIMITATION: " + result["limitations"])
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved results to: {a.output.resolve()}")


if __name__ == "__main__":
    main()
