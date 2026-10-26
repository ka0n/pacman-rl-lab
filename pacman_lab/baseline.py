# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducible random-agent baseline without graphics or torch."""
import argparse
import random
import statistics

from .core import MAPS, PacmanGame


def evaluate_random(episodes: int = 100, maze: str = "small", ghosts: int = 2,
                    seed: int = 123) -> dict:
    policy_rng = random.Random(seed)
    game = PacmanGame(maze=maze, ghosts=ghosts)
    fractions, wins, rewards = [], 0, []
    bumps, still, lengths, reversals = [], [], [], []
    for ep in range(episodes):
        game.reset(seed=seed + ep)
        total_reward = 0.0
        while not game.done:
            # Random choice among ALL 4 actions, including blocked actions.
            outcome = game.step(policy_rng.randrange(4))
            total_reward += outcome.reward
        fractions.append(game.info()["collected_fraction"])
        wins += int(game.won)
        rewards.append(total_reward)
        bumps.append(game.wall_bumps)
        still.append(game.stationary_steps)
        lengths.append(game.steps)
        reversals.append(game.reversals)
    return {"episodes": episodes, "avg_pellet_percent": 100 * statistics.mean(fractions),
            "win_percent": 100 * wins / episodes,
            "avg_reward": statistics.mean(rewards),
            "wall_bumps_per_episode": statistics.mean(bumps),
            "stationary_percent": 100 * sum(still) / max(1, sum(lengths)),
            "reversals_per_episode": statistics.mean(reversals),
            "steps_per_episode": statistics.mean(lengths)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate an untrained random Pac-Man agent")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--maze", choices=tuple(MAPS), default="small")
    parser.add_argument("--ghosts", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    result = evaluate_random(**vars(args))
    for name, value in result.items():
        print(f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
