# SPDX-License-Identifier: GPL-3.0-or-later
"""Pac-Man DQN experiment, with longer and configurable epsilon-greedy exploration."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .core import MAPS, PacmanGame
from .dqn import make_qnet, select_device


def stage(mode: str, fraction: float) -> tuple[int, int]:
    """Retain old curriculum mode for compatibility; use 'direct' in new experiments."""
    if mode == "direct":
        return 2, 1
    if mode != "curriculum":
        raise ValueError(f"invalid mode: {mode}")
    if fraction < 0.10:
        return 0, 1
    if fraction < 0.25:
        return 1, 2
    return 2, 1


def exploration_epsilon(local_step: int, total_steps: int, start: float = 1.0,
                        end: float = 0.10, decay_fraction: float = 0.80) -> float:
    """Linear epsilon schedule, using *local* training steps (not absolute count)."""
    if total_steps <= 0 or local_step < 1:
        raise ValueError("Steps must be positive")
    if not 0 <= end <= start <= 1:
        raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
    if not 0 < decay_fraction <= 1:
        raise ValueError("decay_fraction must be in (0, 1]")
    fraction = min(1., max(0., (local_step - 1) / max(1., total_steps * decay_fraction)))
    return start + (end - start) * fraction


def evaluate(model, device, maze: str, episodes: int = 20, seed: int = 10000) -> dict:
    """Greedy evaluation on fixed seeded games, with two normal-speed ghosts."""
    import torch

    if episodes < 1:
        raise ValueError("episodes must be positive")
    was_training = model.training
    model.eval()
    game = PacmanGame(maze=maze, ghosts=2, ghost_every=1)
    pellets, wins, wall_bumps, still, reversals, lengths = [], 0, [], [], [], []
    try:
        with torch.inference_mode():
            for ep in range(episodes):
                obs, _ = game.reset(seed=seed + ep, ghosts=2, ghost_every=1)
                while not game.done:
                    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(model(tensor).argmax(dim=1).item())
                    obs = game.step(action).observation
                pellets.append(game.info()["collected_fraction"])
                wins += int(game.won)
                wall_bumps.append(game.wall_bumps)
                still.append(game.stationary_steps)
                reversals.append(game.reversals)
                lengths.append(game.steps)
    finally:
        model.train(was_training)
    return {
        "pellet_percent": float(100 * np.mean(pellets)),
        "win_percent": float(100 * wins / episodes),
        "wall_bumps_per_episode": float(np.mean(wall_bumps)),
        "stationary_percent": float(100 * sum(still) / max(1, sum(lengths))),
        "reversals_per_episode": float(np.mean(reversals)),
        "steps_per_episode": float(np.mean(lengths)),
    }


def run_training(mode: str = "direct", steps: int = 300000,
                 maze: str = "small", seed: int = 42, device_name: str = "auto",
                 eval_every: int = 10000, output: str | None = None,
                 batch_size: int = 64, replay_capacity: int = 30000,
                 resume: str | None = None, eval_episodes: int = 20,
                 epsilon_start: float | None = None, epsilon_end: float = 0.10,
                 epsilon_decay_fraction: float = 0.80, eval_seed: int = 10000):
    """Train fresh or warm-start from a trusted checkpoint.

    Warm starts restore weights, target, optimizer (when available). They DO NOT
    restore replay buffer, RNG or partial episode: not bit-exact resume.
    """
    import torch
    import torch.nn.functional as F

    if mode not in ("direct", "curriculum"):
        raise ValueError("Invalid training mode")
    if steps <= 0 or eval_every <= 0 or eval_episodes <= 0:
        raise ValueError("steps, eval_every and eval_episodes must be positive")
    if batch_size <= 0 or replay_capacity < batch_size:
        raise ValueError("Invalid batch_size/replay_capacity")
    if epsilon_start is None:
        epsilon_start = 1.0 if resume is None else 0.40
    exploration_epsilon(1, steps, epsilon_start, epsilon_end, epsilon_decay_fraction)

    # Avoid silently evaluating on the held-out benchmark suite by default.
    if eval_seed < 0 or seed < 0:
        raise ValueError("seeds must be non-negative")
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    start_steps = 0
    previous_episodes = 0
    payload = None
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=True)
        if payload["maze"] != maze:
            raise ValueError(f"Checkpoint maze {payload['maze']!r} != requested {maze!r}")
        start_steps = int(payload["steps"])
        previous_episodes = int(payload.get("episodes", 0))
        print(f"Warm start: {resume} ({start_steps} previous steps). "
              "Replay, RNG and episode restart.", flush=True)
    end_steps = start_steps + steps
    print(f"Training {mode} on {device} | torch={torch.__version__} | "
          f"new steps={steps}, target total={end_steps} | "
          f"epsilon {epsilon_start:.2f}->{epsilon_end:.2f} "
          f"over {epsilon_decay_fraction:.0%} of new steps", flush=True)
    run_dir = (Path(output) if output else
               Path("runs") / f"{datetime.now():%Y%m%d_%H%M%S}_{mode}_{seed}_v03")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output folder already contains files: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {"version": "0.3", "mode": mode, "maze": maze, "seed": seed,
              "training_steps_new": steps, "previous_steps": start_steps,
              "eval_every": eval_every, "eval_episodes": eval_episodes,
              "eval_seed": eval_seed, "batch_size": batch_size,
              "replay_capacity": replay_capacity, "epsilon_start": epsilon_start,
              "epsilon_end": epsilon_end, "epsilon_decay_fraction": epsilon_decay_fraction,
              "resume_from": resume or "", "best_selection": "pellet_percent",
              "training_seed_base": 1_000_000 + seed + start_steps}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    game = PacmanGame(maze=maze, ghosts=2)
    state_shape = game.observe().shape
    online = make_qnet(state_shape).to(device)
    target = make_qnet(state_shape).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=0.0003)
    if payload is not None:
        if tuple(payload["input_shape"]) != state_shape:
            raise ValueError("Incompatible checkpoint observation shape")
        online.load_state_dict(payload["model"])
        target.load_state_dict(payload.get("target", payload["model"]))
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    else:
        target.load_state_dict(online.state_dict())
    target.eval()
    replay = deque(maxlen=replay_capacity)
    ghost_count, ghost_every = stage(mode, start_steps / end_steps)
    episode = previous_episodes
    # Keep training seeds disjoint from validation (10000+) and held-out tests (20000+).
    train_seed_base = 1_000_000 + seed + start_steps
    state, _ = game.reset(seed=train_seed_base + episode, ghosts=ghost_count,
                          ghost_every=ghost_every)
    best_eval = -1.0
    fields = ["step", "episodes", "epsilon", "pellet_percent", "win_percent",
              "wall_bumps_per_episode", "stationary_percent", "reversals_per_episode",
              "steps_per_episode", "best_pellet_percent"]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for local_step in range(1, steps + 1):
            t = start_steps + local_step
            epsilon = exploration_epsilon(local_step, steps, epsilon_start,
                                          epsilon_end, epsilon_decay_fraction)
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                with torch.no_grad():
                    x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online(x).argmax(dim=1).item())
            transition = game.step(action)
            replay.append((state, action, transition.reward, transition.observation,
                           transition.terminated or transition.truncated))
            state = transition.observation
            if game.done:
                episode += 1
                ghost_count, ghost_every = stage(mode, t / end_steps)
                state, _ = game.reset(seed=train_seed_base + episode,
                                      ghosts=ghost_count, ghost_every=ghost_every)
            if local_step >= 500 and local_step % 4 == 0 and len(replay) >= batch_size:
                batch = rng.sample(replay, batch_size)
                states, actions, rewards, next_states, terminated = zip(*batch)
                s = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
                a = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                ns = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
                ends = torch.as_tensor(terminated, dtype=torch.float32, device=device)
                current_q = online(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    target_q = r + (1.0 - ends) * 0.99 * target(ns).max(dim=1).values
                loss = F.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
            if t % 1000 == 0:
                target.load_state_dict(online.state_dict())
            if local_step % eval_every == 0 or local_step == steps:
                metrics = evaluate(online, device, maze=maze, episodes=eval_episodes,
                                   seed=eval_seed)
                score = metrics["pellet_percent"]
                new_best = score > best_eval
                best_eval = max(score, best_eval)
                checkpoint = {"model": online.state_dict(), "target": target.state_dict(),
                              "optimizer": optimizer.state_dict(), "maze": maze,
                              "input_shape": state_shape, "steps": t, "episodes": episode,
                              "seed": seed, "mode": mode, "metrics": metrics,
                              "version": "0.3", "resume_from": resume or "",
                              "epsilon": epsilon, "training_config": config}
                # Preserve a usable last.pt after an interrupted long run.
                temp = run_dir / "last.tmp"
                torch.save(checkpoint, temp)
                temp.replace(run_dir / "last.pt")
                if new_best:
                    temp = run_dir / "best.tmp"
                    torch.save(checkpoint, temp)
                    temp.replace(run_dir / "best.pt")
                writer.writerow({"step": t, "episodes": episode, "epsilon": epsilon,
                                 **metrics, "best_pellet_percent": best_eval})
                file.flush()
                print(f"{t:>7}/{end_steps} | episodes={episode:>5} eps={epsilon:.2f} "
                      f"pellets={score:5.1f}% wins={metrics['win_percent']:4.1f}% "
                      f"wall/ep={metrics['wall_bumps_per_episode']:.1f} "
                      f"stationary={metrics['stationary_percent']:.1f}%", flush=True)
    print(f"Saved checkpoints and metrics to: {run_dir.resolve()}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="DQN direct training with controlled exploration")
    parser.add_argument("--mode", choices=("direct", "curriculum"), default="direct",
                        help="Use direct; curriculum is retained for compatibility")
    parser.add_argument("--steps", type=int, default=300000,
                        help="New steps; with --resume these are additional steps")
    parser.add_argument("--maze", choices=tuple(MAPS), default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Default: 1.0 from scratch, 0.4 for warm start")
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Warm start from trusted v0.1/v0.2/v0.3 checkpoint")
    args = parser.parse_args()
    run_training(mode=args.mode, steps=args.steps, maze=args.maze,
                 seed=args.seed, device_name=args.device, eval_every=args.eval_every,
                 output=args.output, resume=args.resume, eval_episodes=args.eval_episodes,
                 epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                 epsilon_decay_fraction=args.epsilon_decay_fraction,
                 eval_seed=args.eval_seed)


if __name__ == "__main__":
    main()
