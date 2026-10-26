# SPDX-License-Identifier: GPL-3.0-or-later
"""Long-running, checkpointable DQN experiments with optional frame memory.

This deliberately uses the same 128x128 MLP/reward/game as legacy v0.3.
Double DQN changes only target-action selection. Truncation is terminal for
replay targets. A resumed run restarts replay/RNG/episode (not bit-exact).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import deque
from pathlib import Path

import numpy as np

from .core import PacmanGame
from .dqn import make_qnet, select_device
from .memory import FrameHistory
from .train import exploration_epsilon

VARIANTS = {"plain_dqn": (1, False), "memory_dqn": (3, False),
            "memory_double": (3, True)}
FIELDS = ["step", "episodes", "epsilon", "pellet_percent", "win_percent",
          "wall_bumps_per_episode", "stationary_percent", "reversals_per_episode",
          "steps_per_episode", "best_win_percent", "best_pellet_percent"]


def evaluate_agent(model, device, maze="small", episodes=20, seed=10000, frames=1):
    import torch
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    was_training = model.training
    model.eval()
    game = PacmanGame(maze=maze, ghosts=2, ghost_every=1)
    history = FrameHistory(frames)
    pellets, wins, walls, stationary, reversals, lengths = [], 0, [], [], [], []
    try:
        with torch.inference_mode():
            for ep in range(episodes):
                obs, _ = game.reset(seed=seed + ep, ghosts=2, ghost_every=1)
                state = history.reset(obs)
                while not game.done:
                    tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(model(tensor).argmax(1).item())
                    transition = game.step(action)
                    state = history.step(transition.observation)
                info = game.info()
                pellets.append(info["collected_fraction"])
                wins += int(game.won)
                walls.append(game.wall_bumps)
                stationary.append(game.stationary_steps)
                reversals.append(game.reversals)
                lengths.append(game.steps)
    finally:
        model.train(was_training)
    return {"pellet_percent": float(np.mean(pellets) * 100),
            "win_percent": 100.0 * wins / episodes,
            "wins": wins, "episodes": episodes,
            "wall_bumps_per_episode": float(np.mean(walls)),
            "stationary_percent": 100.0 * sum(stationary) / max(1, sum(lengths)),
            "reversals_per_episode": float(np.mean(reversals)),
            "steps_per_episode": float(np.mean(lengths))}


def checkpoint_read(checkpoint: str | Path, device):
    import torch
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    return payload


def atomic_save_torch(payload, path: Path):
    import torch
    temp = path.with_suffix(".partial")
    torch.save(payload, temp)
    os.replace(temp, path)


def resume_best_score(checkpoint, output: Path, device, source=None):
    """Reconcile best files and recover evaluable last saves from the old order.

    An external warm restart copies the best model to its new output directory.
    Only saved evaluation evidence can identify last.pt itself as a best model.
    """
    score = tuple(checkpoint.get("best_score", (-1.0, -1.0)))
    best_path = output / "best.pt"
    paths = [best_path]
    if source is not None:
        sibling = Path(source).parent / "best.pt"
        if sibling.resolve() != best_path.resolve():
            paths.append(sibling)
    best, best_score, output_score = None, (-1.0, -1.0), (-1.0, -1.0)
    config_key = "marathon_config" if "marathon_config" in checkpoint else "config"
    for candidate in paths:
        if not candidate.exists():
            continue
        saved = checkpoint_read(candidate, device)
        if saved.get(config_key) != checkpoint.get(config_key):
            raise ValueError("Best checkpoint belongs to a different training configuration")
        candidate_score = tuple(saved.get("best_score", (-1.0, -1.0)))
        if candidate == best_path:
            output_score = candidate_score
        if candidate_score > best_score:
            best, best_score = saved, candidate_score
    metrics = checkpoint.get("metrics", {})
    evaluated_score = (metrics.get("win_percent"), metrics.get("pellet_percent"))
    if score == evaluated_score and score > best_score:
        best, best_score = checkpoint, score
    if best_score < score:
        raise ValueError("Cannot recover best model: saved best-score weights are unavailable")
    if best is not None and best_score > output_score:
        atomic_save_torch(best, best_path)
    return max(score, best_score)


def run_marathon_training(variant: str, steps: int, seed: int, output: str | Path,
                          device_name="auto", maze="small", eval_every=20000,
                          eval_episodes=20, save_every=10000, batch_size=64,
                          replay_capacity=30000, epsilon_start=1.0,
                          epsilon_end=0.10, epsilon_decay_fraction=0.80,
                          resume: str | None = None, learning_rate=0.0003,
                          train_warmup=500):
    """Train until absolute step target; --resume is a warm restart from last.pt."""
    import torch
    import torch.nn.functional as F

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    if steps < 1 or eval_every < 1 or eval_episodes < 1 or save_every < 1:
        raise ValueError("step and interval arguments must be positive")
    if batch_size < 1 or replay_capacity < batch_size or train_warmup < 0:
        raise ValueError("Invalid buffer or warmup")
    exploration_epsilon(1, steps, epsilon_start, epsilon_end, epsilon_decay_fraction)
    frames, double = VARIANTS[variant]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    game = PacmanGame(maze=maze, ghosts=2)
    history = FrameHistory(frames)
    shape = history.reset(game.observe()).shape
    online, target = make_qnet(shape).to(device), make_qnet(shape).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    start_step, episodes = 0, 0
    best_score = (-1.0, -1.0)
    if resume:
        saved = checkpoint_read(resume, device)
        if (saved.get("variant") != variant or saved.get("maze") != maze
                or int(saved.get("seed", -1)) != seed
                or tuple(saved["input_shape"]) != shape):
            raise ValueError("Resume checkpoint is incompatible with variant, maze, seed or shape")
        start_step = int(saved["steps"])
        episodes = int(saved.get("episodes_total", saved.get("episodes", 0)))
        if start_step >= steps:
            resume_best_score(saved, output, device, resume)
            return output
        old_config = saved.get("marathon_config", {})
        for name, requested in (("steps", steps), ("epsilon_start", epsilon_start),
                                ("epsilon_end", epsilon_end),
                                ("epsilon_decay_fraction", epsilon_decay_fraction)):
            if old_config.get(name) != requested:
                raise ValueError(f"Resume parameter changed: {name}")
        online.load_state_dict(saved["model"])
        target.load_state_dict(saved.get("target", saved["model"]))
        optimizer.load_state_dict(saved["optimizer"])
        best_score = resume_best_score(saved, output, device, resume)
        print(f"Warm restart at {start_step}/{steps}; replay/RNG/episode reset", flush=True)
    else:
        if (output / "last.pt").exists() or (output / "metrics.csv").exists():
            raise FileExistsError(f"Refusing to overwrite run at {output}")
        target.load_state_dict(online.state_dict())
    target.eval()
    config = {"version": "0.5.0", "variant": variant, "steps": steps,
              "seed": seed, "maze": maze, "frames": frames, "double": double,
              "eval_every": eval_every, "save_every": save_every,
              "eval_episodes": eval_episodes, "batch_size": batch_size,
              "replay_capacity": replay_capacity, "epsilon_start": epsilon_start,
              "epsilon_end": epsilon_end, "epsilon_decay_fraction": epsilon_decay_fraction,
              "learning_rate": learning_rate, "train_warmup": train_warmup}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    replay = deque(maxlen=replay_capacity)
    # Training seeds are isolated from 10k validation, 20k final and 40k transfer.
    base = 1_000_000 + seed + start_step
    state = history.reset(game.reset(seed=base + episodes)[0])
    metrics_file = output / "metrics.csv"
    new_file = not metrics_file.exists()
    print(f"{variant} on {device} | steps {start_step}->{steps} | {frames} frames | "
          f"{'Double DQN' if double else 'DQN'}", flush=True)
    with metrics_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for t in range(start_step + 1, steps + 1):
            epsilon = exploration_epsilon(t, steps, epsilon_start,
                                          epsilon_end, epsilon_decay_fraction)
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                with torch.inference_mode():
                    obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online(obs_tensor).argmax(1).item())
            result = game.step(action)
            next_state = history.step(result.observation)
            done = result.terminated or result.truncated
            replay.append((state, action, result.reward, next_state, done))
            state = next_state
            if done:
                episodes += 1
                state = history.reset(game.reset(seed=base + episodes)[0])
            # Minibatches on GPU; small networks: more games can be CPU-bound.
            if t % 4 == 0 and len(replay) >= batch_size and len(replay) >= train_warmup:
                batch = rng.sample(replay, batch_size)
                observations, actions, rewards, next_obs, terminal = zip(*batch)
                s = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
                ns = torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=device)
                a = torch.as_tensor(actions, dtype=torch.long, device=device)[:, None]
                r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                d = torch.as_tensor(terminal, dtype=torch.float32, device=device)
                q = online(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    if double:
                        next_actions = online(ns).argmax(1, keepdim=True)
                        v = target(ns).gather(1, next_actions).squeeze(1)
                    else:
                        v = target(ns).max(1).values
                    y = r + (1.0 - d) * 0.99 * v
                loss = F.smooth_l1_loss(q, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
            if t % 1000 == 0:
                target.load_state_dict(online.state_dict())
            should_eval = (t % eval_every == 0 or t == steps)
            should_save = (t % save_every == 0 or t == steps or should_eval)
            metrics = None
            if should_eval:
                metrics = evaluate_agent(online, device, maze, eval_episodes, 10000, frames)
                score = (metrics["win_percent"], metrics["pellet_percent"])
                improved = score > best_score
                if improved:
                    best_score = score
                writer.writerow({"step": t, "episodes": episodes, "epsilon": epsilon,
                                 **{k: metrics[k] for k in FIELDS if k in metrics},
                                 "best_win_percent": best_score[0],
                                 "best_pellet_percent": best_score[1]})
                csv_file.flush()
                print(f"{t:>8}/{steps} | {variant:<13} eps={epsilon:.2f} "
                      f"pellets={score[1]:5.1f}% wins={score[0]:4.1f}% "
                      f"wall={metrics['wall_bumps_per_episode']:.1f}", flush=True)
            if should_save:
                payload = {"model": online.state_dict(), "target": target.state_dict(),
                           "optimizer": optimizer.state_dict(), "input_shape": shape,
                           "maze": maze, "steps": t, "episodes": episodes,
                           "episodes_total": episodes, "seed": seed,
                           "variant": variant, "frames": frames,
                           "version": "0.5.0", "marathon_config": config,
                           "best_score": best_score,
                           "metrics": metrics or {}}
                if should_eval and improved:
                    atomic_save_torch(payload, output / "best.pt")
                atomic_save_torch(payload, output / "last.pt")
    return output


def main():
    p = argparse.ArgumentParser(description="Train one new Pac-Man agent, long-run capable")
    p.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=str)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    a = p.parse_args()
    run_marathon_training(variant=a.variant, steps=a.steps, seed=a.seed,
                          output=a.output, resume=a.resume,
                          eval_every=a.eval_every, save_every=a.save_every,
                          eval_episodes=a.eval_episodes, device_name=a.device)


if __name__ == "__main__":
    main()
