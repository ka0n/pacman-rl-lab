# SPDX-License-Identifier: GPL-3.0-or-later
"""Train CNN / frame-history / recurrent GRU Double DQN.

Training uses a small semantic grid. Recurrent replay samples episode-contained
sequences (burn-in + unroll) and never crosses terminal boundaries.
Interrupted training is a warm restart, not bit-exact replay restoration.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from .cnn_models import SPECS, make_model, parameters_count
from .core import PacmanGame
from .dqn import select_device
from .cnn_motion import observer_for
from .train import exploration_epsilon
from .marathon_train import atomic_save_torch, checkpoint_read, resume_best_score

FIELDS = ["step", "episodes", "epsilon", "pellet_percent", "win_percent",
          "wall_bumps_per_episode", "stationary_percent", "reversals_per_episode",
          "steps_per_episode", "train_loss", "steps_per_second",
          "best_win_percent", "best_pellet_percent"]


class EpisodeReplay:
    """Sample completed episodes, or bounded tails of overlong live episodes.

    Prefix eviction never introduces a terminal or world boundary inside a
    sequence. Every sample still includes the full requested burn-in + unroll.
    """
    def __init__(self, capacity: int, sequence_length: int):
        if sequence_length < 1 or capacity < sequence_length:
            raise ValueError("Replay capacity must hold a positive sequence length")
        self.capacity = capacity
        self.episode_capacity = capacity
        self.sequence_length = sequence_length
        self.episodes: deque[list] = deque()
        self.current: list = []
        self.total = 0
        self.eligible: list[list] = []

    def append(self, item, done: bool):
        self.current = self._append_episode(self.current, item, done)

    def _inflight(self):
        return [(self.current, self.episode_capacity)]

    def _append_episode(self, current, item, done, limit=None):
        limit = self.episode_capacity if limit is None else limit
        if len(current) == limit:
            del current[0]
            self.total -= 1
        current.append(item)
        self.total += 1
        if done:
            completed, current = current, []
            self.episodes.append(completed)
            if len(completed) >= self.sequence_length:
                self.eligible.append(completed)
        while self.total > self.capacity and self.episodes:
            gone = self.episodes.popleft()
            self.total -= len(gone)
            if len(gone) >= self.sequence_length:
                assert self.eligible.pop(0) is gone
        return current

    @property
    def can_sample(self):
        return bool(self.eligible) or any(
            len(ep) == limit for ep, limit in self._inflight())

    def sample(self, rng: random.Random, batch_size: int):
        eligible = self.eligible + [ep for ep, limit in self._inflight()
                                    if len(ep) == limit]
        if not eligible:
            raise ValueError("No episodes long enough for sequence replay")
        segments = []
        for _ in range(batch_size):
            ep = rng.choice(eligible)
            start = rng.randrange(len(ep) - self.sequence_length + 1)
            segments.append(ep[start:start + self.sequence_length])
        return segments


def eval_model(model, architecture: str, device, *, maze="small", episodes=40, seed=10000, engine="python"):
    import torch
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    spec = SPECS[architecture]
    was_training = model.training
    model.eval()
    pellets, wins, walls, stays, reversals, lengths = [], 0, [], [], [], []
    if engine == "rust":
        from .native import NativePacmanGame
        game = NativePacmanGame(maze=maze, ghosts=2)
    else:
        game = PacmanGame(maze=maze, ghosts=2)
    frames = observer_for(architecture, spec.frames)
    try:
        with torch.inference_mode():
            for n in range(episodes):
                obs, _ = game.reset(seed=seed + n, ghosts=2, ghost_every=1)
                state = frames.reset(obs)
                hidden = None
                while not game.done:
                    v = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                    if spec.recurrent:
                        q, hidden = model(v.unsqueeze(1), hidden)
                        action = int(q[0, -1].argmax().item())
                    else:
                        action = int(model(v).argmax(1).item())
                    state = frames.step(game.step(action).observation)
                info = game.info()
                pellets.append(info["collected_fraction"])
                wins += int(game.won)
                walls.append(game.wall_bumps)
                stays.append(game.stationary_steps)
                reversals.append(game.reversals)
                lengths.append(game.steps)
    finally:
        model.train(was_training)
        if engine == "rust": game.close()
    return {"pellet_percent": round(float(np.mean(pellets)) * 100, 4),
            "win_percent": round(100 * wins / episodes, 4),
            "wins": wins, "episodes": episodes,
            "wall_bumps_per_episode": float(np.mean(walls)),
            "stationary_percent": 100 * sum(stays) / max(1, sum(lengths)),
            "reversals_per_episode": float(np.mean(reversals)),
            "steps_per_episode": float(np.mean(lengths))}


def minibatch_loss(online, target, samples, architecture, device, burnin=4):
    """One Double-DQN minibatch loss; differentiable for GRU unroll."""
    import torch
    import torch.nn.functional as F
    spec = SPECS[architecture]
    if spec.recurrent:
        # [B,L,observation] plus the next state for the last transition.
        seq = np.stack([[x[0] for x in episode] + [episode[-1][3]]
                        for episode in samples])
        a = torch.as_tensor(np.array([[x[1] for x in ep] for ep in samples]),
                            device=device, dtype=torch.long)
        r = torch.as_tensor(np.array([[x[2] for x in ep] for ep in samples]),
                            device=device, dtype=torch.float32)
        d = torch.as_tensor(np.array([[x[4] for x in ep] for ep in samples]),
                            device=device, dtype=torch.float32)
        t = torch.as_tensor(seq, device=device, dtype=torch.float32)
        q_all, _ = online(t)
        q = q_all[:, burnin:-1].gather(2, a[:, burnin:].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            qt_all, _ = target(t)
            next_actions = q_all[:, burnin+1:].detach().argmax(dim=2, keepdim=True)
            future = qt_all[:, burnin+1:].gather(2, next_actions).squeeze(2)
            y = r[:, burnin:] + (1.0 - d[:, burnin:]) * 0.99 * future
        return F.smooth_l1_loss(q, y)
    obs = torch.as_tensor(np.stack([item[0] for item in samples]),
                          device=device, dtype=torch.float32)
    ns = torch.as_tensor(np.stack([item[3] for item in samples]),
                         device=device, dtype=torch.float32)
    a = torch.as_tensor([item[1] for item in samples], device=device, dtype=torch.long)
    r = torch.as_tensor([item[2] for item in samples], device=device, dtype=torch.float32)
    d = torch.as_tensor([item[4] for item in samples], device=device, dtype=torch.float32)
    q = online(obs).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = online(ns).argmax(1, keepdim=True)
        future = target(ns).gather(1, next_actions).squeeze(1)
        y = r + (1.0 - d) * 0.99 * future
    return F.smooth_l1_loss(q, y)


def run_training(*, architecture: str, steps: int, seed: int, output: Path | str,
                 device_name="auto", eval_every=50000, save_every=25000,
                 eval_episodes=40, replay_capacity=30000, batch_size=64,
                 warmup=2000, train_every=8, target_every=2000,
                 burnin=4, unroll=8, learning_rate=0.00015,
                 epsilon_start=1.0, epsilon_end=0.08,
                 epsilon_decay_fraction=0.85, resume: str | None=None,
                 maze: str = "large", engine: str = "python"):
    import torch
    if engine not in ("python", "rust"):
        raise ValueError("engine must be python or rust")
    if architecture not in SPECS:
        raise ValueError(f"Unknown architecture {architecture}")
    if min(steps, eval_every, save_every, eval_episodes, replay_capacity,
           batch_size, train_every, target_every, unroll) <= 0 or burnin < 0 or warmup < 0:
        raise ValueError("Training budgets must be positive and burn-in/warmup nonnegative")
    if batch_size > replay_capacity or not (0 < epsilon_end <= epsilon_start <= 1):
        raise ValueError("Invalid replay capacity or epsilon bounds")
    if not 0 < epsilon_decay_fraction <= 1:
        raise ValueError("Epsilon decay fraction must be in (0,1]")
    spec = SPECS[architecture]
    if spec.recurrent and replay_capacity < burnin + unroll:
        raise ValueError("Replay capacity must hold burn-in + unroll")
    if spec.recurrent and warmup > replay_capacity:
        raise ValueError("Recurrent warmup must not exceed replay capacity")
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if engine == "rust":
        from .native import NativePacmanGame
        game = NativePacmanGame(maze=maze, ghosts=2)
    else:
        game = PacmanGame(maze=maze, ghosts=2)
    history = observer_for(architecture, spec.frames)
    current = history.reset(game.observe())
    shape = tuple(current.shape)
    online = make_model(architecture, shape).to(device)
    target = make_model(architecture, shape).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    config = {"version":"0.7.0", "architecture":architecture,
              "steps":steps,"seed":seed, "maze":maze, "engine":engine, "frames":spec.frames,
              "recurrent":spec.recurrent,"double_dqn":True,
              "parameters":parameters_count(online),"eval_every":eval_every,
              "save_every":save_every,"eval_episodes":eval_episodes,
              "replay_capacity":replay_capacity,"batch_size":batch_size,
              "warmup":warmup,"train_every":train_every,"target_every":target_every,
              "burnin":burnin,"unroll":unroll,"learning_rate":learning_rate,
              "epsilon_start":epsilon_start,"epsilon_end":epsilon_end,
              "epsilon_decay_fraction":epsilon_decay_fraction}
    start_step = 0
    episodes = 0
    best_score = (-1.0, -1.0)
    if resume:
        checkpoint = checkpoint_read(resume, device)
        oldcfg = checkpoint.get("config", {})
        if any(oldcfg.get(k) != value for k,value in config.items()):
            raise ValueError("Resume requires identical training hyperparameters and model")
        if tuple(checkpoint["input_shape"]) != shape or checkpoint["architecture"] != architecture:
            raise ValueError("Checkpoint shape or architecture mismatch")
        start_step = int(checkpoint["step"])
        episodes = int(checkpoint["episodes"])
        best_score = resume_best_score(checkpoint, path, device, resume)
        online.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"WARM RESTART at {start_step}: replay buffer, RNG and episode reset",flush=True)
    else:
        if (path/"last.pt").exists() or (path/"metrics.csv").exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        target.load_state_dict(online.state_dict())
    if start_step >= steps:
        return path
    (path/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    target.eval()
    replay = EpisodeReplay(replay_capacity, burnin+unroll) if spec.recurrent else deque(maxlen=replay_capacity)
    # Never share training seeds with held-out (20000+, 40000+) or validation 10000+.
    episode_base = 1_000_000 + seed + start_step
    state = history.reset(game.reset(seed=episode_base + episodes)[0])
    hidden = None
    last_losses = deque(maxlen=500)
    start_time = time.perf_counter()
    metrics_path = path / "metrics.csv"
    if resume and metrics_path.exists():
        # A CSV evaluation may have been flushed before the matching checkpoint.
        with metrics_path.open(newline="",encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if int(row["step"]) <= start_step]
        with metrics_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    new_csv = not metrics_path.exists()
    print(f"{architecture} on {device} | {parameters_count(online):,} trainable params | "
          f"{start_step}->{steps} steps | GRU={spec.recurrent}", flush=True)
    with metrics_path.open("a",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=FIELDS)
        if new_csv:writer.writeheader()
        for t in range(start_step+1,steps+1):
            epsilon=exploration_epsilon(t,steps,epsilon_start,epsilon_end,epsilon_decay_fraction)
            if spec.recurrent:
                with torch.inference_mode():
                    v=torch.as_tensor(state,device=device,dtype=torch.float32)[None,None]
                    q,hidden=online(v,hidden)
                    greedy=int(q[0,-1].argmax().item()) if rng.random() >= epsilon else None
                action=greedy if greedy is not None else rng.randrange(4)
            else:
                if rng.random() < epsilon:
                    action=rng.randrange(4)
                else:
                    with torch.inference_mode():
                        v=torch.as_tensor(state,device=device,dtype=torch.float32).unsqueeze(0)
                        action=int(online(v).argmax(1).item())
            outcome=game.step(action)
            ns=history.step(outcome.observation)
            done=bool(outcome.terminated or outcome.truncated)
            # uint8 states, terminal flags, no hidden ground-truth ghost direction.
            item=(state.copy(),action,float(outcome.reward),ns.copy(),done)
            if spec.recurrent:
                replay.append(item,done)
            else:
                replay.append(item)
            state=ns
            if done:
                episodes+=1
                state=history.reset(game.reset(seed=episode_base + episodes)[0])
                hidden=None
            if t % train_every == 0 and (t-start_step)>=warmup:
                if spec.recurrent:
                    can_train=replay.total >= max(warmup,batch_size) and replay.can_sample
                else:
                    can_train=len(replay)>=max(warmup,batch_size)
                if can_train:
                    samples=(replay.sample(rng,batch_size) if spec.recurrent
                             else rng.sample(replay,batch_size))
                    loss=minibatch_loss(online,target,samples,architecture,device,burnin)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(),10.0)
                    optimizer.step()
                    last_losses.append(float(loss.detach().item()))
            if t % target_every == 0:
                target.load_state_dict(online.state_dict())
            evaluate=(t % eval_every == 0 or t==steps)
            save=(t % save_every == 0 or evaluate)
            metrics=None
            if evaluate:
                metrics=eval_model(online,architecture,device,maze=maze,episodes=eval_episodes,seed=10000,engine=engine)
                score=(metrics["win_percent"],metrics["pellet_percent"])
                improved=score>best_score
                if improved:best_score=score
                record={"step":t,"episodes":episodes,"epsilon":epsilon,
                        **{k:metrics[k] for k in metrics if k in FIELDS},
                        "train_loss":float(np.mean(last_losses)) if last_losses else None,
                        "steps_per_second":(t-start_step)/max(time.perf_counter()-start_time,1e-5),
                        "best_win_percent":best_score[0],"best_pellet_percent":best_score[1]}
                writer.writerow(record);fp.flush()
                print(f"{t:>9}/{steps} {architecture:<14} eps={epsilon:.2f} "
                      f"pellets={score[1]:5.1f}% wins={score[0]:5.1f}% "
                      f"speed={record['steps_per_second']:.1f} steps/s",flush=True)
            if save:
                payload={"version":"0.7.0","architecture":architecture,
                         "model":online.state_dict(),"target":target.state_dict(),
                         "optimizer":optimizer.state_dict(),"input_shape":shape,
                         "step":t,"steps":t,"episodes":episodes,"config":config,
                         "best_score":best_score,"metrics":metrics or {}}
                if evaluate and improved:
                    atomic_save_torch(payload,path/"best.pt")
                atomic_save_torch(payload,path/"last.pt")
    if engine == "rust": game.close()
    return path


def main():
    p=argparse.ArgumentParser(description="One large CNN/GRU Double DQN training job")
    p.add_argument("--architecture",choices=tuple(SPECS),required=True)
    p.add_argument("--steps",type=int,default=1_000_000)
    p.add_argument("--maze",choices=("small","large","xlarge"),default="large")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("auto","cuda","cpu","mps"),default="auto")
    p.add_argument("--engine",choices=("python","rust"),default="python")
    p.add_argument("--eval-every",type=int,default=50000)
    p.add_argument("--save-every",type=int,default=25000)
    p.add_argument("--eval-episodes",type=int,default=40)
    p.add_argument("--resume",type=str)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--warmup",type=int,default=2000)
    p.add_argument("--replay-capacity",type=int,default=30000)
    p.add_argument("--burnin",type=int,default=4)
    p.add_argument("--unroll",type=int,default=8)
    p.add_argument("--train-every",type=int,default=8)
    p.add_argument("--worlds",type=int,default=1,help="Batched independent environments; 1 preserves v0.7 trainer")
    a=p.parse_args()
    if a.worlds>1:
        from .cnn_batch import run_batched
        runner=run_batched
    else:
        runner=run_training
    runner(architecture=a.architecture,steps=a.steps,seed=a.seed,
                 output=a.output,device_name=a.device,eval_every=a.eval_every,
                 save_every=a.save_every,eval_episodes=a.eval_episodes,
                 batch_size=a.batch_size,warmup=a.warmup,
                 replay_capacity=a.replay_capacity,burnin=a.burnin,
                 unroll=a.unroll,train_every=a.train_every,resume=a.resume,
                 maze=a.maze,engine=a.engine,**({"worlds":a.worlds} if a.worlds>1 else {}))

if __name__ == "__main__":main()
