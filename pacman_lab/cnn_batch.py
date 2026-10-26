# SPDX-License-Identifier: GPL-3.0-or-later
"""Batched CNN/GRU online RL with independent worlds.

Each environment contributes its own episode history; recurrent sequences never
cross a reset or another environment. The step budget counts *transitions*,
not vectorised calls.  Inference runs once per batch (and once per rollout for
GRU hidden states).  The v0.7 single-environment trainer is unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from .cnn_models import SPECS, make_model, parameters_count
from .cnn_train import EpisodeReplay, FIELDS, eval_model, minibatch_loss
from .core import PacmanGame, validate_batch_actions
from .dqn import select_device
from .marathon_train import atomic_save_torch, checkpoint_read, resume_best_score
from .cnn_motion import observer_for, current_frame
from .train import exploration_epsilon
from .cnn_safety import maze_risk, shaped_reward


class PythonBatch:
    """Functional parity fallback; one API call but Python loop internally."""
    def __init__(self, maze: str, count: int, ghosts=2):
        self.games = [PacmanGame(maze=maze, ghosts=ghosts) for _ in range(count)]
        self.count = count
        self.height, self.width = self.games[0].height, self.games[0].width

    def reset_one(self, index: int, seed: int):
        return self.games[index].reset(seed=seed)

    def step(self, actions):
        from .native import BatchOutcome
        actions = validate_batch_actions(actions, self.count)
        if any(game.done for game in self.games):
            raise RuntimeError("episode finished; reset completed episodes before stepping")
        outcomes = [game.step(int(action)) for game,action in zip(self.games,actions)]
        return BatchOutcome(
            observations=np.stack([x.observation for x in outcomes]),
            rewards=np.asarray([x.reward for x in outcomes],dtype=np.float64),
            terminated=np.asarray([x.terminated for x in outcomes],dtype=np.uint8),
            truncated=np.asarray([x.truncated for x in outcomes],dtype=np.uint8),
            stats=[x.info for x in outcomes],
        )

    def close(self):
        pass


class VectorEpisodeReplay(EpisodeReplay):
    """Store independent unfinished episodes, preserving recurrent boundaries."""
    def __init__(self, capacity: int, sequence_length: int, worlds: int):
        super().__init__(capacity, sequence_length)
        if worlds < 1 or capacity < worlds * sequence_length:
            raise ValueError("Replay capacity must hold burn-in + unroll for every world")
        self.episode_capacities = [capacity // worlds + int(i < capacity % worlds)
                                   for i in range(worlds)]
        self.current_by_world: list[list] = [[] for _ in range(worlds)]

    def _inflight(self):
        return zip(self.current_by_world, self.episode_capacities)

    def append_at(self, index: int, item, done: bool):
        self.current_by_world[index] = self._append_episode(
            self.current_by_world[index], item, done, self.episode_capacities[index])


def run_batched(*, architecture: str, steps: int, seed: int, output: Path | str,
                worlds: int=16, device_name="auto", eval_every=50000,
                save_every=25000, eval_episodes=40, replay_capacity=30000,
                batch_size=64, warmup=2000, train_every=8, target_every=2000,
                burnin=4, unroll=8, learning_rate=0.00015,
                epsilon_start=1.0, epsilon_end=0.08,
                epsilon_decay_fraction=0.85, resume: str | None=None,
                init_checkpoint: str | None=None,
                maze: str="large", engine: str="rust",
                death_penalty: float | None = None, risk_shaping_weight: float = 0.0,
                fresh_optimizer: bool = False, matched_scratch_init: bool = False):
    import torch
    if init_checkpoint is not None and resume is not None:
        raise ValueError("Use either init_checkpoint (fresh fine-tune run) OR resume")
    if matched_scratch_init and (architecture != "cnn_motion" or init_checkpoint is not None):
        raise ValueError("Matched scratch initialization is only valid for cnn_motion training from random weights")
    if engine not in ("python", "rust") or architecture not in SPECS:
        raise ValueError("Unsupported engine or architecture")
    if min(worlds,steps,eval_every,save_every,eval_episodes,replay_capacity,
           batch_size,train_every,target_every,unroll) <= 0 or burnin < 0 or warmup < 0:
        raise ValueError("Invalid training budget")
    if batch_size > replay_capacity or not (0 < epsilon_end <= epsilon_start <= 1):
        raise ValueError("Invalid batch size or epsilon")
    if not 0 < epsilon_decay_fraction <= 1:
        raise ValueError("Invalid epsilon schedule")
    worlds = min(worlds, steps)
    if SPECS[architecture].recurrent and replay_capacity < worlds * (burnin + unroll):
        raise ValueError("Replay capacity must hold burn-in + unroll for every world")
    if SPECS[architecture].recurrent and warmup > replay_capacity:
        raise ValueError("Recurrent warmup must not exceed replay capacity")
    if death_penalty is not None and death_penalty >= 0:
        raise ValueError('Death penalty must be negative')
    if risk_shaping_weight < 0:
        raise ValueError('Risk shaping weight must be nonnegative')
    path = Path(output)
    path.mkdir(parents=True,exist_ok=True)
    device=select_device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rng=random.Random(seed)
    if engine == "rust":
        from .native import NativeBatch
        env=NativeBatch(maze=maze,count=worlds,ghosts=2,seed=seed)
    else:
        env=PythonBatch(maze=maze,count=worlds,ghosts=2)

    try:
        spec=SPECS[architecture]
        frames=[observer_for(architecture, spec.frames) for _ in range(worlds)]
        dummy=frames[0].reset(np.zeros((4,env.height,env.width),dtype=np.uint8))
        shape=tuple(dummy.shape)
        online=make_model(architecture,shape).to(device)
        target=make_model(architecture,shape).to(device)
        if matched_scratch_init and not resume:
            # Match all shared weights against cnn_single started with the same
            # seed. The extra (past ghost) channel starts at exactly zero.
            # fork_rng keeps our additional reference initialization from
            # consuming the learner's random-number stream.
            from .cnn_motion import transfer_cnn_single_to_motion
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                reference = make_model("cnn_single", (4, *shape[1:]))
                transfer_cnn_single_to_motion(reference.state_dict(), online)
                del reference
            target.load_state_dict(online.state_dict())
        optimizer=torch.optim.Adam(online.parameters(),lr=learning_rate)
        config={"version":"0.8.0","architecture":architecture,"steps":steps,
                "seed":seed,"maze":maze,"engine":engine,"worlds":worlds,
                "frames":spec.frames,"recurrent":spec.recurrent,"double_dqn":True,
                "parameters":parameters_count(online),"eval_every":eval_every,
                "save_every":save_every,"eval_episodes":eval_episodes,
                "replay_capacity":replay_capacity,"batch_size":batch_size,
                "warmup":warmup,"train_every":train_every,"target_every":target_every,
                "burnin":burnin,"unroll":unroll,"learning_rate":learning_rate,
                "epsilon_start":epsilon_start,"epsilon_end":epsilon_end,
                "epsilon_decay_fraction":epsilon_decay_fraction}
        # Optional objective adjustment. Default omitted for exact v0.8/v0.9
        # config compatibility; resume enforces matching safety parameters.
        if death_penalty is not None:
            config['death_penalty'] = death_penalty
        if risk_shaping_weight:
            config['risk_shaping_weight'] = risk_shaping_weight
        if fresh_optimizer:
            config['fresh_optimizer'] = True
        if matched_scratch_init:
            config['matched_scratch_init'] = True
        topology = maze_risk(maze) if risk_shaping_weight else None
        start_step=0
        episodes=0
        best_score=(-1.0,-1.0)
        fine_tune=None
        if resume:
            payload=checkpoint_read(resume,device)
            if payload.get("config") != config:
                raise ValueError("Resume requires the same v0.8 batch settings. Start a new suite for a changed configuration.")
            if tuple(payload["input_shape"]) != shape:
                raise ValueError("Checkpoint shape mismatch")
            start_step=int(payload["step"])
            episodes=int(payload["episodes"])
            best_score=resume_best_score(payload, path, device, resume)
            fine_tune=payload.get("fine_tune")
            online.load_state_dict(payload["model"])
            target.load_state_dict(payload["target"])
            optimizer.load_state_dict(payload["optimizer"])
            print(f"WARM RESTART at {start_step}: replay/RNG/worlds restart",flush=True)
        else:
            if any((path / name).exists() for name in ("last.pt", "best.pt", "metrics.csv", "config.json")):
                raise FileExistsError(f"Refusing to overwrite {path}")
            if init_checkpoint:
                src=Path(init_checkpoint).resolve(strict=True)
                checkpoint=checkpoint_read(src,device)
                transfer_motion = (architecture == "cnn_motion" and
                                   checkpoint.get("architecture") == "cnn_single")
                if not transfer_motion and checkpoint.get("architecture") != architecture:
                    raise ValueError("Source architecture mismatch")
                source_shape = tuple(checkpoint.get("input_shape",()))
                expected_source = (4,*shape[1:]) if transfer_motion else shape
                if source_shape != expected_source:
                    raise ValueError("Source input_shape mismatch")
                oldcfg=checkpoint.get("config",{})
                if oldcfg.get("maze") != maze:
                    raise ValueError("Source maze mismatch (fine-tune on same maze first)")
                if transfer_motion:
                    from .cnn_motion import transfer_cnn_single_to_motion
                    transfer_cnn_single_to_motion(checkpoint["model"], online)
                else:
                    online.load_state_dict(checkpoint["model"])
                # No stale target lag when starting a new fine-tune run.
                target.load_state_dict(online.state_dict())
                restore_optimizer = not (transfer_motion or fresh_optimizer) and "optimizer" in checkpoint
                if restore_optimizer:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                fine_tune={"checkpoint":str(src),
                           "sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
                           "source_step":int(checkpoint.get("step",0)),
                           "source_architecture":checkpoint["architecture"],
                           "source_seed":oldcfg.get("seed"),
                           "optimizer_restored":restore_optimizer,
                           "replay_restored":False,
                           "motion_transfer":transfer_motion,
                           "extra_input_channels_zero_initialized":int(transfer_motion),
                           "schedule_restarted":True}
                (path/"fine_tune.json").write_text(json.dumps(fine_tune,indent=2),encoding="utf-8")
                print(f"FINE-TUNE from {src} at {fine_tune['source_step']} source steps: "
                      f"new +{steps} steps, optimizer {'restored' if restore_optimizer else 'fresh'}, "
                      f"replay reset",flush=True)
            else:
                target.load_state_dict(online.state_dict())
        if start_step >= steps: return path
        (path/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        target.eval()
        replay=(VectorEpisodeReplay(replay_capacity,burnin+unroll,worlds)
                if spec.recurrent else deque(maxlen=replay_capacity))
        # Use different world seeds; never overlap validation or test seed sets.
        seed_base=1_000_000+seed+start_step
        next_seed=seed_base
        state=[]
        for i in range(worlds):
            obs,_=env.reset_one(i,seed=next_seed)
            next_seed+=1
            state.append(frames[i].reset(obs))
        hidden=None
        losses=deque(maxlen=500)
        start_time=time.perf_counter()
        metrics_path=path/"metrics.csv"
        if resume and metrics_path.exists():
            with metrics_path.open(newline="",encoding="utf-8") as f:
                rows=[row for row in csv.DictReader(f) if int(row["step"])<=start_step]
            with metrics_path.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS)
                writer.writeheader();writer.writerows(rows)
        new_csv=not metrics_path.exists()
        print(f"{architecture} batched {worlds} worlds on {device} | "
              f"{parameters_count(online):,} params | {start_step}->{steps} steps",
              flush=True)
        with metrics_path.open("a",newline="",encoding="utf-8") as fp:
            writer=csv.DictWriter(fp,fieldnames=FIELDS)
            if new_csv:writer.writeheader()
            completed=start_step
            while completed < steps:
                active=min(worlds,steps-completed)
                epsilon=exploration_epsilon(completed+active,steps,epsilon_start,
                                            epsilon_end,epsilon_decay_fraction)
                # One batched GPU invocation; recurrent hidden state updated even for
                # exploratory moves, otherwise it would lose ghost motion context.
                explore=np.asarray([rng.random()<epsilon for _ in range(worlds)],dtype=bool)
                actions=np.asarray([rng.randrange(4) for _ in range(worlds)],dtype=np.uint8)
                if spec.recurrent or not explore.all():
                    with torch.inference_mode():
                        batch=torch.as_tensor(np.stack(state),device=device,dtype=torch.float32)
                        if spec.recurrent:
                            q,hidden=online(batch[:,None],hidden)
                            greedy=q[:, -1].argmax(dim=-1).cpu().numpy()
                        else:
                            greedy=online(batch).argmax(dim=-1).cpu().numpy()
                    actions[~explore]=greedy[~explore]
                outcome=env.step(actions)
                new_states=[]
                for i in range(worlds):
                    ns=frames[i].step(outcome.observations[i])
                    done=bool(outcome.terminated[i] or outcome.truncated[i])
                    if i < active:
                        reward = shaped_reward(
                            float(outcome.rewards[i]), current_frame(state[i], architecture), current_frame(ns, architecture),
                            topology=topology, risk_weight=risk_shaping_weight,
                            terminated=bool(outcome.terminated[i]),
                            truncated=bool(outcome.truncated[i]),
                            won=bool(outcome.stats[i]['won']),
                            death_penalty=death_penalty)
                        transition=(state[i].copy(),int(actions[i]),reward,ns.copy(),done)
                        if spec.recurrent:
                            replay.append_at(i,transition,done)
                        else:replay.append(transition)
                    if done:
                        if i < active:episodes+=1
                        obs,_=env.reset_one(i,seed=next_seed)
                        next_seed+=1
                        ns=frames[i].reset(obs)
                        if hidden is not None:
                            # Reset only this environment's hidden state.
                            with torch.inference_mode():
                                hidden[:,i,:]=0
                    new_states.append(ns)
                state=new_states
                previous=completed
                completed+=active
                nupdates=(completed//train_every)-(previous//train_every)
                if completed-start_step>=warmup:
                    for _ in range(nupdates):
                        ready=(replay.total>=max(warmup,batch_size) and replay.can_sample
                               if spec.recurrent else len(replay)>=max(warmup,batch_size))
                        if not ready:break
                        samples=(replay.sample(rng,batch_size) if spec.recurrent
                                 else rng.sample(replay,batch_size))
                        loss=minibatch_loss(online,target,samples,architecture,device,burnin)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(online.parameters(),10.)
                        optimizer.step()
                        losses.append(float(loss.detach().item()))
                if completed//target_every > previous//target_every:
                    target.load_state_dict(online.state_dict())
                evaluate=completed//eval_every>previous//eval_every or completed==steps
                save=completed//save_every>previous//save_every or evaluate
                metrics=None
                if evaluate:
                    metrics=eval_batched(online,architecture,device,maze=maze,
                                         episodes=eval_episodes,seed=10000,engine=engine,
                                         worlds=worlds)
                    score=(metrics["win_percent"],metrics["pellet_percent"])
                    improved=score>best_score
                    if improved:best_score=score
                    record={"step":completed,"episodes":episodes,"epsilon":epsilon,
                            **{k:metrics[k] for k in metrics if k in FIELDS},
                            "train_loss":float(np.mean(losses)) if losses else None,
                            "steps_per_second":(completed-start_step)/max(time.perf_counter()-start_time,1e-5),
                            "best_win_percent":best_score[0],"best_pellet_percent":best_score[1]}
                    writer.writerow(record);fp.flush()
                    print(f"{completed:>9}/{steps} {architecture:<14} eps={epsilon:.2f} "
                          f"pellets={score[1]:.1f}% wins={score[0]:.1f}% "
                          f"speed={record['steps_per_second']:.1f} steps/s",flush=True)
                if save:
                    payload={"version":"0.9.0" if fine_tune else "0.8.0","architecture":architecture,
                             "model":online.state_dict(),"target":target.state_dict(),
                             "optimizer":optimizer.state_dict(),"input_shape":shape,
                             "step":completed,"steps":completed,"episodes":episodes,
                             "config":config,"best_score":best_score,"metrics":metrics or {},
                             "fine_tune":fine_tune}
                    if evaluate and improved:atomic_save_torch(payload,path/"best.pt")
                    atomic_save_torch(payload,path/"last.pt")
        return path
    finally:
        env.close()


def eval_batched(model, architecture: str, device, *, maze="large", episodes=40,
                 seed=10000, engine="rust", worlds=16):
    """Same deterministic per-episode seeds as eval_model, with batched inference.

    Finished lanes are reset to filler episodes solely because NativeBatch.step
    requires all lanes active. They cannot affect other environments and are
    excluded from metrics. No extra reward is counted for those lanes.
    """
    import torch
    if episodes < 1 or worlds < 1:raise ValueError("Episodes/worlds must be >= 1")
    spec=SPECS[architecture]
    was_training=model.training
    model.eval()
    collected=[]
    win_count=0
    walls=[];stays=[];reversals=[];lengths=[]
    try:
        with torch.inference_mode():
            for base in range(0,episodes,worlds):
                active=min(worlds,episodes-base)
                if engine=="rust":
                    from .native import NativeBatch
                    env=NativeBatch(maze=maze,count=active,ghosts=2)
                else:
                    env=PythonBatch(maze=maze,count=active,ghosts=2)
                try:
                    hist=[observer_for(architecture, spec.frames) for _ in range(active)]
                    states=[]
                    for i in range(active):
                        obs,_=env.reset_one(i,seed=seed+base+i)
                        states.append(hist[i].reset(obs))
                    done=np.zeros(active,dtype=bool)
                    hidden=None
                    filler_seed=seed+episodes+base*1_000_000
                    while not bool(done.all()):
                        batch=torch.as_tensor(np.stack(states),device=device,dtype=torch.float32)
                        if spec.recurrent:
                            q,hidden=model(batch[:,None],hidden)
                            actions=q[:,-1].argmax(dim=-1).cpu().numpy().astype(np.uint8)
                        else:
                            actions=model(batch).argmax(dim=-1).cpu().numpy().astype(np.uint8)
                        outcome=env.step(actions)
                        for i in range(active):
                            if not done[i]:
                                ended=bool(outcome.terminated[i] or outcome.truncated[i])
                                if ended:
                                    done[i]=True
                                    info=outcome.stats[i]
                                    collected.append(info["collected_fraction"])
                                    win_count+=int(info["won"])
                                    walls.append(info["wall_bumps"])
                                    stays.append(info["stationary_steps"])
                                    reversals.append(info["reversals"])
                                    lengths.append(info["steps"])
                            if done[i]:
                                # Placeholder world; never counted as an evaluation.
                                obs,_=env.reset_one(i,seed=filler_seed)
                                filler_seed+=1
                                states[i]=hist[i].reset(obs)
                                if hidden is not None:hidden[:,i,:]=0
                            else:
                                states[i]=hist[i].step(outcome.observations[i])
                finally:
                    env.close()
    finally:
        model.train(was_training)
    return {"pellet_percent":round(float(np.mean(collected))*100,4),
            "win_percent":round(100*win_count/episodes,4),
            "wins":win_count,"episodes":episodes,
            "wall_bumps_per_episode":float(np.mean(walls)),
            "stationary_percent":100*sum(stays)/max(1,sum(lengths)),
            "reversals_per_episode":float(np.mean(reversals)),
            "steps_per_episode":float(np.mean(lengths))}
