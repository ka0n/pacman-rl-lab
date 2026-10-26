# SPDX-License-Identifier: GPL-3.0-or-later
"""Regressions: batched eval fidelity, replay isolation, warm resume and smoke."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from pacman_lab.cnn_batch import VectorEpisodeReplay, PythonBatch, eval_batched, run_batched
from pacman_lab.cnn_train import eval_model
from pacman_lab.cnn_models import SPECS, make_model
from pacman_lab.core import PacmanGame


def test_python_batch_matches_individual_game():
    batch=PythonBatch("small",3)
    games=[PacmanGame("small",ghosts=2) for _ in range(3)]
    try:
        for i in range(3):
            obs,info=batch.reset_one(i,seed=100+i)
            ref,_=games[i].reset(seed=100+i)
            assert np.array_equal(obs,ref)
        for t in range(9):
            actions=[(t+i)%4 for i in range(3)]
            outcome=batch.step(actions)
            for i,g in enumerate(games):
                expect=g.step(actions[i])
                assert np.array_equal(outcome.observations[i],expect.observation)
                assert outcome.rewards[i]==expect.reward
                assert bool(outcome.terminated[i])==expect.terminated
                assert bool(outcome.truncated[i])==expect.truncated
                if expect.terminated or expect.truncated:
                    batch.reset_one(i,seed=10000+t*100+i)
                    g.reset(seed=10000+t*100+i)
    finally:batch.close()


def test_recurrent_replay_never_crosses_worlds_or_terminals():
    replay=VectorEpisodeReplay(capacity=80,sequence_length=3,worlds=2)
    for t in range(20):
        for w in range(2):
            # Encode both lane and episode in the sample, so mixing is detectable.
            episode=t//(5+w)
            replay.append_at(w,(w,episode,t),done=(t+1)%(5+w)==0)
    assert replay.eligible
    import random
    for sample in replay.sample(random.Random(19),100):
        assert len({(entry[0],entry[1]) for entry in sample})==1
        assert all(b[2]==a[2]+1 for a,b in zip(sample,sample[1:]))


@pytest.mark.parametrize("arch",tuple(SPECS))
def test_batch_eval_same_as_single_on_cpu(arch):
    torch=pytest.importorskip("torch")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model=make_model(arch,((5 if arch=="cnn_motion" else 4*SPECS[arch].frames),7,9))
    simple=eval_model(model,arch,torch.device("cpu"),maze="small",episodes=7,seed=33,engine="python")
    batch=eval_batched(model,arch,torch.device("cpu"),maze="small",episodes=7,seed=33,engine="python",worlds=3)
    assert simple==batch


@pytest.mark.parametrize("arch",("cnn_history4","cnn_gru"))
def test_batch_trains_checkpoints_and_warm_restarts(tmp_path:Path,arch):
    torch=pytest.importorskip("torch")
    torch.set_num_threads(1)
    target=tmp_path/arch
    kws=dict(architecture=arch,steps=96,seed=42,output=target,
             worlds=4,device_name="cpu",eval_every=48,save_every=48,
             eval_episodes=3,replay_capacity=128,batch_size=4,
             warmup=8,train_every=8,burnin=1,unroll=2,
             maze="small",engine="python")
    run_batched(**kws)
    p=torch.load(target/"last.pt",map_location="cpu",weights_only=False)
    assert p["step"]==96 and p["config"]["worlds"]==4
    with (target/"metrics.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows]==[48,96]
    assert (target/"best.pt").exists()
    run_batched(**dict(kws,resume=str(target/"last.pt")))
    assert list(target.glob("best.pt"))
    assert p["architecture"]==arch


def test_batched_invalid_worlds(tmp_path):
    with pytest.raises(ValueError):
        run_batched(architecture="cnn_single",steps=10,seed=42,output=tmp_path/"bad",
                    worlds=0,device_name="cpu",engine="python")
