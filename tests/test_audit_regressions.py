# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #8: executable reproductions of the six independent audit findings."""
import importlib
import json
import random

import numpy as np
import pytest

from pacman_lab.core import PacmanGame
from pacman_lab.cnn_batch import PythonBatch, VectorEpisodeReplay
from pacman_lab.cnn_train import EpisodeReplay
from pacman_lab.gym_env import PacmanEnv
from pacman_lab.native import NativeBatch, NativePacmanGame, _load_library


@pytest.fixture
def native_available():
    try:
        _load_library()
    except FileNotFoundError:
        pytest.skip("Build native simulator with cargo build --release")


@pytest.mark.parametrize("maze", ("small", "large"))
@pytest.mark.parametrize("initial,requested", ((0, 2), (1, 2), (2, 0)))
def test_ghost_reset_pool_and_gym_parity(native_available, maze, initial, requested):
    python = PacmanGame(maze, ghosts=initial)
    env = PacmanEnv(maze, ghosts=initial)
    native = NativePacmanGame(maze, ghosts=initial)
    reference = PacmanGame(maze, ghosts=requested, max_steps=24)
    try:
        expected, info = reference.reset(seed=7)
        for obs, actual_info in (python.reset(seed=7, ghosts=requested),
                                 native.reset(seed=7, ghosts=requested),
                                 env.reset(seed=7, options={"ghosts": requested})):
            np.testing.assert_array_equal(obs, expected)
            assert actual_info == info
        assert len(python.ghosts) == requested
        # Exercise RNG and movement after changing the active count as well.
        for _ in range(12):
            action = random.Random(python.steps).randrange(4)
            a, b = python.step(action), native.step(action)
            np.testing.assert_array_equal(a.observation, b.observation)
            assert (a.reward, a.terminated, a.truncated, a.info) == (
                b.reward, b.terminated, b.truncated, b.info)
            if python.done:
                break
    finally:
        native.close()
        env.close()


@pytest.mark.parametrize("engine", ("python", "rust"))
@pytest.mark.parametrize("actions", (
    np.array([0, 256], dtype=np.int64), np.array([0, -256]),
    np.array([0, -1]), np.array([0, 4]), np.array([0., 1.5]),
    np.array([0., 1.]), np.array([0., np.nan]), np.array([0., np.inf]),
    np.array([0, 2**64-1], dtype=np.uint64), np.array([False, True]),
    np.array(["0", "1"]), np.array([0]), np.array([[0, 1]]),
))
def test_invalid_batch_actions_are_atomic(request, engine, actions):
    if engine == "rust":
        request.getfixturevalue("native_available")
    batch = (PythonBatch("small", 2, ghosts=0) if engine == "python" else
             NativeBatch("small", count=2, ghosts=0))
    try:
        before = [batch.reset_one(i, seed=71+i) for i in range(2)]
        with pytest.raises(ValueError):
            batch.step(actions)
        after = ([(g.observe(), g.info()) for g in batch.games] if engine == "python"
                 else [(obs, batch._info(stats)) for obs, stats in
                       zip(batch.observe(), batch._stats)])
        for (obs, info), (old_obs, old_info) in zip(after, before):
            np.testing.assert_array_equal(obs, old_obs)
            assert info == old_info
        # Rejection must leave the batch usable.
        assert all(s["steps"] == 1 for s in batch.step([0, 3]).stats)
    finally:
        batch.close()


@pytest.mark.parametrize("engine", ("python", "rust"))
def test_finished_batch_lane_does_not_advance_other_lanes(request, engine):
    if engine == "rust":
        request.getfixturevalue("native_available")
    batch = (PythonBatch("small", 2, ghosts=0) if engine == "python" else
             NativeBatch("small", count=2, ghosts=0, max_steps=1))
    try:
        if engine == "python":
            for game in batch.games:
                game.max_steps = 1
        batch.step([0, 0])
        obs, info = batch.reset_one(0, seed=7)
        with pytest.raises(RuntimeError):
            batch.step([3, 3])
        if engine == "python":
            np.testing.assert_array_equal(batch.games[0].observe(), obs)
            assert batch.games[0].info() == info
        else:
            np.testing.assert_array_equal(batch.observe()[0], obs)
            assert batch._info(batch._stats[0]) == info
    finally:
        batch.close()


def test_native_reset_metadata_commits_only_on_success(native_available, monkeypatch):
    game = NativePacmanGame("large", ghosts=1, ghost_every=1)
    try:
        game.reset(seed=7, ghosts=2, ghost_every=3)
        assert (game.ghost_count, game.ghost_every) == (2, 3)
        assert (game.batch.ghost_count, game.batch.ghost_every) == (2, 3)
        before = game.observe(), game.info()
        for kwargs in ({"ghosts": 0, "ghost_every": 0},
                       {"ghosts": 0, "ghost_every": 2, "seed": -1}):
            with pytest.raises(ValueError):
                game.reset(**kwargs)
            assert (game.batch.ghost_count, game.batch.ghost_every) == (2, 3)
        with monkeypatch.context() as patch:
            patch.setattr(game.batch.lib, "pacman_reset_one", lambda *args: -3)
            with pytest.raises(RuntimeError):
                game.reset(seed=8, ghosts=0, ghost_every=2)
        assert (game.ghost_count, game.ghost_every) == (2, 3)
        assert (game.batch.ghost_count, game.batch.ghost_every) == (2, 3)
        np.testing.assert_array_equal(game.observe(), before[0])
        assert game.info() == before[1]
        # A reset without overrides must still use the last successful settings.
        _, info = game.reset(seed=7)
        assert info["ghosts"] == 2
        python = PacmanGame("large", ghosts=2, ghost_every=3, seed=7)
        for action in (0, 3, 3, 1):
            a, b = game.step(action), python.step(action)
            np.testing.assert_array_equal(a.observation, b.observation)
            assert a.info == b.info
    finally:
        game.close()


@pytest.mark.parametrize("worlds", (1, 2, 3))
def test_long_recurrent_episodes_remain_bounded_and_sampleable(worlds):
    capacity = 5 * worlds + (1 if worlds > 1 else 0)
    replay = (EpisodeReplay(capacity, 3) if worlds == 1 else
              VectorEpisodeReplay(capacity, 3, worlds))
    for episode in range(2):
        for step in range(16):
            for world in range(worlds):
                item = (world, episode, step)
                if worlds == 1:
                    replay.append(item, done=step == 15)
                else:
                    replay.append_at(world, item, done=step == 15)
                assert replay.total <= replay.capacity
            if step >= 6:
                for sample in replay.sample(random.Random(7), 8):
                    assert len(sample) == 3
                    assert len({(x[0], x[1]) for x in sample}) == 1
                    assert all(b[2] == a[2]+1 for a, b in zip(sample, sample[1:]))


@pytest.mark.parametrize("module_name,capacity,warmup,worlds", (
    ("cnn_train", 2, 1, 1), ("cnn_train", 6, 7, 1),
    ("cnn_batch", 2, 1, 1), ("cnn_batch", 5, 1, 2), ("cnn_batch", 6, 7, 2)))
def test_gru_trainers_reject_infeasible_replay_before_output(
        tmp_path, module_name, capacity, warmup, worlds):
    module = importlib.import_module(f"pacman_lab.{module_name}")
    kwargs = dict(architecture="cnn_gru", steps=12, seed=42, output=tmp_path/"invalid",
                  device_name="cpu", replay_capacity=capacity, warmup=warmup,
                  batch_size=1, burnin=1, unroll=2)
    if module_name == "cnn_batch":
        kwargs["worlds"] = worlds
        runner = module.run_batched
    else:
        runner = module.run_training
    with pytest.raises(ValueError):
        runner(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("args", ((0, 1), (5, 0), (2, 3)))
def test_recurrent_replay_rejects_impossible_capacity(args):
    with pytest.raises(ValueError):
        EpisodeReplay(*args)


def test_vector_replay_requires_one_sequence_per_world():
    with pytest.raises(ValueError):
        VectorEpisodeReplay(5, 3, worlds=2)


@pytest.mark.parametrize("module_name", ("cnn_train", "cnn_batch"))
def test_long_episode_gru_trainer_really_updates(tmp_path, monkeypatch, module_name):
    torch = pytest.importorskip("torch")
    module = importlib.import_module(f"pacman_lab.{module_name}")
    # A real six-transition timeout episode exceeds the five-transition buffer.
    def long_game(*args, **kwargs):
        return PacmanGame(*args, **dict(kwargs, ghosts=0, max_steps=6))
    monkeypatch.setattr(module, "PacmanGame", long_game)
    monkeypatch.setattr(module, "eval_model" if module_name == "cnn_train" else
                        "eval_batched", lambda *args, **kwargs: metrics(10))
    kwargs = dict(architecture="cnn_gru", steps=12, seed=42, output=tmp_path/module_name,
                  device_name="cpu", engine="python", maze="small", eval_every=12,
                  save_every=12, eval_episodes=1, replay_capacity=5, batch_size=1,
                  warmup=1, train_every=1, burnin=1, unroll=2)
    if module_name == "cnn_batch":
        kwargs["worlds"] = 1
    runner = module.run_training if module_name == "cnn_train" else module.run_batched
    runner(**kwargs)
    payload = torch.load(kwargs["output"]/"last.pt", weights_only=True)
    assert payload["optimizer"]["state"]
    assert max(s["step"].item() for s in payload["optimizer"]["state"].values()) > 0
    assert any(not torch.equal(payload["model"][key], value)
               for key, value in payload["target"].items())


def metrics(score):
    return dict(win_percent=score, pellet_percent=score, wall_bumps_per_episode=0.)


@pytest.mark.parametrize("module_name", ("cnn_train", "cnn_batch", "marathon_train"))
@pytest.mark.parametrize("existing_best", (False, True))
@pytest.mark.parametrize("relocated", (False, True))
def test_improved_final_checkpoint_survives_mid_save_crash(
        tmp_path, monkeypatch, module_name, existing_best, relocated):
    torch = pytest.importorskip("torch")
    module = importlib.import_module(f"pacman_lab.{module_name}")
    evaluation = {"cnn_train": "eval_model", "cnn_batch": "eval_batched",
                  "marathon_train": "evaluate_agent"}[module_name]
    scores = iter((10, 20) if existing_best else (20,))
    monkeypatch.setattr(module, evaluation, lambda *args, **kwargs: metrics(next(scores)))
    output = tmp_path/module_name
    kwargs = dict(steps=4, seed=42, output=output, device_name="cpu", maze="small",
                  eval_every=2 if existing_best else 4,
                  save_every=1 if existing_best else 4, eval_episodes=1, batch_size=1,
                  replay_capacity=16)
    if module_name == "marathon_train":
        runner = module.run_marathon_training
        kwargs.update(variant="plain_dqn", train_warmup=10)
    else:
        runner = module.run_training if module_name == "cnn_train" else module.run_batched
        kwargs.update(architecture="cnn_single", engine="python", warmup=10)
        if module_name == "cnn_batch":
            kwargs["worlds"] = 1
    save = module.atomic_save_torch
    class Crash(Exception):
        pass
    def crash_between_saves(payload, path):
        save(payload, path)
        if payload["steps"] == 4:
            raise Crash
    monkeypatch.setattr(module, "atomic_save_torch", crash_between_saves)
    with pytest.raises(Crash):
        runner(**kwargs)
    best = torch.load(output/"best.pt", weights_only=True)
    assert best["best_score"] == (20, 20)
    assert best["steps"] == 4
    if existing_best:
        last = torch.load(output/"last.pt", weights_only=True)
        assert last["steps"] == 3
    else:
        assert not (output/"last.pt").exists()
    monkeypatch.setattr(module, "atomic_save_torch", save)
    # Warm restart can evaluate differently; never overwrite the durable best.
    monkeypatch.setattr(module, evaluation, lambda *args, **kwargs: metrics(15))
    recovery = output/("last.pt" if existing_best else "best.pt")
    destination = tmp_path/"relocated" if relocated else output
    runner(**dict(kwargs, output=destination, resume=str(recovery)))
    best_after = torch.load(destination/"best.pt", weights_only=True)
    assert best_after["best_score"] == (20, 20)
    if existing_best:
        last_after = torch.load(destination/"last.pt", weights_only=True)
        assert last_after["best_score"] == (20, 20)
    for key in best["model"]:
        assert torch.equal(best["model"][key], best_after["model"][key])
    final_recovery = destination/("last.pt" if existing_best else "best.pt")
    runner(**dict(kwargs, output=destination, resume=str(final_recovery)))
    assert torch.load(destination/"best.pt", weights_only=True)["best_score"] == (20, 20)


def test_cnn_suite_preserves_best_when_first_last_save_was_interrupted(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from pacman_lab import cnn_marathon
    run = tmp_path/"run"
    run.mkdir()
    torch.save({"step": 4}, run/"best.pt")
    before = (run/"best.pt").read_bytes()
    manifest = {"settings": {"steps": 4, "engine": "python"},
                "runs": [{"run_dir": "run", "status": "running"}]}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cnn_marathon, "build_report", lambda root: root)
    monkeypatch.setattr(cnn_marathon.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("Completed best must be recovered"))
    cnn_marathon.run_suite(tmp_path, resume=True)
    assert (run/"best.pt").read_bytes() == before
    assert json.loads((tmp_path/"manifest.json").read_text())["runs"][0]["status"] == "completed"


def test_legacy_timeout_bellman_target_has_no_bootstrap(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional
    from pacman_lab import train
    def timeout_game(*args, **kwargs):
        return PacmanGame(*args, **dict(kwargs, ghosts=0, max_steps=1))
    class ConstantQ(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Parameter(torch.full((4,), 8.))
        def forward(self, x):
            return self.q.expand(len(x), -1)
    monkeypatch.setattr(train, "PacmanGame", timeout_game)
    monkeypatch.setattr(train, "make_qnet", lambda shape: ConstantQ())
    monkeypatch.setattr(train, "stage", lambda *args: (0, 1))
    monkeypatch.setattr(train, "evaluate", lambda *args, **kwargs: dict(
        metrics(0), stationary_percent=0., reversals_per_episode=0., steps_per_episode=1.))
    loss = functional.smooth_l1_loss
    targets = []
    def capture(q, target):
        targets.append(target.detach().clone())
        return loss(q, target)
    monkeypatch.setattr(functional, "smooth_l1_loss", capture)
    train.run_training(steps=500, output=str(tmp_path/"legacy"), device_name="cpu",
                       batch_size=1, replay_capacity=1, eval_every=500, eval_episodes=1,
                       epsilon_start=0., epsilon_end=0.)
    assert len(targets) == 1
    # Greedy up bumps into a wall: -0.01 - 0.1, with no gamma * 8.
    assert targets[0].item() == pytest.approx(-0.11)
