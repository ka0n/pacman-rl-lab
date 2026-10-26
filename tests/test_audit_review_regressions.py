# SPDX-License-Identifier: GPL-3.0-or-later
"""Follow-up edge cases identified by Sourcery on issue #8."""
import json

import pytest

from pacman_lab.core import PacmanGame
from pacman_lab.marathon_train import checkpoint_read, resume_best_score


def test_gru_capacity_uses_effective_world_count(tmp_path, monkeypatch):
    from pacman_lab import cnn_batch
    monkeypatch.setattr(cnn_batch, "eval_batched", lambda *args, **kwargs:
                        {"win_percent": 0., "pellet_percent": 0.})
    cnn_batch.run_batched(architecture="cnn_gru", steps=1, seed=7, output=tmp_path,
                         worlds=8, device_name="cpu", engine="python", maze="small",
                         replay_capacity=3, batch_size=1, warmup=1, burnin=1, unroll=2)
    assert checkpoint_read(tmp_path/"last.pt", "cpu")["config"]["worlds"] == 1


def test_profile_samples_live_recurrent_tail(tmp_path, monkeypatch):
    from pacman_lab import profile_pipeline
    monkeypatch.setattr(profile_pipeline, "PacmanGame", lambda *args, **kwargs:
                        PacmanGame(*args, **dict(kwargs, ghosts=0, max_steps=6)))
    result = profile_pipeline.run_profile(
        architecture="cnn_gru", engine="python", device_name="cpu", maze="small",
        steps=6, warmup=1, batch_size=1, replay_capacity=5, train_every=1,
        burnin=1, unroll=2, output=tmp_path/"profile.json")
    assert result["counts"]["updates"] == 2


def checkpoint(score, marker=1):
    import torch
    return {"step": 4, "steps": 4, "best_score": (score, score),
            "metrics": {"win_percent": score, "pellet_percent": score},
            "model": {"weight": torch.tensor([marker])}, "config": {"seed": 7}}


@pytest.mark.parametrize("stale", (False, True))
def test_recover_legacy_last_before_best_crash(tmp_path, stale):
    import torch
    payload = checkpoint(20, 2)
    torch.save(payload, tmp_path/"last.pt")
    if stale:
        torch.save(checkpoint(10), tmp_path/"best.pt")
    assert resume_best_score(payload, tmp_path, "cpu") == (20, 20)
    best = checkpoint_read(tmp_path/"best.pt", "cpu")
    assert best["best_score"] == (20, 20)
    assert torch.equal(best["model"]["weight"], payload["model"]["weight"])


def test_external_resume_copies_newer_sibling_best(tmp_path):
    import torch
    source, output = tmp_path/"source", tmp_path/"output"
    source.mkdir()
    output.mkdir()
    saved = checkpoint(10)
    torch.save(saved, source/"last.pt")
    torch.save(checkpoint(20, 2), source/"best.pt")
    before = (source/"best.pt").read_bytes()
    assert resume_best_score(saved, output, "cpu", source/"last.pt") == (20, 20)
    best = checkpoint_read(output/"best.pt", "cpu")
    assert best["model"]["weight"].item() == 2
    assert (source/"best.pt").read_bytes() == before


def test_missing_historical_best_does_not_fabricate_weights(tmp_path):
    payload = checkpoint(20)
    payload["metrics"] = {"win_percent": 10, "pellet_percent": 10}
    with pytest.raises(ValueError, match="recover"):
        resume_best_score(payload, tmp_path, "cpu")
    assert not (tmp_path/"best.pt").exists()


@pytest.mark.parametrize("suite", ("cnn", "marathon"))
@pytest.mark.parametrize("state", ("lone_best", "legacy_missing", "legacy_stale"))
def test_suites_recover_before_marking_complete(tmp_path, monkeypatch, suite, state):
    import torch
    from pacman_lab import cnn_marathon, marathon
    module = cnn_marathon if suite == "cnn" else marathon
    run = tmp_path/"run"
    run.mkdir()
    if state == "lone_best":
        torch.save(checkpoint(20, 2), run/"best.pt")
    else:
        torch.save(checkpoint(20, 2), run/"last.pt")
        if state == "legacy_stale":
            torch.save(checkpoint(10), run/"best.pt")
    manifest = {"settings": {"steps": 4, "engine": "python", "device": "cpu",
                             "test_episodes": 1, "transfer_episodes": 1},
                "runs": [{"run_dir": "run", "status": "running",
                          "variant": "plain_dqn", "seed": 7}]}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    if suite == "cnn":
        monkeypatch.setattr(module, "build_report", lambda root: root)
    else:
        monkeypatch.setattr(module, "eval_checkpoints", lambda *args, **kwargs: {"agents": []})
        monkeypatch.setattr(module, "generate_report", lambda summary, root: root)
        monkeypatch.setattr(module.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("Recover completed checkpoints without spawning training"))
    module.run_suite(tmp_path, **({"resume": True} if suite == "cnn" else {"resume_suite": True}))
    assert checkpoint_read(run/"best.pt", "cpu")["best_score"] == (20, 20)
    assert json.loads((tmp_path/"manifest.json").read_text())["runs"][0]["status"] == "completed"


def test_multigame_cadence_override_cannot_leak(native_available):
    from pacman_lab.native import NativeBatch
    with NativeBatch("small", count=2, ghosts=0) as batch:
        with pytest.raises(ValueError, match="single-game"):
            batch.reset_one(0, seed=7, ghost_every=3)
        assert batch.ghost_every == 1


@pytest.fixture
def native_available():
    from pacman_lab.native import _load_library
    try:
        _load_library()
    except FileNotFoundError:
        pytest.skip("Build native simulator first")
