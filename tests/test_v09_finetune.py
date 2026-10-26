# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only compatibility and provenance checks against v0.8 checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pacman_lab.cnn_batch import run_batched
from pacman_lab.cnn_v09 import evaluate_checkpoints
from pacman_lab.marathon_train import checkpoint_read


def test_v09_finetune_preserves_source_resume_and_audit(tmp_path: Path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    base = tmp_path / "base"
    opts = dict(architecture="cnn_single", steps=64, seed=42, output=base,
                worlds=4, device_name="cpu", eval_every=32, save_every=32,
                eval_episodes=3, replay_capacity=64, batch_size=4,
                warmup=8, train_every=8, burnin=1, unroll=2,
                maze="small", engine="python")
    run_batched(**opts)
    cp = base / "best.pt"
    digest = hashlib.sha256(cp.read_bytes()).hexdigest()
    source = checkpoint_read(cp, torch.device("cpu"))
    assert source["config"]["version"] == "0.8.0"
    ft = tmp_path / "fine_tune"
    new = dict(opts, seed=5_000_042, steps=64, output=ft,
               learning_rate=0.00003, epsilon_start=0.10, epsilon_end=0.05,
               init_checkpoint=str(cp))
    run_batched(**new)
    out = checkpoint_read(ft / "last.pt", torch.device("cpu"))
    assert out["step"] == 64
    assert out["fine_tune"]["source_step"] == source["step"]
    assert out["fine_tune"]["sha256"] == digest
    assert out["fine_tune"]["optimizer_restored"] is True
    assert out["config"]["learning_rate"] == 0.00003
    assert (ft / "fine_tune.json").is_file()
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == digest
    assert out["optimizer"]["param_groups"][0]["lr"] == 0.00003
    run_batched(**dict(new, init_checkpoint=None, resume=str(ft / "last.pt")))
    with pytest.raises(ValueError, match="either"):
        run_batched(**dict(new, resume=str(ft / "last.pt")))
    with pytest.raises(FileExistsError):
        run_batched(**dict(new, init_checkpoint=None))
    result = evaluate_checkpoints([cp, ft / "best.pt"], output=tmp_path / "audit.json",
                                  episodes=5, mazes=("small",), seed=70000,
                                  device_name="cpu", engine="python", worlds=3)
    assert len(result["results"]) == 2
    assert result["results"][0]["sha256"] == digest
    assert set(result["results"][0]["metrics"]["small"]) >= {"wins", "episodes"}
    with pytest.raises(FileExistsError):
        evaluate_checkpoints([cp], output=tmp_path / "audit.json",
                              episodes=2, mazes=("small",), seed=70000,
                              device_name="cpu", engine="python", worlds=3)


def test_incompatible_source_rejected(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from pacman_lab.cnn_models import make_model
    from pacman_lab.marathon_train import atomic_save_torch
    model = make_model("cnn_single", (4, 7, 9))
    checkpoint = tmp_path / "bad.pt"
    atomic_save_torch({"architecture": "cnn_single", "input_shape": (4, 7, 9),
                       "model": model.state_dict(), "config": {"maze": "small", "seed": 42},
                       "step": 100}, checkpoint)
    with pytest.raises(ValueError, match="Source architecture mismatch"):
        run_batched(architecture="mlp_wide", steps=8, seed=22, output=tmp_path / "out",
                    worlds=2, device_name="cpu", engine="python", maze="small",
                    init_checkpoint=str(checkpoint))
