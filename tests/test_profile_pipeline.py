# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration smoke tests for the added profiler (CPU, no Rust compiler needed)."""
from pathlib import Path
import json


def test_profile_report_and_grads(tmp_path: Path):
    from pacman_lab.profile_pipeline import run_profile
    from pacman_lab.profile_report import create_report

    result = run_profile(architecture="cnn_single", engine="python", device_name="cpu",
                         steps=32, warmup=8, batch_size=4, replay_capacity=64,
                         train_every=4, target_every=16, seed=6,
                         output=tmp_path / "profile_cnn_single_python.json")
    assert result["counts"]["updates"] >= 1
    assert result["settings"]["device"] == "cpu"
    assert result["settings"]["gpu_sync_each_section"] is False
    assert result["settings"]["cuda_sync_each_section"] is False
    assert result["gpu_name"] is None
    assert result["steps_per_second"] > 0
    archive = create_report(tmp_path)
    assert archive.is_file() and archive.stat().st_size > 0
    assert (tmp_path / "PROFILE_REPORT.md").exists()
