# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
import json
import zipfile

import numpy as np
import pytest

from pacman_lab.core import PacmanGame
from pacman_lab.memory import FrameHistory, erase_ghost_motion, infer_frames


def test_frame_stack_reset_and_time_order():
    obs = PacmanGame(maze="small", ghosts=2, seed=4).observe()
    h = FrameHistory(3)
    state0 = h.reset(obs)
    assert state0.shape == (12, 7, 9)
    for k in (0, 4, 8):
        assert np.array_equal(state0[k:k + 4], obs)
    changed = obs.copy()
    changed[3] = 0
    changed[3, 2, 2] = 1
    state1 = h.step(changed)
    assert np.array_equal(state1[0:4], obs)
    assert np.array_equal(state1[4:8], obs)
    assert np.array_equal(state1[8:12], changed)
    erased = erase_ghost_motion(state1, 3)
    assert np.array_equal(erased[3], changed[3])
    assert np.array_equal(erased[7], changed[3])
    assert np.array_equal(erased[11], changed[3])
    assert np.array_equal(erased[2], state1[2])
    assert infer_frames(state1.shape) == 3
    assert infer_frames(obs.shape) == 1


def test_memory_validations():
    with pytest.raises(ValueError):
        FrameHistory(0)
    with pytest.raises(RuntimeError):
        FrameHistory(2).observe()
    with pytest.raises(ValueError):
        infer_frames((5, 7, 9))


@pytest.mark.parametrize("variant,frames", [("plain_dqn", 1), ("memory_dqn", 3),
                                            ("memory_double", 3)])
def test_marathon_checkpoint_evaluation_and_report(tmp_path, variant, frames):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from pacman_lab.marathon_train import run_marathon_training, checkpoint_read, evaluate_agent
    from pacman_lab.marathon_report import eval_checkpoints, generate_report
    from pacman_lab.dqn import make_qnet
    folder = tmp_path / variant
    run_marathon_training(variant=variant, steps=14, seed=42, output=folder,
                          device_name="cpu", eval_every=7, save_every=4,
                          eval_episodes=2, batch_size=4, replay_capacity=10,
                          train_warmup=4)
    checkpoint = folder / "best.pt"
    payload = checkpoint_read(checkpoint, "cpu")
    assert payload["input_shape"] == (4 * frames, 7, 9)
    assert (folder / "last.pt").exists()
    assert payload["variant"] == variant
    model = make_qnet(tuple(payload["input_shape"]))
    model.load_state_dict(payload["model"])
    m = evaluate_agent(model, torch.device("cpu"), episodes=2, frames=frames)
    assert 0 <= m["win_percent"] <= 100
    entry = [{"label": variant, "variant": variant, "seed": 42,
              "run_dir": variant, "checkpoint": str(checkpoint), "reference": False}]
    summary = eval_checkpoints(entry, tmp_path, episodes=2,
                               transfer_episodes=1, device_name="cpu")
    assert len(summary["agents"]) == 1
    assert summary["agents"][0]["frames"] == frames
    if frames == 1:
        assert summary["agents"][0]["motion_history_ablation"] is None
    summary["runs"] = [{"run_dir": variant, "status": "completed"}]
    report = generate_report(summary, tmp_path)
    assert "Held-out results" in (report / "report.md").read_text(encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "UPLOAD_THIS_report.zip") as z:
        assert "report.md" in z.namelist()
        assert "summary.json" in z.namelist()
        assert f"runs/{variant}/metrics.csv" in z.namelist()
        assert not any(n.endswith(".pt") for n in z.namelist())


def test_run_suite_sequential(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from pacman_lab.marathon import run_suite
    root = tmp_path / "suite"
    run_suite(root, steps=4, repeats=1, seed=42,
              eval_every=4, save_every=4, eval_episodes=1,
              test_episodes=1, transfer_episodes=1, device="cpu")
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["runs"]) == 3
    assert all(r["status"] == "completed" for r in manifest["runs"])
    assert (root / "UPLOAD_THIS_report.zip").exists()
    run_suite(root, device="cpu", resume_suite=True)
