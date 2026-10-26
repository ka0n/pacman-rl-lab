# SPDX-License-Identifier: GPL-3.0-or-later
"""Safety and protocol regressions for the full architecture marathon."""
import json

import pytest


def test_plan_has_all_five_architectures_and_three_shared_seeds():
    from pacman_lab.cnn_marathon import ARCHITECTURES, TRAINING_SEEDS, _runs

    runs = _runs()
    assert len(runs) == 15
    assert [run["architecture"] for run in runs[:3]] == [ARCHITECTURES[0]] * 3
    assert set(run["architecture"] for run in runs) == {
        "mlp_wide", "cnn_single", "cnn_history4", "cnn_gru", "cnn_motion"}
    assert all([r["seed"] for r in runs if r["architecture"] == arch] ==
               list(TRAINING_SEEDS) for arch in ARCHITECTURES)
    assert all(run["status"] == "pending" and not run["attempts"] for run in runs)


def test_uncheckpointed_attempt_is_preserved_and_retried_in_fresh_directory(tmp_path):
    from pacman_lab.cnn_marathon import _attempt_paths, _runs

    item = _runs(("cnn_single",), (731026,))[0]
    old = tmp_path / item["run_id"] / "attempt_01"
    old.mkdir(parents=True)
    marker = old / "training.log"
    marker.write_text("preserve me")
    item["attempts"].append({"number": 1, "run_dir": old.relative_to(tmp_path).as_posix(),
                             "status": "failed"})

    new, recovery, warm = _attempt_paths(tmp_path, item)
    assert new == tmp_path / item["run_id"] / "attempt_02"
    assert recovery is None and not warm
    assert marker.read_text() == "preserve me"


def test_checkpoint_snapshot_never_overwrites_existing_file(tmp_path):
    torch = pytest.importorskip("torch")
    from pacman_lab.cnn_batch import _write_checkpoint_snapshot

    target = tmp_path / "step_000025000.pt"
    torch.save({"step": 25_000}, target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_checkpoint_snapshot({"step": 50_000}, target)
    assert target.read_bytes() == original


def test_batched_training_persists_requested_checkpoint_snapshot(tmp_path, monkeypatch):
    from pacman_lab import cnn_batch

    monkeypatch.setattr(cnn_batch, "eval_batched", lambda *args, **kwargs: {
        "win_percent": 0.0, "pellet_percent": 12.5, "wins": 0, "episodes": 1,
        "wall_bumps_per_episode": 0.0, "stationary_percent": 0.0,
        "reversals_per_episode": 0.0, "steps_per_episode": 4.0})
    output = tmp_path / "run"
    cnn_batch.run_batched(architecture="cnn_single", steps=4, seed=731026,
                          output=output, worlds=1, device_name="cpu", engine="python",
                          maze="small", warmup=10, replay_capacity=16,
                          batch_size=1, eval_every=4, save_every=4,
                          eval_episodes=1)
    import torch
    snapshot = output / "checkpoints" / "step_000000004.pt"
    assert torch.load(snapshot, map_location="cpu", weights_only=True)["step"] == 4


def test_heldout_seed_exclusivity_scan_passes_for_reserved_window(tmp_path):
    from pacman_lab.cnn_marathon import HELDOUT_EPISODES, HELDOUT_SEED, _verify_heldout_window

    record = _verify_heldout_window(tmp_path)
    assert record["checked"] is True
    assert record["seed_start"] == HELDOUT_SEED
    assert record["episodes"] == HELDOUT_EPISODES


def test_suite_records_cnn_motion_worker_and_completed_checkpoints(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from pacman_lab import cnn_marathon

    commands = []
    monkeypatch.setattr(cnn_marathon, "_verify_heldout_window",
                        lambda _root: {"checked": True})
    monkeypatch.setattr(cnn_marathon, "build_report", lambda root: root)

    def fake_worker(command, *, cwd, log, on_sample):
        commands.append(command)
        output = __import__("pathlib").Path(command[command.index("--output") + 1])
        architecture = command[command.index("--architecture") + 1]
        seed = int(command[command.index("--seed") + 1])
        settings = cnn_marathon._settings(
            steps=4, eval_every=2, save_every=4, eval_episodes=1,
            worlds=1, engine="python", device="cpu", warmup=0,
            replay_capacity=128, batch_size=4)
        item = {"architecture": architecture, "seed": seed, "run_id": f"{architecture}_seed{seed}"}
        payload = {"architecture": architecture, "step": 4,
                   "config": cnn_marathon._expected_checkpoint_config(item, settings)}
        torch.save(payload, output / "best.pt")
        torch.save(payload, output / "last.pt")
        (output / "checkpoints").mkdir()
        torch.save(payload, output / "checkpoints" / "step_000000004.pt")
        log.write("finished synthetic runner test\n")
        return 0

    monkeypatch.setattr(cnn_marathon, "_monitor_worker", fake_worker)
    root = tmp_path / "suite"
    result = cnn_marathon.run_suite(
        root, steps=4, seeds=(731026,), architectures=("cnn_motion",),
        engine="python", device="cpu", eval_every=2, save_every=4,
        eval_episodes=1, worlds=1, warmup=0, replay_capacity=128, batch_size=4)
    assert result == root.resolve()
    assert commands[0][commands[0].index("--architecture") + 1] == "cnn_motion"
    manifest = json.loads((root / "manifest.json").read_text())
    run = manifest["runs"][0]
    assert run["status"] == "completed"
    assert run["attempts"][0]["restart_kind"] == "fresh"
    assert len(run["attempts"][0]["checkpoint_paths"]) == 3


def test_manifest_records_fixed_test_window_and_full_run_metadata(tmp_path):
    from pacman_lab.cnn_marathon import (
        ARCHITECTURES, HELDOUT_EPISODES, HELDOUT_SEED, TRAINING_SEEDS,
        _new_manifest, _settings,
    )

    manifest = _new_manifest(_settings(), ARCHITECTURES, TRAINING_SEEDS)
    assert len(manifest["runs"]) == 15
    assert manifest["heldout"] == {
        "seed_start": HELDOUT_SEED, "episodes_per_maze": HELDOUT_EPISODES,
        "mazes": ["large", "large_unseen_a", "large_unseen_b"], "exclusive": True}
    assert "cnn_motion" in manifest["model_versions"]
    assert all("parameters" in v and "spec" in v
               for v in manifest["model_versions"].values())
    assert all("checkpoint_paths" not in run for run in manifest["runs"])
    assert json.loads(json.dumps(manifest))["training_seeds"] == list(TRAINING_SEEDS)
