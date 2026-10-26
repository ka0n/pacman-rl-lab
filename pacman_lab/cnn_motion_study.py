# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-registered, paired-seed from-scratch control vs motion experiment.

Usage:
    python -m pacman_lab.cnn_motion_study plan
    python -m pacman_lab.cnn_motion_study run

Training/validation seeds are separated from the final held-out audit. Existing
fine-tuned checkpoints are never included as control conditions. Interrupted
runs fail closed because the old trainer's warm restart resets replay and RNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .cnn_batch import run_batched
from .cnn_v09 import evaluate_checkpoints
from .marathon_train import checkpoint_read

SEEDS = (930042, 930043, 930044)
STEPS = 1_000_000
VALIDATION_EPISODES = 100
VALIDATION_EVERY = 25_000
AUDIT_EPISODES = 1_000
AUDIT_SEED = 620_000

# The run_batched defaults are pinned here to make configuration verifiable.
HYPERPARAMS = {
    "steps": STEPS, "maze": "large", "worlds": 16,
    "learning_rate": 0.00015, "epsilon_start": 1.0,
    "epsilon_end": 0.08, "epsilon_decay_fraction": 0.85,
    "death_penalty": -20.0, "risk_shaping_weight": 0.0,
    "eval_every": VALIDATION_EVERY, "save_every": VALIDATION_EVERY,
    "eval_episodes": VALIDATION_EPISODES,
    "replay_capacity": 30000, "batch_size": 64, "warmup": 2000,
    "train_every": 8, "target_every": 2000,
    "burnin": 4, "unroll": 8,
}


def verify_checkpoint(path: Path, *, architecture: str, seed: int, engine: str,
                      device="cpu", matched_motion=False, complete_dir=None):
    """Reject fine-tuned, altered, unfinished or mismatched scratch runs."""
    import torch
    path = Path(path)
    payload = checkpoint_read(path, torch.device(device))
    cfg = payload.get("config", {})
    if payload.get("architecture") != architecture:
        raise ValueError(f"{path}: wrong architecture")
    if cfg.get("architecture") != architecture:
        raise ValueError(f"{path}: config architecture mismatch")
    if payload.get("fine_tune") is not None:
        raise ValueError(f"{path}: not a from-scratch model (fine_tune present)")
    for key, expected in (list(HYPERPARAMS.items()) + [("seed", seed), ("engine", engine)]):
        actual = cfg.get(key, 0 if key == "risk_shaping_weight" else None)
        if actual != expected:
            raise ValueError(f"{path}: mismatched {key}: {actual!r} != {expected!r}")
    if bool(cfg.get("matched_scratch_init", False)) != matched_motion:
        raise ValueError(f"{path}: matched scratch initialization mismatch")
    expected_shape = (5 if architecture == "cnn_motion" else 4, 11, 13)
    if tuple(payload.get("input_shape", ())) != expected_shape:
        raise ValueError(f"{path}: expected shape {expected_shape}")
    if not 0 < int(payload.get("step", 0)) <= STEPS:
        raise ValueError(f"{path}: checkpoint step outside study budget")
    if complete_dir is not None:
        last = Path(complete_dir) / "last.pt"
        # Completion is evidence only when last.pt has the same provenance.
        final = verify_checkpoint(last, architecture=architecture, seed=seed, engine=engine,
                                  device=device, matched_motion=matched_motion)
        if final['source_step'] != STEPS:
            raise ValueError(f"{last}: training not complete")
    return {"checkpoint": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_step": int(payload.get("step", 0)),
            "architecture": architecture, "seed": seed}


def _run_dir(root: Path, architecture: str, seed: int):
    return root / f"{architecture}_seed{seed}"


def prepare_study(root: Path, baseline: Path, engine: str, *, verbose=True):
    """Plan and preflight without modifying any training files."""
    existing = verify_checkpoint(baseline, architecture="cnn_single",
                                 seed=SEEDS[0], engine=engine,
                                 complete_dir=baseline.parent)
    tasks = []
    checkpoints = []
    for seed in SEEDS:
        for architecture in ("cnn_single", "cnn_motion"):
            matched = architecture == "cnn_motion"
            if seed == SEEDS[0] and architecture == "cnn_single":
                checkpoints.append(existing)
                if verbose: print(f"REUSE verified completed scratch control: {baseline}", flush=True)
                continue
            folder = _run_dir(root, architecture, seed)
            best = folder / "best.pt"
            last = folder / "last.pt"
            if best.exists() and last.exists():
                try:
                    info = verify_checkpoint(best, architecture=architecture,
                                             seed=seed, engine=engine,
                                             matched_motion=matched,
                                             complete_dir=folder)
                except (FileNotFoundError, ValueError) as exc:
                    raise ValueError(f"Incomplete/incompatible run at {folder}: {exc}") from exc
                checkpoints.append(info)
                if verbose: print(f"REUSE completed scratch run: {best}", flush=True)
            elif folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"Partial run at {folder}; move it aside before rerunning. "
                                      "Do not warm-resume a paired experiment.")
            else:
                tasks.append((architecture, seed, folder, matched))
                if verbose: print(f"TRAIN {architecture}, seed {seed}, {STEPS:,} transitions -> {folder}", flush=True)
    return tasks, checkpoints


def write_summary(audit: dict, path: Path):
    records = {}
    for result in audit["results"]:
        arch = result["architecture"]
        seed = int(result["source_seed"])
        records[(arch, seed)] = result["metrics"]["large"]
    summary = {"heldout_episodes_per_run": AUDIT_EPISODES,
               "heldout_seed": AUDIT_SEED,
               "results": {}, "paired_motion_minus_control_pp": {}}
    for arch in ("cnn_single", "cnn_motion"):
        values = [records[(arch,seed)]["win_percent"] for seed in SEEDS]
        summary["results"][arch] = {
            "win_percent_per_seed": dict(zip(map(str, SEEDS), values)),
            "mean_win_percent": statistics.mean(values),
            "sample_stddev_pp": statistics.stdev(values),
            "mean_pellet_percent": statistics.mean(records[(arch, seed)]["pellet_percent"]
                                                     for seed in SEEDS),
        }
    deltas = {str(seed): round(records[("cnn_motion",seed)]["win_percent"] -
                               records[("cnn_single",seed)]["win_percent"], 4)
              for seed in SEEDS}
    summary["paired_motion_minus_control_pp"] = deltas
    summary["mean_paired_difference_pp"] = statistics.mean(deltas.values())
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== HELD-OUT RESULTS (3 paired seeds) ===", flush=True)
    for arch, vals in summary["results"].items():
        print(f"{arch:12} mean wins {vals['mean_win_percent']:.2f}% "
              f"(seed SD {vals['sample_stddev_pp']:.2f} pp), "
              f"pellets {vals['mean_pellet_percent']:.2f}%", flush=True)
    print(f"Motion - control, paired mean: {summary['mean_paired_difference_pp']:+.2f} pp", flush=True)
    print("Individual seed deltas:", deltas, flush=True)
    print("Caution: three seeds are exploratory, not a proof of superiority.", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--root", type=Path, default=Path("runs/v094_motion_scratch"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("runs/v092_scratch_death20/best.pt"))
    parser.add_argument("--engine", choices=("rust", "python"), default="rust")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args(argv)
    tasks, _ = prepare_study(args.root, args.baseline, args.engine)
    print(f"Remaining training: {len(tasks)} x {STEPS:,} transitions. "
          f"Final evaluation: {len(SEEDS)*2} models x {AUDIT_EPISODES} games, "
          f"held-out seeds {AUDIT_SEED}..{AUDIT_SEED+AUDIT_EPISODES-1}", flush=True)
    if args.command == "plan":
        return
    if (args.root / "audit_620000.json").exists():
        raise FileExistsError("Final audit already exists. Do not repeatedly peek at the held-out test.")
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = args.root / "study_protocol.json"
    expected_protocol = {"version":"0.9.4", "seeds":list(SEEDS),
                         "params":HYPERPARAMS, "audit_seed": AUDIT_SEED,
                         "audit_episodes": AUDIT_EPISODES,
                         "control_reuse": str(args.baseline.resolve()),
                         "engine":args.engine, "device":args.device}
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8")) != expected_protocol:
            raise ValueError("Experiment protocol changed since its first invocation")
    else:
        protocol.write_text(json.dumps(expected_protocol, indent=2), encoding="utf-8")
    for arch, seed, folder, matched in tasks:
        print(f"\n=== {arch} / seed {seed} / matched init {matched} ===",flush=True)
        run_batched(architecture=arch,seed=seed,output=folder,
                    engine=args.engine,device_name=args.device,
                    matched_scratch_init=matched,**HYPERPARAMS)
        verify_checkpoint(folder / "best.pt", architecture=arch,seed=seed,
                          engine=args.engine,matched_motion=matched,complete_dir=folder)
    tasks_remaining, verified = prepare_study(args.root, args.baseline, args.engine,
                                              verbose=False)
    if tasks_remaining:
        raise RuntimeError("Not all runs completed: " + str(tasks_remaining))
    ordered = []
    for seed in SEEDS:
        for arch in ("cnn_single", "cnn_motion"):
            ordered.append(next(Path(item["checkpoint"]) for item in verified
                                if item["seed"] == seed and item["architecture"] == arch))
    audit_file = args.root / "audit_620000.json"
    audit = evaluate_checkpoints(ordered, output=audit_file,episodes=AUDIT_EPISODES,
                                 mazes=("large",),seed=AUDIT_SEED,
                                 engine=args.engine,device_name=args.device,worlds=16)
    write_summary(audit,args.root / "summary_620000.json")
    print(f"All done. Final report: {args.root / 'summary_620000.json'}",flush=True)


if __name__ == "__main__":
    main()
