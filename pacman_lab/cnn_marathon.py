# SPDX-License-Identifier: GPL-3.0-or-later
"""Resumable, sequential five-architecture Pac-Man experiment runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .cnn_models import SPECS, make_model, parameters_count
from .cnn_train import eval_model
from .cnn_batch import eval_batched
from .dqn import select_device
from .marathon import prevent_sleep, atomic_json
from .marathon_train import checkpoint_read
from .core import MAPS, PacmanGame

TRAIN_MAZE = "large"
TEST_MAZES = ("large", "large_unseen_a", "large_unseen_b")
ARCHITECTURES = ("mlp_wide", "cnn_single", "cnn_history4", "cnn_gru", "cnn_motion")
TRAINING_SEEDS = (731026, 732026, 733026)
HELDOUT_SEED = 9_600_000
HELDOUT_EPISODES = 1_000
RUNNER_VERSION = "1.0.0"


def _runs(architectures=ARCHITECTURES, seeds=TRAINING_SEEDS):
    """Architecture-major plan with identical seed set for every architecture."""
    return [{"architecture": architecture, "seed": seed,
             "run_id": f"{architecture}_seed{seed}",
             "run_dir": f"{architecture}_seed{seed}",
             "status": "pending", "attempts": [], "restart_count": 0,
             "best_checkpoint": None, "last_checkpoint": None,
             "evaluation_path": None}
            for architecture in architectures for seed in seeds]


def _git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _model_versions():
    import torch
    versions = {}
    for arch in ARCHITECTURES:
        frames = SPECS[arch].frames
        channels = 5 if arch == "cnn_motion" else 4 * frames
        model = make_model(arch, (channels, len(MAPS[TRAIN_MAZE]),
                                  len(MAPS[TRAIN_MAZE][0])))
        versions[arch] = {"spec": {"frames": frames,
                                    "recurrent": SPECS[arch].recurrent},
                          "parameters": parameters_count(model),
                          "checkpoint_version": "0.8.0"}
    return versions


def _settings(*, steps=1_000_000, eval_every=50_000, save_every=25_000,
              eval_episodes=40, worlds=16, engine="rust", device="mps",
              warmup=2_000, replay_capacity=30_000, batch_size=64,
              train_every=8, target_every=2_000, burnin=4, unroll=8,
              learning_rate=0.00015,
              epsilon_start=1.0, epsilon_end=0.08,
              epsilon_decay_fraction=0.85):
    if (steps < 1 or eval_every < 1 or save_every < 1 or eval_episodes < 1 or
            worlds < 1 or warmup < 0 or replay_capacity < batch_size or
            batch_size < 1 or burnin < 0 or unroll < 1 or engine not in ("rust", "python") or
            device not in ("mps", "cpu", "cuda", "auto")):
        raise ValueError("Invalid marathon configuration")
    return {"steps": steps, "maze": TRAIN_MAZE, "worlds": worlds,
            "engine": engine, "device": device, "eval_every": eval_every,
            "save_every": save_every, "eval_episodes": eval_episodes,
            "warmup": warmup, "replay_capacity": replay_capacity,
            "batch_size": batch_size, "train_every": train_every,
            "target_every": target_every, "burnin": burnin, "unroll": unroll,
            "learning_rate": learning_rate,
            "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
            "epsilon_decay_fraction": epsilon_decay_fraction,
            "death_penalty": None, "risk_shaping_weight": 0.0,
            "checkpoint_selection": "lexicographic validation wins, then pellets; fixed 10000 seed base"}


def _new_manifest(settings, architectures=ARCHITECTURES, seeds=TRAINING_SEEDS):
    import torch
    return {"runner_version": RUNNER_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "prepared", "code_revision": _git_revision(),
            "model_versions": _model_versions(), "settings": settings,
            "training_seeds": list(seeds),
            "heldout": {"seed_start": HELDOUT_SEED,
                        "episodes_per_maze": HELDOUT_EPISODES,
                        "mazes": list(TEST_MAZES), "exclusive": True},
            "environment": {"python": sys.version.split()[0],
                            "platform": platform.platform(),
                            "torch": torch.__version__,
                            "mps_built": torch.backends.mps.is_built(),
                            "mps_available": torch.backends.mps.is_available(),
                            "disk_free_bytes_at_start": shutil.disk_usage(Path.cwd()).free,
                            "gpu_utilization": "not exposed by non-privileged runner"},
            "runs": _runs(architectures, seeds)}


def _verify_heldout_window(root: Path):
    """Reject an already referenced test seed in repository docs or prior runs."""
    scan_roots = [Path.cwd() / "docs", Path.cwd() / "runs",
                  Path.cwd() / "pacman_lab"]
    pattern = re.compile(r"(?<!\d)\d{7,}(?!\d)")
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                    ".json", ".csv", ".md", ".txt", ".log"}:
                continue
            try:
                resolved = path.resolve()
                if resolved == Path(__file__).resolve():
                    continue
                if resolved == root or root in resolved.parents:
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for token in pattern.findall(content):
                seed = int(token)
                if HELDOUT_SEED <= seed < HELDOUT_SEED + HELDOUT_EPISODES:
                    raise ValueError(f"Held-out seed {seed} already occurs in {path}")
    return {"checked": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "seed_start": HELDOUT_SEED, "episodes": HELDOUT_EPISODES,
            "scanned_roots": [str(p.relative_to(Path.cwd())) for p in scan_roots if p.exists()],
            "training_world_seed_ranges": "< 3,000,000; held-out begins at 9,600,000"}


@contextmanager
def _suite_lock(root: Path):
    """Prevent two runner instances from mutating the same experiment concurrently."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".runner.lock"
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Experiment runner already active: {root}") from exc
        yield


def _safe_relative(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Manifest path escapes experiment directory: {value}")
    return path


def _attempt_paths(root: Path, item: dict):
    attempts = item.setdefault("attempts", [])
    if attempts:
        prior = attempts[-1]
        run_path = _safe_relative(root, prior["run_dir"])
        candidates = [run_path / "last.pt", run_path / "best.pt"]
        candidates.extend(sorted((run_path / "checkpoints").glob("step_*.pt")))
        found = []
        for candidate in candidates:
            if candidate.exists():
                try:
                    saved = checkpoint_read(candidate, select_device("cpu"))
                    found.append((int(saved.get("step", saved.get("steps", 0))), candidate))
                except Exception:
                    continue
        recovery = max(found, key=lambda pair: pair[0])[1] if found else None
        if recovery:
            return run_path, recovery, True
        if prior.get("status") == "running":
            prior["status"] = "interrupted"
            prior["finished_utc"] = datetime.now(timezone.utc).isoformat()
            prior["note"] = "Runner restarted; no valid checkpoint found, artifacts preserved"
    # No checkpoint is recoverable. Preserve all artifacts by starting a new
    # attempt directory rather than deleting or reusing a non-empty one.
    number = len(attempts) + 1
    run_path = _safe_relative(root, Path(item["run_id"]) / f"attempt_{number:02d}")
    while run_path.exists() and any(run_path.iterdir()):
        number += 1
        run_path = _safe_relative(root, Path(item["run_id"]) / f"attempt_{number:02d}")
    return run_path, None, False


def _resource_sample(pid: int):
    """Best-effort, non-privileged macOS CPU/RSS/thermal snapshot."""
    sample = {"utc": datetime.now(timezone.utc).isoformat(), "pid": pid}
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "%cpu=,rss="],
                                capture_output=True, text=True, timeout=2)
        values = result.stdout.split()
        if result.returncode == 0 and len(values) == 2:
            sample["cpu_percent"] = float(values[0])
            sample["rss_mb"] = round(int(values[1]) / 1024, 1)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                                    text=True, timeout=2)
            sample["thermal"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            sample["thermal"] = "unavailable"
    return sample


def _monitor_worker(command, *, cwd, log, on_sample):
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        try:
            last_sample = time.monotonic()
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if time.monotonic() - last_sample >= 60:
                    on_sample(_resource_sample(process.pid))
                    last_sample = time.monotonic()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            raise


def _load_or_create(root, *, resume, report_only, settings=None,
                    architectures=ARCHITECTURES, seeds=TRAINING_SEEDS):
    manifest_path = root / "manifest.json"
    if resume or report_only:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Experiment manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runner_version") != RUNNER_VERSION:
            raise ValueError("Unsupported or legacy manifest; refusing unsafe resume")
        if settings is not None and manifest["settings"] != settings:
            raise ValueError("Resume settings differ from the original manifest")
        if manifest.get("training_seeds") != list(seeds):
            raise ValueError("Resume seed list differs from the original manifest")
        if [r["architecture"] for r in manifest["runs"]] != [
                a for a in architectures for _ in seeds]:
            raise ValueError("Resume architecture plan differs from the original manifest")
        return manifest
    if root.exists() and any(p.name != ".runner.lock" for p in root.iterdir()):
        raise FileExistsError(f"Experiment folder is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = _new_manifest(settings, architectures, seeds)
    atomic_json(manifest_path, manifest)
    return manifest


def _recover_legacy_completed(root: Path):
    """Repair only fully checkpointed legacy suites; never delete or resume them."""
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    settings = manifest.get("settings", {})
    for item in manifest.get("runs", []):
        run_path = _safe_relative(root, item["run_dir"])
        last, best = run_path / "last.pt", run_path / "best.pt"
        recovery = last if last.exists() else best if best.exists() else None
        if recovery is None:
            raise ValueError("Legacy suite has no checkpoint; preserved for manual recovery")
        payload = checkpoint_read(recovery, select_device("cpu"))
        step = int(payload.get("step", payload.get("steps", 0)))
        if step < int(settings.get("steps", 1)):
            raise ValueError("Legacy incomplete suite lacks safe restart metadata; no files changed")
        from .marathon_train import resume_best_score
        resume_best_score(payload, run_path, select_device("cpu"), recovery)
        item["status"] = "completed"
    atomic_json(path, manifest)
    return root


def _record(manifest, manifest_path):
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)


def _expected_checkpoint_config(item, settings):
    return {"architecture": item["architecture"], "seed": item["seed"],
            "maze": settings["maze"], "engine": settings["engine"],
            "worlds": settings["worlds"], "steps": settings["steps"],
            "eval_every": settings["eval_every"],
            "save_every": settings["save_every"],
            "eval_episodes": settings["eval_episodes"],
            "warmup": settings["warmup"],
            "replay_capacity": settings["replay_capacity"],
            "batch_size": settings["batch_size"],
            "train_every": settings["train_every"],
            "target_every": settings["target_every"],
            "learning_rate": settings["learning_rate"],
            "epsilon_start": settings["epsilon_start"],
            "epsilon_end": settings["epsilon_end"],
            "epsilon_decay_fraction": settings["epsilon_decay_fraction"],
            "burnin": settings["burnin"], "unroll": settings["unroll"]}


def _verify_checkpoint(path: Path, item: dict, settings: dict, *, complete=False):
    payload = checkpoint_read(path, select_device("cpu"))
    config = payload.get("config", {})
    expected = _expected_checkpoint_config(item, settings)
    mismatches = {key: (config.get(key), value) for key, value in expected.items()
                  if config.get(key) != value}
    if payload.get("architecture") != item["architecture"] or mismatches:
        raise ValueError(f"Checkpoint does not match manifest for {item['run_id']}: {mismatches}")
    step = int(payload.get("step", payload.get("steps", 0)))
    if not (0 < step <= settings["steps"]):
        raise ValueError(f"Invalid checkpoint step {step}: {path}")
    if complete and step != settings["steps"]:
        raise ValueError(f"Completed run checkpoint stops at {step}: {path}")
    return payload


def _summarize_resources(item):
    samples = [sample for attempt in item.get("attempts", [])
               for sample in attempt.get("resource_samples", [])]
    cpus = [sample["cpu_percent"] for sample in samples if "cpu_percent" in sample]
    rss = [sample["rss_mb"] for sample in samples if "rss_mb" in sample]
    thermal = list(dict.fromkeys(sample["thermal"] for sample in samples
                                 if sample.get("thermal") not in (None, "", "unavailable")))
    return {"sample_count": len(samples),
            "mean_cpu_percent": round(statistics.mean(cpus), 2) if cpus else None,
            "peak_cpu_percent": round(max(cpus), 2) if cpus else None,
            "peak_rss_mb": round(max(rss), 1) if rss else None,
            "thermal": thermal,
            "gpu_utilization": "not measured without a privileged sampler",
            "mps_memory": "not externally measurable by this runner"}


def build_report(root: Path):
    """Select validation-best checkpoints and evaluate on the locked test seeds."""
    import torch
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    device = select_device(settings["device"])
    evaluation_dir = root / "evaluations"
    evaluation_dir.mkdir(exist_ok=True)
    curves = []
    agents = []
    for item in manifest["runs"]:
        if item["status"] != "completed":
            continue
        best = _safe_relative(root, item["best_checkpoint"])
        payload = _verify_checkpoint(best, item, settings)
        checkpoint_hash = hashlib.sha256(best.read_bytes()).hexdigest()
        eval_path = evaluation_dir / f"{item['run_id']}.json"
        expected_eval = {"checkpoint_sha256": checkpoint_hash,
                         "seed_start": manifest["heldout"]["seed_start"],
                         "episodes": manifest["heldout"]["episodes_per_maze"],
                         "mazes": manifest["heldout"]["mazes"]}
        arch = item["architecture"]
        shape = tuple(payload["input_shape"])
        model = make_model(arch, shape).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        if eval_path.exists():
            cached = json.loads(eval_path.read_text(encoding="utf-8"))
            if any(cached.get(k) != v for k, v in expected_eval.items()):
                raise ValueError(f"Existing held-out results do not match: {eval_path}")
            scores = cached["metrics"]
            progress = cached.setdefault("progress", {})
        else:
            scores = {}
            progress = {}
        for maze in manifest["heldout"]["mazes"]:
            current = scores.get(maze)
            if isinstance(current, dict) and current.get("episodes") == manifest["heldout"]["episodes_per_maze"]:
                continue
            try:
                h, w = PacmanGame(maze=maze, ghosts=2).observe().shape[1:]
                if (h, w) != shape[1:]:
                    scores[maze] = {"status": "incompatible_shape",
                                    "observation_shape": [4, h, w],
                                    "checkpoint_shape": list(shape)}
                    atomic_json(eval_path, dict(expected_eval, run_id=item["run_id"],
                                                 metrics=scores, progress=progress))
                    continue
                chunks = progress.setdefault(maze, [])
                completed_offsets = {chunk["offset"] for chunk in chunks}
                evaluator = eval_batched if settings["worlds"] > 1 else eval_model
                kwargs = {"worlds": settings["worlds"]} if settings["worlds"] > 1 else {}
                for offset in range(0, manifest["heldout"]["episodes_per_maze"], 100):
                    if offset in completed_offsets:
                        continue
                    count = min(100, manifest["heldout"]["episodes_per_maze"] - offset)
                    result = evaluator(model, arch, device, maze=maze,
                        seed=manifest["heldout"]["seed_start"] + offset,
                        episodes=count, engine=settings["engine"], **kwargs)
                    chunks.append({"offset": offset, "episodes": count, "metrics": result})
                    chunks.sort(key=lambda x: x["offset"])
                    total = sum(chunk["episodes"] for chunk in chunks)
                    if total == manifest["heldout"]["episodes_per_maze"]:
                        wins = sum(chunk["metrics"]["wins"] for chunk in chunks)
                        fields = ("pellet_percent", "wall_bumps_per_episode",
                                  "stationary_percent", "reversals_per_episode",
                                  "steps_per_episode")
                        aggregate = {field: sum(chunk["metrics"][field] * chunk["episodes"]
                                                for chunk in chunks) / total
                                     for field in fields}
                        aggregate.update({"wins": wins, "episodes": total,
                                          "win_percent": 100 * wins / total})
                        scores[maze] = aggregate
                    atomic_json(eval_path, dict(expected_eval, run_id=item["run_id"],
                                                 metrics=scores, progress=progress))
                    print(f"EVAL {item['run_id']} {maze}: {min(total, manifest['heldout']['episodes_per_maze'])}/"
                          f"{manifest['heldout']['episodes_per_maze']} episodes", flush=True)
            except (KeyError, ValueError) as exc:
                scores[maze] = {"status": "unavailable", "reason": str(exc)}
                atomic_json(eval_path, dict(expected_eval, run_id=item["run_id"],
                                             metrics=scores, progress=progress))
        item["evaluation_path"] = eval_path.relative_to(root).as_posix()
        _record(manifest, root / "manifest.json")
        metrics_path = _safe_relative(root, item["run_dir"]) / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        curves.append((item["run_id"], item["architecture"], item["seed"], records))
        agents.append({"architecture": item["architecture"], "seed": item["seed"],
                       "run_id": item["run_id"], "run_dir": item["run_dir"],
                       "trained_steps": int(payload["step"]),
                       "parameters": manifest["model_versions"][item["architecture"]]["parameters"],
                       "checkpoint_path": item["best_checkpoint"],
                       "checkpoint_sha256": checkpoint_hash,
                       "validation_selection": {"rule": "highest win_percent, then pellet_percent",
                           "score": payload.get("best_score"),
                           "metrics": payload.get("metrics", {})},
                       "mean_logged_steps_per_second": statistics.mean(
                           [float(r["steps_per_second"]) for r in records
                            if r.get("steps_per_second") not in (None, "")]) if records else None,
                       "resources": _summarize_resources(item),
                       "metrics": scores})
        print(f"EVALUATED {item['run_id']}", flush=True)
    if not agents:
        raise RuntimeError("No completed checkpoints available for final report")
    summary = {"runner_version": RUNNER_VERSION,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "status_counts": {status: sum(r["status"] == status for r in manifest["runs"])
                                 for status in ("completed", "running", "failed", "interrupted", "pending")},
               "settings": settings, "heldout": manifest["heldout"],
               "model_versions": manifest["model_versions"], "agents": agents,
               "notes": ["Held-out seeds were locked before training and are distinct from training/validation seeds.",
                         "Three training seeds support only a bounded comparison, not a general architecture claim.",
                         "A warm restart resets replay, RNG, and environment states; such runs are marked in the manifest.",
                         "Different parameter counts confound architecture and capacity.",
                         "Training throughput includes the selected backend and is not a simulator-only benchmark."],
               "environment": manifest["environment"]}
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    atomic_json(report_dir / "summary.json", summary)
    colors = {"mlp_wide": "#2563eb", "cnn_single": "#ea580c",
              "cnn_history4": "#16a34a", "cnn_gru": "#9333ea",
              "cnn_motion": "#0891b2"}

    def plot_svg(metric, title, filename):
        width, height, left, right, top, bottom = 1100, 500, 72, 20, 45, 90
        iw, ih = width - left - right, height - top - bottom
        el = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{left}" y="25" font-size="18" font-family="sans-serif">{html.escape(title)}</text>']
        for y in (0, 25, 50, 75, 100):
            yy = top + ih * (1 - y / 100)
            el += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>',
                   f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12">{y}%</text>']
        dashes = {TRAINING_SEEDS[0]: "", TRAINING_SEEDS[1]: "6 3", TRAINING_SEEDS[2]: "2 3"}
        for run_id, arch, seed, records in curves:
            pts = []
            for row in records:
                try:
                    x = left + iw * float(row["step"]) / settings["steps"]
                    y = top + ih * (1 - min(100, max(0, float(row[metric]))) / 100)
                except (ValueError, KeyError, TypeError):
                    continue
                pts.append(f"{x:.2f},{y:.2f}")
            if pts:
                dash = f' stroke-dasharray="{dashes.get(seed, "")}"' if dashes.get(seed) else ""
                el.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[arch]}" '
                          f'stroke-width="1.4" opacity=".65"{dash}><title>{run_id}</title></polyline>')
        for i, arch in enumerate(ARCHITECTURES):
            x = left + (i % 3) * 205
            y = height - 45 + (i // 3) * 18
            el += [f'<line x1="{x}" y1="{y}" x2="{x+18}" y2="{y}" stroke="{colors[arch]}" stroke-width="3"/>',
                   f'<text x="{x+24}" y="{y+4}" font-size="12" font-family="sans-serif">{arch} (3 seeds)</text>']
        el.append("</svg>")
        (report_dir / filename).write_text("\n".join(el), encoding="utf-8")

    plot_svg("win_percent", "Validation win rate", "learning_wins.svg")
    plot_svg("pellet_percent", "Validation pellet collection", "learning_pellets.svg")
    md = ["# Pac-Man RL Lab — five-architecture comparison", "",
          f"- Complete runs: {summary['status_counts']['completed']}/{len(manifest['runs'])}",
          f"- Training: {settings['steps']:,} steps/run; MPS/Rust; {settings['worlds']} worlds; maze `{TRAIN_MAZE}`.",
          f"- Held-out: {manifest['heldout']['episodes_per_maze']:,} shared seeds beginning at "
          f"{manifest['heldout']['seed_start']:,} on each compatible maze.", "",
          "## Per-run held-out results", "",
          "| Architecture | Seed | Main wins % | Main pellets % | Unseen A wins % | Unseen B wins % | Steps/s | Status |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for a in agents:
        main = a["metrics"].get("large", {})
        ua = a["metrics"].get("large_unseen_a", {})
        ub = a["metrics"].get("large_unseen_b", {})
        md.append(f"| {a['architecture']} | {a['seed']} | {main.get('win_percent', 'n/a')} | "
                  f"{main.get('pellet_percent', 'n/a')} | {ua.get('win_percent', 'n/a')} | "
                  f"{ub.get('win_percent', 'n/a')} | {a['mean_logged_steps_per_second'] or 'n/a'} | completed |")
    md += ["", "## Across-seed summary (large)", "",
           "| Architecture | Runs | Win rate mean ± sample SD | Pellets mean ± sample SD |",
           "|---|---:|---:|---:|"]
    for arch in ARCHITECTURES:
        group = [a for a in agents if a["architecture"] == arch and
                 isinstance(a["metrics"].get("large", {}).get("win_percent"), (float, int))]
        if not group:
            continue
        wins = [a["metrics"]["large"]["win_percent"] for a in group]
        pellets = [a["metrics"]["large"]["pellet_percent"] for a in group]
        sdw = statistics.stdev(wins) if len(wins) > 1 else 0.0
        sdp = statistics.stdev(pellets) if len(pellets) > 1 else 0.0
        md.append(f"| {arch} | {len(group)} | {statistics.mean(wins):.2f} ± {sdw:.2f} | "
                  f"{statistics.mean(pellets):.2f} ± {sdp:.2f} |")
    md += ["", "## Learning curves", "", "![Validation wins](learning_wins.svg)", "",
           "![Validation pellets](learning_pellets.svg)", "",
           "## Interpretation", "",
           "Reported outcomes are measured results. Differences among only three seeds do not establish general architecture superiority.",
           "Lower transfer scores are observed generalization outcomes; they do not identify a cause. Wall bumps, stationary time, reversals, episode length and losses are included in raw JSON for cautious behavior analysis.",
          "Architecture and parameter count differ together, so capacity is a confound.",
          "GPU utilization is not available from the non-privileged runner. Per-run CPU/RSS samples and `pmset` thermal text, when returned by macOS, are in the manifest.", "",
          "## Resource use", "",
          "| Run | Mean sampled CPU % | Peak sampled RSS MiB | Thermal samples |", "|---|---:|---:|---|"]
    for a in agents:
        resource = a["resources"]
        thermal = "; ".join(resource.get("thermal", [])) or "unavailable"
        md.append(f"| {a['run_id']} | {resource.get('mean_cpu_percent', 'n/a')} | "
                  f"{resource.get('peak_rss_mb', 'n/a')} | {html.escape(thermal)} |")
    md += ["",
           "## Run completeness", "",
           "| Run | Status | Warm restarts | Checkpoint |", "|---|---|---:|---|"]
    for item in manifest["runs"]:
        md.append(f"| {item['run_id']} | {item['status']} | {item.get('restart_count', 0)} | "
                  f"{item.get('best_checkpoint') or '—'} |")
    md += ["", "## Viewer commands", ""]
    for a in agents:
        md.append(f"```sh\n.venv/bin/python -m pacman_lab.cnn_inspect watch --checkpoint "
                  f"{a['checkpoint_path']} --maze large --device mps\n```")
    text = "\n".join(md) + "\n"
    (report_dir / "report.md").write_text(text, encoding="utf-8")
    (report_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Pac-Man architecture comparison</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #bbb}img{max-width:100%}</style>"
        f"<h1>Pac-Man architecture comparison</h1><p>See <a href='report.md'>Markdown report</a>.</p>"
        "<img src='learning_wins.svg'><img src='learning_pellets.svg'>"
        f"<pre>{html.escape(text)}</pre>", encoding="utf-8")
    manifest["report_path"] = "report/report.md"
    manifest["status"] = "completed" if all(r["status"] == "completed" for r in manifest["runs"]) else "incomplete"
    _record(manifest, root / "manifest.json")
    return report_dir


def _run_suite_locked(root: Path, *, steps=1_000_000, seeds=TRAINING_SEEDS,
              architectures=ARCHITECTURES, engine="rust", device="mps",
              eval_every=50_000, save_every=25_000, eval_episodes=40,
              worlds=16, warmup=2_000, replay_capacity=30_000,
              batch_size=64, train_every=8, target_every=2_000,
              burnin=4, unroll=8,
              learning_rate=0.00015, epsilon_start=1.0, epsilon_end=0.08,
              epsilon_decay_fraction=0.85, resume=False, report_only=False):
    if len(set(seeds)) != len(seeds) or len(set(architectures)) != len(architectures):
        raise ValueError("Seeds and architectures must be unique")
    if not seeds or not architectures or any(a not in SPECS for a in architectures):
        raise ValueError("Unknown/empty architecture or seed plan")
    config = _settings(steps=steps, eval_every=eval_every, save_every=save_every,
                       eval_episodes=eval_episodes, worlds=worlds, engine=engine,
                       device=device, warmup=warmup, replay_capacity=replay_capacity,
                       batch_size=batch_size, train_every=train_every,
                       target_every=target_every, burnin=burnin, unroll=unroll,
                       learning_rate=learning_rate,
                       epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                       epsilon_decay_fraction=epsilon_decay_fraction)
    root = Path(root).resolve()
    exclusivity = None
    if not resume and not report_only:
        exclusivity = _verify_heldout_window(root)
    legacy_path = root / "manifest.json"
    if resume and legacy_path.exists():
        old = json.loads(legacy_path.read_text(encoding="utf-8"))
        if "runner_version" not in old:
            return _recover_legacy_completed(root)
    manifest = _load_or_create(root, resume=resume, report_only=report_only,
                               settings=None if report_only else config,
                               architectures=architectures, seeds=seeds)
    manifest_path = root / "manifest.json"
    if exclusivity is not None:
        manifest["heldout_exclusivity_check"] = exclusivity
        _record(manifest, manifest_path)
    if report_only:
        return build_report(root)
    if manifest["settings"]["device"] == "mps":
        device_obj = select_device("mps")
        if device_obj.type != "mps":
            raise RuntimeError("MPS requested but unavailable; CPU fallback is forbidden")
    if manifest["settings"]["engine"] == "rust":
        from .native_bench import verify
        verify(episodes=30)
    manifest["preflight"] = {"completed_utc": datetime.now(timezone.utc).isoformat(),
                            "mps_available": __import__("torch").backends.mps.is_available(),
                            "rust_parity": "passed" if config["engine"] == "rust" else "not_requested"}
    manifest["status"] = "running"
    _record(manifest, manifest_path)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    for item in manifest["runs"]:
        run_settings = manifest["settings"]
        base_dir = root / item["run_id"]
        if item["status"] == "completed":
            best = _safe_relative(root, item["best_checkpoint"])
            last = _safe_relative(root, item["last_checkpoint"])
            _verify_checkpoint(best, item, run_settings)
            _verify_checkpoint(last, item, run_settings, complete=True)
            continue
        attempt_dir, recovery, warm_restart = _attempt_paths(root, item)
        attempt_number = len(item.setdefault("attempts", [])) + 1
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{item['run_id']}_attempt_{attempt_number:02d}.log"
        attempt = {"number": attempt_number,
                   "run_dir": attempt_dir.relative_to(root).as_posix(),
                   "log_path": log_path.relative_to(root).as_posix(),
                   "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "restart_kind": ("warm_restart" if warm_restart else
                                    "fresh_retry" if item["attempts"] else "fresh"),
                   "resumed_from": recovery.relative_to(root).as_posix() if recovery else None,
                   "resource_samples": []}
        if recovery:
            recovered_payload = _verify_checkpoint(recovery, item, run_settings)
            recovered_step = int(recovered_payload["step"])
            snapshot_dir = attempt_dir / "checkpoints"
            snapshot_dir.mkdir(exist_ok=True)
            recovered_snapshot = snapshot_dir / f"step_{recovered_step:09d}.pt"
            if not recovered_snapshot.exists():
                from .marathon_train import atomic_save_torch
                atomic_save_torch(recovered_payload, recovered_snapshot)
            # Preserve original CSV evidence before resume reconciliation.
            metrics = attempt_dir / "metrics.csv"
            if metrics.exists():
                snapshot = attempt_dir / f"metrics_before_restart_{attempt_number:02d}.csv"
                if not snapshot.exists():
                    snapshot.write_bytes(metrics.read_bytes())
            item["restart_count"] = item.get("restart_count", 0) + 1
        item["attempts"].append(attempt)
        item["run_dir"] = attempt_dir.relative_to(root).as_posix()
        item["status"] = "running"
        item["last_checkpoint"] = (attempt_dir / "last.pt").relative_to(root).as_posix()
        item["best_checkpoint"] = (attempt_dir / "best.pt").relative_to(root).as_posix()
        _record(manifest, manifest_path)
        command = [sys.executable, "-u", "-m", "pacman_lab.cnn_train",
                   "--architecture", item["architecture"], "--steps", str(run_settings["steps"]),
                   "--maze", run_settings["maze"], "--seed", str(item["seed"]),
                   "--output", str(attempt_dir), "--device", run_settings["device"],
                   "--engine", run_settings["engine"],
                   "--eval-every", str(run_settings["eval_every"]),
                   "--save-every", str(run_settings["save_every"]),
                   "--eval-episodes", str(run_settings["eval_episodes"]),
                   "--worlds", str(run_settings["worlds"]),
                   "--warmup", str(run_settings["warmup"]),
                   "--replay-capacity", str(run_settings["replay_capacity"]),
                   "--batch-size", str(run_settings["batch_size"]),
                   "--train-every", str(run_settings["train_every"])]
        command += ["--target-every", str(run_settings["target_every"]),
                    "--burnin", str(run_settings["burnin"]),
                    "--unroll", str(run_settings["unroll"]),
                    "--learning-rate", str(run_settings["learning_rate"]),
                    "--epsilon-start", str(run_settings["epsilon_start"]),
                    "--epsilon-end", str(run_settings["epsilon_end"]),
                    "--epsilon-decay-fraction", str(run_settings["epsilon_decay_fraction"])]
        if recovery:
            command += ["--resume", str(recovery)]
        print(f"START {item['run_id']} ({attempt['restart_kind']})", flush=True)
        def save_sample(sample):
            attempt["resource_samples"].append(sample)
            _record(manifest, manifest_path)
        try:
            with log_path.open("x", encoding="utf-8", buffering=1) as log:
                return_code = _monitor_worker(command, cwd=Path.cwd(), log=log,
                                              on_sample=save_sample)
            if return_code:
                raise RuntimeError(f"Worker exited with status {return_code}")
            best = attempt_dir / "best.pt"
            last = attempt_dir / "last.pt"
            if not best.exists() or not last.exists():
                raise RuntimeError("Training exited without both best.pt and last.pt")
            _verify_checkpoint(last, item, run_settings, complete=True)
            _verify_checkpoint(best, item, run_settings)
            snapshots = list((attempt_dir / "checkpoints").glob("step_*.pt"))
            expected_snapshots = (run_settings["steps"] + run_settings["save_every"] - 1) // run_settings["save_every"]
            if len(snapshots) < expected_snapshots:
                raise RuntimeError(f"Only {len(snapshots)}/{expected_snapshots} immutable step checkpoints exist")
            item["status"] = "completed"
            attempt["status"] = "completed"
            attempt["finished_utc"] = datetime.now(timezone.utc).isoformat()
            attempt["last_checkpoint_step"] = run_settings["steps"]
            attempt["checkpoint_paths"] = [
                p.relative_to(root).as_posix()
                for p in sorted((attempt_dir / "checkpoints").glob("step_*.pt"))]
            attempt["checkpoint_paths"] += [item["best_checkpoint"], item["last_checkpoint"]]
            _record(manifest, manifest_path)
        except BaseException as exc:
            attempt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            attempt["finished_utc"] = datetime.now(timezone.utc).isoformat()
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            item["status"] = attempt["status"]
            manifest["status"] = attempt["status"]
            _record(manifest, manifest_path)
            raise
    return build_report(root)


def run_suite(root: Path, **kwargs):
    root = Path(root).resolve()
    with _suite_lock(root):
        return _run_suite_locked(root, **kwargs)


def _handle_stop_signal(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}; preserving the last durable checkpoint")


def main(argv=None):
    p = argparse.ArgumentParser(description="Sequential reproducible five-architecture MPS/Rust experiment")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=list(TRAINING_SEEDS))
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES,
                   default=list(ARCHITECTURES))
    p.add_argument("--engine", choices=("python", "rust"), default="rust")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="mps")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--save-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--replay-capacity", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-every", type=int, default=8)
    p.add_argument("--target-every", type=int, default=2_000)
    p.add_argument("--burnin", type=int, default=4)
    p.add_argument("--unroll", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.00015)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.08)
    p.add_argument("--epsilon-decay-fraction", type=float, default=0.85)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume-suite", type=Path)
    p.add_argument("--report-only", type=Path)
    a = p.parse_args(argv)
    if sum(x is not None for x in (a.output, a.resume_suite, a.report_only)) > 1:
        p.error("Choose only one of --output/--resume-suite/--report-only")
    root = (a.output or a.resume_suite or a.report_only or
            Path("runs") / f"cnn_{datetime.now():%Y%m%d_%H%M%S}")
    args = dict(steps=a.steps, seeds=tuple(a.seeds), architectures=tuple(a.architectures),
                engine=a.engine, device=a.device, eval_every=a.eval_every,
                save_every=a.save_every, eval_episodes=a.eval_episodes,
                worlds=a.worlds, warmup=a.warmup, replay_capacity=a.replay_capacity,
                batch_size=a.batch_size, train_every=a.train_every,
                target_every=a.target_every, burnin=a.burnin, unroll=a.unroll,
                learning_rate=a.learning_rate,
                epsilon_start=a.epsilon_start, epsilon_end=a.epsilon_end,
                epsilon_decay_fraction=a.epsilon_decay_fraction,
                resume=a.resume_suite is not None, report_only=a.report_only is not None)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    with prevent_sleep():
        run_suite(root, **args)


if __name__ == "__main__":
    main()
