# SPDX-License-Identifier: GPL-3.0-or-later
"""Unattended sequential experiment runner and final uploadable report."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .marathon_train import VARIANTS
from .marathon_report import eval_checkpoints, generate_report


@contextmanager
def prevent_sleep():
    """Keep macOS awake during a foreground suite; release on exit or failure."""
    if sys.platform != "darwin":
        yield
        return
    process = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def atomic_json(path: Path, data: dict) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(partial, path)


def run_suite(output: str | Path, *, steps=400000, repeats=2, seed=42,
              eval_every=20000, save_every=10000, eval_episodes=20,
              test_episodes=100, transfer_episodes=50,
              reference: str | None = None, device="auto", resume_suite=False,
              report_only=False) -> Path:
    if repeats < 1 or steps < 1 or eval_every < 1 or save_every < 1:
        raise ValueError("Invalid experiment budget or interval")
    if test_episodes < 1 or transfer_episodes < 1:
        raise ValueError("Invalid final test count")
    root = Path(output).resolve()
    manifest_path = root / "manifest.json"
    if resume_suite or report_only:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Suite manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Suite folder is not empty: {root}. Use --resume-suite.")
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "suite_version": "0.5.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": {"steps": steps, "repeats": repeats, "seed": seed,
                         "eval_every": eval_every, "save_every": save_every,
                         "eval_episodes": eval_episodes, "test_episodes": test_episodes,
                         "transfer_episodes": transfer_episodes, "reference": reference,
                         "device": device},
            "runs": [{"variant": variant, "seed": seed + i * 1000,
                      "run_dir": f"{variant}_seed{seed + i * 1000}",
                      "status": "pending"}
                     for i in range(repeats) for variant in VARIANTS]
        }
        atomic_json(manifest_path, manifest)
    cfg = manifest["settings"]
    print(f"Suite root: {root} | {len(manifest['runs'])} runs | "
          f"{cfg['steps']} steps/run | resume={resume_suite}", flush=True)
    if not report_only:
        for run in manifest["runs"]:
            path = root / run["run_dir"]
            last = path / "last.pt"
            if not resume_suite and run["status"] != "pending":
                raise RuntimeError("Run state inconsistent; use --resume-suite")
            # Repair an interrupted run based on checkpoint, not stale manifest status.
            checkpoint = None
            recovery = last if last.exists() else path / "best.pt"
            if recovery.exists():
                from .marathon_train import checkpoint_read, resume_best_score
                from .dqn import select_device
                payload = checkpoint_read(recovery, select_device("cpu"))
                resume_best_score(payload, path, select_device("cpu"), recovery)
                if (run["status"] == "completed" or payload["steps"] >= cfg["steps"]) and (path / "best.pt").exists():
                    run["status"] = "completed"
                    atomic_json(manifest_path, manifest)
                    continue
                checkpoint = str(recovery)
            elif path.exists() and any(path.iterdir()):
                # A run can have only a log/config before first checkpoint.
                # No learning progress is recoverable, clean up these artifacts.
                import shutil
                shutil.rmtree(path)
            run["status"] = "running"
            run["started_utc"] = datetime.now(timezone.utc).isoformat()
            run.pop("error", None)
            atomic_json(manifest_path, manifest)
            command = [sys.executable, "-u", "-m", "pacman_lab.marathon_train",
                       "--variant", run["variant"], "--steps", str(cfg["steps"]),
                       "--seed", str(run["seed"]), "--output", str(path),
                       "--eval-every", str(cfg["eval_every"]),
                       "--save-every", str(cfg["save_every"]),
                       "--eval-episodes", str(cfg["eval_episodes"]),
                       "--device", cfg["device"]]
            if checkpoint is not None:
                command += ["--resume", checkpoint]
            print(f"\nSTART {run['run_dir']} {'(warm restart)' if checkpoint else ''}", flush=True)
            logpath = root / f"{run['run_dir']}.log"
            with logpath.open("a", encoding="utf-8", buffering=1) as log:
                with subprocess.Popen(command, cwd=Path.cwd(), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      errors="replace", bufsize=1) as worker:
                    assert worker.stdout is not None
                    for line in worker.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    returncode = worker.wait()
            run["completed_utc"] = datetime.now(timezone.utc).isoformat()
            if returncode:
                run["status"] = "failed"
                run["error"] = f"Worker exited with status {returncode}, see {logpath.name}"
                atomic_json(manifest_path, manifest)
                print(f"FAILED: {run['error']}", flush=True)
                raise RuntimeError(run["error"])
            if not (path / "best.pt").exists():
                raise RuntimeError(f"Worker completed without best.pt: {path}")
            run["status"] = "completed"
            atomic_json(manifest_path, manifest)
            print(f"COMPLETE {run['run_dir']}", flush=True)
    checks = []
    for run in manifest["runs"]:
        checkpoint = root / run["run_dir"] / "best.pt"
        if checkpoint.exists():
            checks.append({"label": run["run_dir"], "variant": run["variant"],
                           "seed": run["seed"], "run_dir": run["run_dir"],
                           "checkpoint": str(checkpoint), "reference": False})
    if cfg.get("reference"):
        checks.append({"label": "legacy_490k", "variant": "legacy_490k",
                       "seed": None, "run_dir": "", "checkpoint": cfg["reference"],
                       "reference": True})
    if not checks:
        raise RuntimeError("No completed checkpoints available for reporting")
    print(f"\nFinal held-out evaluation: {len(checks)} checkpoints", flush=True)
    summary = eval_checkpoints(checks, root, episodes=cfg["test_episodes"],
                               transfer_episodes=cfg["transfer_episodes"],
                               seed=20000, device_name=cfg["device"])
    summary["runs"] = manifest["runs"]
    summary["settings"] = cfg
    try:
        import torch
        summary["environment"] = {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "requested_device": cfg["device"]}
    except ImportError:
        summary["environment"] = {"python": sys.version.split()[0]}
    # Avoid bundling identifying local absolute paths in the portable report.
    for agent in summary["agents"]:
        if agent.get("reference"):
            agent["checkpoint"] = "external_reference/" + Path(agent["checkpoint"]).name
        else:
            agent["checkpoint"] = str(Path(agent["run_dir"]) / "best.pt")
    report_dir = generate_report(summary, root)
    print(f"\nFINISHED. Report: {report_dir / 'report.md'}", flush=True)
    print(f"UPLOAD THIS: {root / 'UPLOAD_THIS_report.zip'}", flush=True)
    return root


def main():
    p = argparse.ArgumentParser(description="Run several RL variants sequentially and generate uploadable report")
    p.add_argument("--output", type=Path,
                   default=None, help="Suite directory (default timestamped runs/marathon_...)")
    p.add_argument("--steps", type=int, default=400000, help="Training steps PER variant and seed")
    p.add_argument("--repeats", type=int, default=2, help="Independent seeds per variant")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--test-episodes", type=int, default=100)
    p.add_argument("--transfer-episodes", type=int, default=50)
    p.add_argument("--reference", type=str, help="Optional legacy 490k best.pt reference")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--resume-suite", type=Path, help="Existing suite directory; skip completed and warm-restart partial runs")
    p.add_argument("--report-only", type=Path, help="Make report for available checkpoints without training")
    a = p.parse_args()
    if a.resume_suite and a.report_only:
        p.error("Specify only one of --resume-suite and --report-only")
    if a.output and (a.resume_suite or a.report_only):
        p.error("--output is not used with --resume-suite or --report-only")
    root = (a.resume_suite or a.report_only or a.output or
            Path("runs") / f"marathon_{datetime.now():%Y%m%d_%H%M%S}")
    with prevent_sleep():
        run_suite(root, steps=a.steps, repeats=a.repeats, seed=a.seed,
                  eval_every=a.eval_every, save_every=a.save_every,
                  eval_episodes=a.eval_episodes, test_episodes=a.test_episodes,
                  transfer_episodes=a.transfer_episodes,
                  reference=a.reference, device=a.device,
                  resume_suite=bool(a.resume_suite), report_only=bool(a.report_only))


if __name__ == "__main__":
    main()
