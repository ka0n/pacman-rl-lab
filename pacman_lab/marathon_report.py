# SPDX-License-Identifier: GPL-3.0-or-later
"""Portable user-uploadable research report with figures, raw metrics and provenance."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .baseline import evaluate_random
from .core import MAPS
from .dqn import make_qnet, select_device
from .marathon_train import evaluate_agent, checkpoint_read
from .memory import FrameHistory, erase_ghost_motion, infer_frames


def measure_motion_sensitivity(model, device, maze: str, frames: int,
                               seed: int, max_states=200) -> dict | None:
    """Ghost motion-history ablation on trajectories actually reached by agent.

    Deliberately changes prior ghost planes, NOT the current game state. These
    counterfactual histories may be impossible. Report only input dependence.
    """
    import torch
    from .core import PacmanGame

    if frames == 1:
        return None
    game = PacmanGame(maze=maze, ghosts=2)
    history = FrameHistory(frames)
    changed, deltas, seen = 0, [], 0
    with torch.inference_mode():
        for ep in range(20):
            state = history.reset(game.reset(seed=seed + ep)[0])
            while not game.done and seen < max_states:
                tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_before = model(tensor)[0]
                erased = erase_ghost_motion(state, frames)
                altered = torch.as_tensor(erased, dtype=torch.float32, device=device).unsqueeze(0)
                q_after = model(altered)[0]
                changed += int(int(q_before.argmax()) != int(q_after.argmax()))
                deltas.append(float((q_before - q_after).abs().max().item()))
                action = int(q_before.argmax())
                state = history.step(game.step(action).observation)
                seen += 1
            if seen >= max_states:
                break
    return {"sampled_decisions": seen,
            "action_change_percent": 100 * changed / max(1, seen),
            "mean_max_q_delta": statistics.mean(deltas) if deltas else 0.0,
            "note": "Ablating historical ghost positions measures network sensitivity, not ghost prediction or safety."}


def eval_checkpoints(run_paths: list[dict], suite_dir: Path, *, episodes=100,
                     seed=20000, transfer_episodes=50, device_name="auto") -> dict:
    from .core import PacmanGame
    import torch

    device = select_device(device_name)
    evaluations = []
    mazes = ("small", "unseen_a", "unseen_b")
    random_metrics = {}
    for i, maze in enumerate(mazes):
        s = seed if i == 0 else 40000 + (i - 1) * 1000
        n = episodes if i == 0 else transfer_episodes
        random_metrics[maze] = evaluate_random(episodes=n, maze=maze, ghosts=2, seed=s)
    for item in run_paths:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.exists():
            evaluations.append({**item, "error": f"Missing checkpoint {checkpoint}"})
            continue
        payload = checkpoint_read(checkpoint, device)
        shape = tuple(payload["input_shape"])
        frames = infer_frames(shape)
        net = make_qnet(shape).to(device)
        net.load_state_dict(payload["model"])
        net.eval()
        metrics = {}
        for i, maze in enumerate(mazes):
            expected = PacmanGame(maze=maze, ghosts=2).observe().shape
            if shape != (expected[0] * frames, expected[1], expected[2]):
                raise ValueError(f"Incompatible maze {maze} for checkpoint {checkpoint}")
            s = seed if i == 0 else 40000 + (i - 1) * 1000
            n = episodes if i == 0 else transfer_episodes
            metrics[maze] = evaluate_agent(net, device, maze=maze, episodes=n, seed=s, frames=frames)
        motion = measure_motion_sensitivity(net, device, "small", frames, seed=50000)
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        evaluations.append({**item, "sha256": checksum,
                            "steps": payload.get("steps"), "frames": frames,
                            "algorithm": payload.get("variant", payload.get("mode", "legacy")),
                            "evaluations": metrics, "motion_history_ablation": motion})
        print(f"Evaluated {item['label']}: small wins={metrics['small']['win_percent']:.1f}% "
              f"pellets={metrics['small']['pellet_percent']:.1f}%", flush=True)
    return {"report_version": "0.5.0", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "training_maze": "small", "ghosts": 2, "episodes_main": episodes,
            "episodes_transfer_per_maze": transfer_episodes, "main_seed": seed,
            "transfer_seeds": {"unseen_a": 40000, "unseen_b": 41000},
            "random": random_metrics, "agents": evaluations,
            "cautions": ["Unseen seeds are NOT unseen maze geometry; unseen_a and unseen_b are separate topologies of equal shape.",
                         "Checkpoint selection uses 20 validation seeds and can overfit them.",
                         "Two training seeds do not establish statistical significance.",
                         "Three-frame stacking supplies observed recent positions; no recurrent state or explicit motion predictor.",
                         "Ablation interventions can be synthetic/unreachable and do not prove ghost anticipation.",
                         "Resume is warm restart: replay buffer, RNG and episode state are not fully restored."]}


def _metrics_rows(run_dir: Path):
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_svg(summary: dict, suite_dir: Path) -> str:
    colors = {"plain_dqn": "#5d89d8", "memory_dqn": "#e6a23c",
              "memory_double": "#27b291", "legacy_490k": "#a78bfa"}
    points = []
    for agent in summary["agents"]:
        if agent.get("reference"):
            continue
        path = suite_dir / agent["run_dir"]
        rows = _metrics_rows(path)
        xy = [(float(row["step"]), float(row["win_percent"])) for row in rows]
        if xy:
            points.append((agent["label"], agent["variant"], xy))
    w, h = 910, 400
    max_x = max((x for _, _, data in points for x, _ in data), default=1)
    def coord(x, y):
        return (65 + x / max_x * 810, 350 - y / 100 * 300)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Win-rate by training step">',
           '<rect width="910" height="400" fill="#101a2e"/>',
           '<text x="65" y="25" fill="#fff" font-size="15">Validation win rate (%), 20 games per point</text>']
    for yy in (0, 20, 40, 60, 80, 100):
        sx, sy = coord(0, yy)
        out.append(f'<line x1="65" y1="{sy:.1f}" x2="875" y2="{sy:.1f}" stroke="#344357"/>')
        out.append(f'<text x="35" y="{sy+4:.1f}" fill="#ced6e3" font-size="12">{yy}</text>')
    for label, variant, data in points:
        poly = " ".join(f"{coord(x,y)[0]:.1f},{coord(x,y)[1]:.1f}" for x,y in data)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{colors[variant]}" stroke-width="2.0" opacity="0.8"/>')
    for ix, (_, label, _) in enumerate([(None, "plain_dqn", 0), (None, "memory_dqn", 1), (None, "memory_double", 2)]):
        x = 65 + ix * 265
        out.append(f'<rect x="{x}" y="371" width="14" height="4" fill="{colors[label]}"/>')
        out.append(f'<text x="{x+20}" y="378" fill="#dae4f1" font-size="12">{html.escape(label)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def generate_report(summary: dict, suite_dir: Path, *, include_bundle=True) -> Path:
    report = suite_dir / "report"
    report.mkdir(exist_ok=True, parents=True)
    (report / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report / "learning_curves.svg").write_text(build_svg(summary, suite_dir), encoding="utf-8")
    status_counts = defaultdict(int)
    for run in summary.get("runs", []):
        status_counts[run["status"]] += 1
    lines = ["# Pac-Man RL Lab – Marathon Research Report", "",
             f"Generated: {summary['timestamp_utc']} (UTC)", "",
             f"Environment: {json.dumps(summary.get('environment', {}), ensure_ascii=False)}", "",
             "## Experiment design", "",
             "- **Environment:** small 7×9 maze, two ghosts, fixed game rewards; all networks start without strategy demonstrations.",
             "- **Control:** plain DQN with one observed frame; **Memory DQN:** three sequential frames; **Memory Double DQN:** three frames with Double DQN targets.",
             "- **Architecture:** Same 128–128 hidden-layer MLP; input sizes differ as needed for memory; 4 actions.",
             "- **Budget:** Equal configured environment-step count per variant and seed. These runs are independent fresh starts (not warmstarts of the legacy model).",
             "- **Checkpoint selection:** Lexicographic validation wins, then pellets, on fixed 20 validation episodes (seed 10000).",
             f"- **Held-out:** {summary['episodes_main']} small-maze episodes from seed {summary['main_seed']}; {summary['episodes_transfer_per_maze']} per unseen topology.",
             "- **Controls:** Random agent with same action space; optional legacy 490k checkpoint for context only (unequal training budget).",
             "- **Resume caveat:** Interrupted runs resume weights/optimizer, not replay buffer, RNG state or an active episode.", "",
             f"## Completion status: {len(summary.get('runs', []))} runs", "",
             f"{dict(status_counts)}", "",
             "## Held-out results", "",
             "| Model | Steps | Frames | Small pellets % | Small wins % | Unseen A wins % | Unseen B wins % | Wall/episode |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    random = summary["random"]
    lines.append(f"| Uniform random | — | 0 | {random['small']['avg_pellet_percent']:.1f} | "
                 f"{random['small']['win_percent']:.1f} | {random['unseen_a']['win_percent']:.1f} | "
                 f"{random['unseen_b']['win_percent']:.1f} | {random['small']['wall_bumps_per_episode']:.1f} |")
    grouped = defaultdict(list)
    for agent in summary["agents"]:
        if "evaluations" not in agent:
            lines.append(f"| {agent['label']} | — | — | ERROR | — | — | — | — |")
            continue
        m = agent["evaluations"]
        s = m["small"]
        lines.append(f"| {agent['label']} | {agent['steps']} | {agent['frames']} | "
                     f"{s['pellet_percent']:.1f} | {s['win_percent']:.1f} "
                     f"({s['wins']}/{s['episodes']}) | "
                     f"{m['unseen_a']['win_percent']:.1f} | {m['unseen_b']['win_percent']:.1f} | "
                     f"{s['wall_bumps_per_episode']:.1f} |")
        if not agent.get("reference"):
            grouped[agent["variant"]].append(s)
    lines += ["", "## Group means across independent training seeds", "",
              "These averages are descriptive; small numbers of seeds are not formal significance tests.", "",
              "| Variant | Runs | Mean pellets % | Mean wins % |",
              "|---|---:|---:|---:|"]
    for variant, metrics in sorted(grouped.items()):
        lines.append(f"| {variant} | {len(metrics)} | "
                     f"{statistics.mean(x['pellet_percent'] for x in metrics):.1f} | "
                     f"{statistics.mean(x['win_percent'] for x in metrics):.1f} |")
    if len(grouped) == 3:
        p = statistics.mean(x["win_percent"] for x in grouped["plain_dqn"])
        m = statistics.mean(x["win_percent"] for x in grouped["memory_dqn"])
        d = statistics.mean(x["win_percent"] for x in grouped["memory_double"])
        lines += ["", "## Exploratory differences", "",
                  f"- Adding three-frame history to vanilla DQN: **{m-p:+.1f} percentage points** "
                  "(memory_dqn minus plain_dqn, mean held-out win rate).",
                  f"- Adding Double DQN to the memory model: **{d-m:+.1f} percentage points** "
                  "(memory_double minus memory_dqn).",
                  "- These are **descriptive differences**, not statistical significance claims. "
                  "The MLP input parameter count also differs for frame-stacked agents."]
    lines += ["", "## Ghost movement history ablation", "",
              "For memory agents, earlier ghost-position channels are replaced with the **current** ghost channel. "
              "The current observation and player/pellet histories stay the same. "
              "This probes input sensitivity, **not** optimal ghost avoidance or a valid alternative ghost trajectory.", "",
              "| Agent | Changed action % | Mean maximum ΔQ (absolute) | Samples |",
              "|---|---:|---:|---:|"]
    for agent in summary["agents"]:
        m = agent.get("motion_history_ablation")
        if m:
            lines.append(f"| {agent['label']} | {m['action_change_percent']:.1f} | "
                         f"{m['mean_max_q_delta']:.3f} | {m['sampled_decisions']} |")
    lines += ["", "## Model provenance (SHA-256 of best checkpoints)", "",
              "The digest identifies the exact checkpoint analyzed; checkpoints themselves are not in the upload bundle.", ""]
    for agent in summary["agents"]:
        if "sha256" in agent:
            lines.append(f"- `{agent['label']}`: `{agent['sha256']}`")
    lines += ["", "## Validation learning curves", "", "![Training curves](learning_curves.svg)", "",
              "## Interpretation and limitations", "",
              "- A temporal input allows estimates of recent ghost motion; it does not guarantee direction tracking or planning.",
              "- No training on unseen maze layouts. Transfer is zero-shot and may fail badly.",
              "- The historical 490k reference has a different step budget and is **not** a controlled algorithm comparison.",
              "- Best checkpoints are selected using the validation seeds. Final held-out test seeds were excluded from selection.",
              "- Training with a random action schedule cannot guarantee convergence or optimality.",
              "- Sampling noise remains in 100-episode outcome rates; repeat across more independent training seeds for scientific conclusions.",
              "- This demonstration environment is inspired by Pac-Man, not the commercial arcade game.", "",
              "## Reproduce & inspect", "",
              "The report bundle contains `summary.json`, per-run `config.json` and `metrics.csv` and a preview chart. "
              "Full PyTorch checkpoints remain on your PC; do not upload them unless requested.", "",
              "Example local playback: `python -m pacman_lab.play --mode dqn --checkpoint "
              "runs\\marathon_...\\memory_double_seed42\\best.pt --ghosts 2 --fps 8`", ""]
    (report / "report.md").write_text('\n'.join(lines), encoding="utf-8")
    md = '\n'.join(lines)
    # HTML has a readable comparison table, not only a raw Markdown text dump.
    html_rows = []
    for agent in summary["agents"]:
        if "evaluations" not in agent:
            continue
        m = agent["evaluations"]
        html_rows.append("<tr><td>" + html.escape(agent["label"]) + "</td>" +
                         "".join(f"<td>{m[maze][key]:.1f}%</td>" for maze, key in
                                 [("small", "win_percent"), ("small", "pellet_percent"),
                                  ("unseen_a", "win_percent"), ("unseen_b", "win_percent")]) + "</tr>")
    html_table = ('<table><tr><th>Agent</th><th>Small wins</th><th>Small pellets</th>'
                  '<th>Unseen A wins</th><th>Unseen B wins</th></tr>' +
                  ''.join(html_rows) + '</table>')
    rendered = ('<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>Pac-Man RL Marathon</title>'
                '<style>body{max-width:1050px;margin:40px auto;background:#101a2e;color:#e9edf2;'
                'font:15px/1.55 system-ui}table{border-collapse:collapse;width:100%;margin:16px 0}'
                'td,th{border-bottom:1px solid #36465b;padding:9px;text-align:left}a{color:#8db9ff}'
                'pre{white-space:pre-wrap}</style><h1>Pac-Man RL Marathon</h1>'
                '<p>The primary report is <code>report.md</code> (Markdown); raw data is in '
                '<code>summary.json</code>. Graph and data tables are available in this package.</p>'
                '<h2>Held-out test results</h2>' + html_table +
                '<img alt="Training curves" style="width:100%" src="learning_curves.svg"/>'
                '<h2>Methodology, interpretation and limitations</h2><pre>'+html.escape(md)+'</pre></html>')
    (report / "report.html").write_text(rendered, encoding="utf-8")
    if include_bundle:
        for run in summary.get("runs", []):
            rel = Path(run["run_dir"])
            for name in ("config.json", "metrics.csv"):
                original = suite_dir / rel / name
                if original.exists():
                    dest = report / "runs" / rel.name / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(original.read_bytes())
        archive = suite_dir / "UPLOAD_THIS_report.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(report.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(report))
    return report
