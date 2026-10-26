# SPDX-License-Identifier: GPL-3.0-or-later
"""Automated, resumable CNN/GRU comparison with an uploadable report ZIP."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import os
import platform
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .cnn_models import SPECS, make_model, parameters_count
from .cnn_train import eval_model
from .cnn_batch import eval_batched
from .dqn import select_device
from .marathon import prevent_sleep, atomic_json
from .marathon_train import checkpoint_read
from .core import PacmanGame

TRAIN_MAZE = "large"
TEST_MAZES = ("large", "large_unseen_a", "large_unseen_b")
LEGACY_MARATHON_ARCHITECTURES = ("mlp_wide", "cnn_single", "cnn_history4", "cnn_gru")


def _runs(repeats, base_seed):
    return [{"architecture": name, "seed": base_seed + 1000 * r,
             "run_dir": f"{name}_seed{base_seed+1000*r}","status":"pending"}
            for r in range(repeats) for name in LEGACY_MARATHON_ARCHITECTURES]


def build_report(root: Path):
    import torch
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    settings=manifest["settings"]
    device=select_device(settings["device"])
    agents=[]
    curves=[]
    for item in manifest["runs"]:
        path=root/item["run_dir"]
        best=path/"best.pt"
        if not best.exists(): continue
        checkpoint=checkpoint_read(best,device)
        arch=item["architecture"]
        shape=tuple(checkpoint["input_shape"])
        model=make_model(arch,shape).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        scores={}
        for index,maze in enumerate(TEST_MAZES):
            h,w = PacmanGame(maze=maze,ghosts=2).observe().shape[1:]
            if (h,w)!=shape[1:]:
                raise ValueError(f"Maze {maze} shape {(h,w)} does not match model {shape}")
            evaluation = eval_batched if settings.get("worlds", 1) > 1 else eval_model
            kwargs = {"worlds":settings["worlds"]} if settings.get("worlds",1)>1 else {}
            scores[maze]=evaluation(model,arch,device,maze=maze,
                            seed=20000+index*1000,
                            episodes=settings["test_episodes"] if index==0 else settings["transfer_episodes"],
                            engine=settings["engine"],**kwargs)
        metrics_path=path/"metrics.csv"
        with metrics_path.open(newline="",encoding="utf-8") as f:
            records=list(csv.DictReader(f))
        curves.append((item["run_dir"],arch,records))
        agents.append({"architecture":arch,"seed":item["seed"],
                       "run_dir":item["run_dir"],"trained_steps":checkpoint["step"],
                       "parameters":parameters_count(model),
                       "sha256":hashlib.sha256(best.read_bytes()).hexdigest(),
                       "mean_logged_steps_per_second":sum(float(r["steps_per_second"]) for r in records)/max(1,len(records)),
                       "metrics":scores})
        print(f"FINAL {item['run_dir']}: {scores[TRAIN_MAZE]['win_percent']:.1f}% wins",flush=True)
    if not agents: raise RuntimeError("No trained checkpoints available")
    summary={"version":"0.8.0", "created_utc":datetime.now(timezone.utc).isoformat(),
             "notes":["Held-out seeds differ from training and validation.",
                      "Different CNN/MLP/GRU sizes; architecture and parameter count are confounded.",
                      "Zero-shot map transfer only changes walls, not map dimensions.",
                      "Warm restart discards replay buffer and restarts RNG and episode.",
                      "Train/validation/test all share reward specification.",
                      "No frame-history or recurrent ablation is claimed without paired interventions."],
             "settings":settings,"agents":agents,
             "environment":{"python":sys.version.split()[0],"platform":platform.platform(),
                            "torch":torch.__version__,"cuda":torch.version.cuda,
                            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    directory=root/"report"
    directory.mkdir(exist_ok=True)
    (directory/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    md=["# CNN / GRU Pac-Man Marathon — final report", "",
        f"- Training: **{settings['steps']:,}** environment steps per run, "
        f"{settings['repeats']} independent seed(s), maze `{TRAIN_MAZE}`",
        f"- Engine: `{settings['engine']}`; device: `{settings['device']}`",
        f"- Final evaluation: {settings['test_episodes']} main-map episodes and "
        f"{settings['transfer_episodes']} episodes on each unseen topology", "",
        "## Held-out results", "",
        "| Agent | Seed | Params | Main win % | Main pellets % | Unseen A win % | Unseen B win % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for a in agents:
        main=a["metrics"]["large"]
        one=a["metrics"]["large_unseen_a"]
        two=a["metrics"]["large_unseen_b"]
        md.append(f"| {a['architecture']} | {a['seed']} | {a['parameters']:,} | "
                  f"{main['win_percent']:.1f} | {main['pellet_percent']:.1f} | "
                  f"{one['win_percent']:.1f} | {two['win_percent']:.1f} |")
    md += ["", "## Across-seed summary", "", "| Architecture | Runs | Mean win % ± sample SD | Mean pellet % ± sample SD |",
           "|---|---:|---:|---:|"]
    for name in LEGACY_MARATHON_ARCHITECTURES:
        group=[a for a in agents if a["architecture"]==name]
        if not group:continue
        wins=[a["metrics"]["large"]["win_percent"] for a in group]
        pellets=[a["metrics"]["large"]["pellet_percent"] for a in group]
        sdw=statistics.stdev(wins) if len(wins)>1 else 0.0
        sdp=statistics.stdev(pellets) if len(pellets)>1 else 0.0
        md.append(f"| {name} | {len(group)} | {statistics.mean(wins):.2f} ± {sdw:.2f} | "
                  f"{statistics.mean(pellets):.2f} ± {sdp:.2f} |")

    def plot_svg(metric, title, filename):
        colors={"mlp_wide":"#2563eb", "cnn_single":"#ea580c",
                "cnn_history4":"#16a34a", "cnn_gru":"#9333ea"}
        width,height=900,420
        left,right,top,bottom=64,20,46,62
        iw,ih=width-left-right,height-top-bottom
        elements=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="900">',
                  '<rect width="100%" height="100%" fill="#fff"/>',
                  f'<text x="{left}" y="25" fill="#222" font-family="Arial" font-size="18">{html.escape(title)}</text>']
        for y in (0,25,50,75,100):
            yy=top+ih*(1-y/100)
            elements += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>',
                         f'<text x="{left-7}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#444">{y}%</text>']
        for label, arch, rows in curves:
            pts=[]
            for row in rows:
                try:
                    x=left+iw*float(row["step"])/settings["steps"]
                    y=top+ih*(1-min(100,max(0,float(row[metric])))/100)
                except (ValueError,KeyError):continue
                pts.append(f"{x:.2f},{y:.2f}")
            if pts:
                elements.append(f'<polyline points="{" ".join(pts)}" stroke="{colors[arch]}" '
                                'stroke-width="1.8" opacity=".65" fill="none"/>')
                elements.append(f'<circle cx="{pts[-1].split(",")[0]}" cy="{pts[-1].split(",")[1]}" r="2.4" fill="{colors[arch]}"/>')
        for i,(name,color) in enumerate(colors.items()):
            x=left+i*200
            elements += [f'<rect x="{x}" y="{height-30}" width="12" height="10" fill="{color}"/>',
                         f'<text x="{x+17}" y="{height-21}" font-size="12" font-family="Arial" fill="#333">{html.escape(name)}</text>']
        elements += [f'<text x="{width/2}" y="{height-43}" text-anchor="middle" font-size="12" fill="#555">Environment steps (0–{settings["steps"]:,})</text>', '</svg>']
        (directory/filename).write_text("\n".join(elements),encoding="utf-8")

    plot_svg("win_percent","Validation win rate by training step","learning_wins.svg")
    plot_svg("pellet_percent","Validation pellet collection by training step","learning_pellets.svg")
    md += ["", "## Learning curves", "", "![Win rates](learning_wins.svg)", "",
           "![Pellet collection](learning_pellets.svg)", ""]
    md += ["", "## What this does *not* establish", "",
           "- A network that observes ghost positions is not necessarily predicting ghost motion.",
           "- A GRU advantage (if observed) may come from capacity or training stochasticity.",
           "- Transfer to unseen *same-sized* layouts does not prove general map generalization.",
           "- Simulator speed and training speed must be compared separately.","",
           "## Per-run details",""]
    for a in agents:
        md += [f"### {a['run_dir']}",
               f"Best checkpoint step: {a['trained_steps']:,}; SHA256: `{a['sha256']}`",
               f"Average logged throughput: {a['mean_logged_steps_per_second']:.1f} steps/sec.",""]
    (directory/"report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    (directory/"report.html").write_text("<!doctype html><meta charset='utf-8'><title>Pac-Man CNN Report</title>"
            "<style>body{font:16px system-ui;max-width:1050px;margin:3em auto;line-height:1.5}pre{white-space:pre-wrap}</style>"
            "<h1>Pac-Man CNN / GRU Report</h1><pre>"+html.escape("\n".join(md))+"</pre>",encoding="utf-8")
    artifact=root/"UPLOAD_THIS_CNN_report.zip"
    with zipfile.ZipFile(artifact,"w",zipfile.ZIP_DEFLATED) as z:
        for f in directory.iterdir():z.write(f,arcname=f"report/{f.name}")
        # Include per-run config+CSV only. Never include model weights/log files/absolute paths.
        for a in agents:
            for f in ("config.json","metrics.csv"):
                x=root/a["run_dir"]/f
                if x.exists(): z.write(x,arcname=f"runs/{a['run_dir']}/{f}")
        clean_manifest=dict(manifest)
        clean_manifest["settings"]=dict(settings)
        z.writestr("manifest.json",json.dumps(clean_manifest,indent=2))
    print(f"UPLOAD REPORT: {artifact}",flush=True)
    return artifact


def run_suite(root: Path, *,steps=1000000,repeats=3,seed=42,engine="python",device="auto",
              eval_every=50000,save_every=25000,eval_episodes=40,
              test_episodes=200,transfer_episodes=100,
              resume=False,report_only=False,worlds=1):
    if engine not in ("rust","python") or steps<1 or repeats<1 or eval_every<1 or save_every<1 or worlds<1:
        raise ValueError("Invalid marathon configuration")
    root=root.resolve(); manifest_path=root/"manifest.json"
    if resume or report_only:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if root.exists() and any(root.iterdir()):raise FileExistsError(root)
        root.mkdir(parents=True,exist_ok=True)
        manifest={"version":"0.8.0", "settings":{"steps":steps,"repeats":repeats,"seed":seed,"worlds":worlds,
                  "engine":engine,"device":device,"eval_every":eval_every,"save_every":save_every,
                  "eval_episodes":eval_episodes,"test_episodes":test_episodes,
                  "transfer_episodes":transfer_episodes},"runs":_runs(repeats,seed)}
        atomic_json(manifest_path,manifest)
    config=manifest["settings"]
    if config["engine"]=="rust":
        from .native_bench import verify
        print("Rust selected: checking deterministic Python/Rust parity first",flush=True)
        verify(episodes=3)
    if not report_only:
        for item in manifest["runs"]:
            path=root/item["run_dir"]
            last=path/"last.pt"
            if item["status"]=="completed" and (path/"best.pt").exists():continue
            checkpoint=None
            # best.pt is committed first; a crash can leave it as the only save.
            recovery = last if last.exists() else path/"best.pt"
            if recovery.exists():
                payload=checkpoint_read(recovery,select_device("cpu"))
                if int(payload["step"])>=config["steps"] and (path/"best.pt").exists():
                    item["status"]="completed"
                    atomic_json(manifest_path,manifest)
                    continue
                checkpoint=str(recovery)
            elif path.exists() and any(path.iterdir()):
                import shutil
                shutil.rmtree(path)
            item["status"]="running"
            atomic_json(manifest_path,manifest)
            command=[sys.executable,"-u","-m","pacman_lab.cnn_train",
                     "--architecture",item["architecture"],"--steps",str(config["steps"]),
                     "--maze",TRAIN_MAZE,"--seed",str(item["seed"]),
                     "--output",str(path),"--device",config["device"],
                     "--engine",config["engine"],"--eval-every",str(config["eval_every"]),
                     "--save-every",str(config["save_every"]),
                     "--eval-episodes",str(config["eval_episodes"])]
            if config.get("worlds",1)>1:command.extend(["--worlds",str(config["worlds"])])
            if checkpoint: command.extend(["--resume",checkpoint])
            print(f"START {item['run_dir']} {'resume' if checkpoint else ''}",flush=True)
            with (root/f"{item['run_dir']}.log").open("a",encoding="utf-8",buffering=1) as log:
                with subprocess.Popen(command,cwd=Path.cwd(),stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1) as process:
                    for line in process.stdout:
                        print(line,end="",flush=True)
                        log.write(line)
                    ret=process.wait()
            if ret:
                item["status"]="failed"
                item["error"]=f"Worker exit code {ret}; check log"
                atomic_json(manifest_path,manifest)
                raise RuntimeError(f"Job {item['run_dir']} failed, exit code {ret}")
            if not (path/"best.pt").exists():raise RuntimeError(f"No best.pt: {path}")
            item["status"]="completed"
            atomic_json(manifest_path,manifest)
    return build_report(root)


def main():
    p=argparse.ArgumentParser(description="Long-running CNN/GRU suite; creates uploadable report")
    p.add_argument("--steps",type=int,default=1000000)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--engine",choices=("python","rust"),default="python")
    p.add_argument("--device",choices=("auto","cpu","cuda","mps"),default="auto")
    p.add_argument("--eval-every",type=int,default=50000)
    p.add_argument("--save-every",type=int,default=25000)
    p.add_argument("--eval-episodes",type=int,default=40)
    p.add_argument("--test-episodes",type=int,default=200)
    p.add_argument("--transfer-episodes",type=int,default=100)
    p.add_argument("--worlds",type=int,default=16,help="Parallel simulated worlds / inference batch; 1 runs v0.7 trainer")
    p.add_argument("--output",type=Path)
    p.add_argument("--resume-suite",type=Path)
    p.add_argument("--report-only",type=Path)
    a=p.parse_args()
    if sum(x is not None for x in (a.output,a.resume_suite,a.report_only))>1:
        p.error("Choose only one of --output/--resume-suite/--report-only")
    root=a.output or a.resume_suite or a.report_only or (Path("runs")/f"cnn_{datetime.now():%Y%m%d_%H%M%S}")
    with prevent_sleep():
        run_suite(root,steps=a.steps,repeats=a.repeats,seed=a.seed,
                  engine=a.engine,device=a.device,eval_every=a.eval_every,
                  save_every=a.save_every,eval_episodes=a.eval_episodes,
                  test_episodes=a.test_episodes,transfer_episodes=a.transfer_episodes,
                  resume=bool(a.resume_suite),report_only=bool(a.report_only),worlds=a.worlds)

if __name__=="__main__":main()
