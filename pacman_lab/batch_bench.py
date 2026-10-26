# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end *training* throughput comparison with short independent runs.

Includes initialization, replay, real optimizer updates, occasional evaluation,
checkpointing. The reported speed is total steps / wall time. Warm-up and
GPU synchronization differ from production; always repeat the benchmark.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from .cnn_train import run_training
from .cnn_batch import run_batched


def run_benchmark(*, output: Path, engine="rust", device="auto", steps=3000,
                  worlds=16, architecture="cnn_history4", eval_episodes=5):
    if steps < 32 or worlds < 2:
        raise ValueError("steps>=32 and worlds>=2 required")
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    kwargs=dict(architecture=architecture,steps=steps,seed=907,device_name=device,
                eval_every=steps,save_every=steps,eval_episodes=eval_episodes,
                replay_capacity=max(128,steps),batch_size=16,warmup=64,
                train_every=8,target_every=200,engine=engine,maze="large")
    results=[]
    for label,runner,extra in (("single",run_training,{}),
                                ("batched",run_batched,{"worlds":worlds})):
        start=time.perf_counter()
        runner(output=root/label,**kwargs,**extra)
        duration=time.perf_counter()-start
        row={"name":label,"steps":steps,"wall_seconds":round(duration,3),
             "throughput_steps_s":round(steps/duration,2)}
        with (root/label/"metrics.csv").open(newline="",encoding="utf-8") as f:
            last=list(csv.DictReader(f))[-1]
        row["win_percent"]=float(last["win_percent"])
        row["weight_updates_observed"] = bool(last.get("train_loss"))
        results.append(row)
        print(f"{label}: {row['throughput_steps_s']:.1f} steps/s (including test and save)",flush=True)
    if not all(r["weight_updates_observed"] for r in results):
        print("WARNING: At least one run did not update model weights; increase --steps for a meaningful training benchmark.",flush=True)
    ratio=results[1]["throughput_steps_s"]/results[0]["throughput_steps_s"]
    info={"architecture":architecture,"engine":engine,"device":device,
          "worlds":worlds,"steps":steps,"runs":results,"speedup":round(ratio,3),
          "caveat":"Short runs include compilation/warm-up, setup, validation and disk I/O. Results do not measure learning quality."}
    (root/"throughput.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    (root/"README_RESULT.txt").write_text(
        f"CNN training benchmark | {engine} | {architecture} | {worlds} worlds\n"
        f"Single: {results[0]['throughput_steps_s']:.1f} env steps/s\n"
        f"Batched: {results[1]['throughput_steps_s']:.1f} env steps/s\n"
        f"Speedup: {ratio:.2f}x\n",encoding="utf-8")
    return info


def main():
    p=argparse.ArgumentParser(description="Compare old 1-world trainer vs 16-world training on YOUR machine")
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--engine",choices=("python","rust"),default="rust")
    p.add_argument("--device",choices=("auto","cpu","cuda","mps"),default="auto")
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--worlds",type=int,default=16)
    p.add_argument("--architecture",choices=("mlp_wide","cnn_single","cnn_history4","cnn_gru"),default="cnn_history4")
    p.add_argument("--eval-episodes",type=int,default=5)
    a=p.parse_args()
    result=run_benchmark(output=a.output or Path("runs")/f"batch_benchmark_{datetime.now():%Y%m%d_%H%M%S}",
                         engine=a.engine,device=a.device,steps=a.steps,
                         worlds=a.worlds,architecture=a.architecture,eval_episodes=a.eval_episodes)
    print(f"SPEEDUP: {result['speedup']:.2f}x",flush=True)

if __name__=="__main__":main()
