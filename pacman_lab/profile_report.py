# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize v0.7 pipeline profiles and prepare a small upload ZIP."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def create_report(folder: Path) -> Path:
    profiles=[]
    for f in sorted(folder.glob('profile_*.json')):
        data=json.loads(f.read_text(encoding='utf-8'))
        if data.get('benchmark') == 'pacman_rl_lab_v0.7_pipeline_profile_v1':
            profiles.append((f,data))
    if not profiles:
        raise ValueError(f'No profile_*.json benchmark outputs in {folder}')
    by_arch=defaultdict(dict)
    for filename,p in profiles:
        spec=p['settings']
        by_arch[(spec['architecture'], spec['device'])][spec['engine']] = p
    lines=[
        '# Pac-Man RL Lab v0.7 — Pipeline Benchmark',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}',
        '',
        'These are measured *profiling* runs, not learning-quality tests.',
        'GPU synchronization in each timed section has overhead. '
        'CPU/Python, CPU/Rust and MPS results must be distinguished.',
        '',
        '| Architecture | Device | Engine | Steps/s | Train updates | Largest measured component | GPU peak allocated |',
        '|---|---|---|---:|---:|---|---:|',
    ]
    for _,p in profiles:
        s=p['settings']
        d=p['timings_percent']
        bottleneck=max((k for k in d if k not in ('setup', 'other_python_and_profiling_overhead')),
                       key=lambda k:d[k])
        peak=p.get('gpu_peak_allocated_mb')
        peak_label = f'{peak:,.1f} MB' if peak is not None else 'n/a'
        lines.append(f"| {s['architecture']} | {s['device']} | {s['engine']} | {p['steps_per_second']:,.1f} | "
                     f"{p['counts']['updates']} | {bottleneck} ({d[bottleneck]:.1f}%) | "
                     f"{peak_label} |")
    lines.extend(['', '## Python vs Rust, whole training pipeline', ''])
    for (arch, device),variants in sorted(by_arch.items()):
        if 'python' in variants and 'rust' in variants:
            py=variants['python']['steps_per_second']; rs=variants['rust']['steps_per_second']
            ratio=rs/py if py else 0
            lines.append(f'- **{arch}, {device}:** Python {py:,.1f} vs Rust {rs:,.1f} steps/s; '
                         f'Rust/Python = **{ratio:.2f}×** (these profiled runs only).')
        else:
            lines.append(f'- **{arch}, {device}:** both Python and Rust results needed for a speedup ratio.')
    lines.extend(['', '## Interpretation', '',
                  '- `game_step_and_reset` = game physics + ghost moves + episode reset + frame history.',
                  '- `policy_forward` = action inference, including device transfer and action result.',
                  '- `replay_write` / `replay_sample` = experience storage and sample selection.',
                  '- `loss_and_transfer` = batch construction, forward passes and Double DQN target computation.',
                  '- `backward` / `optimizer_step` = gradient computation and weight update.',
                  '- Fresh random models receive benchmark optimizer updates; weights are not saved. '
                  'Existing checkpoints and experiment results are untouched.',
                  '- The default constant epsilon of 0.20 makes this a representative but not identical sample of the long marathon.',
                  '- Synchronization per GPU section adds overhead compared with uninstrumented training.',
                  '', '## Recommendations', ''])
    for (arch, device), variants in sorted(by_arch.items()):
        for engine,p in sorted(variants.items()):
            t=p['timings_percent']; training=(t.get('loss_and_transfer',0)+t.get('backward',0)+t.get('optimizer_step',0))
            env=t.get('game_step_and_reset',0)
            policy=t.get('policy_forward',0)
            if training>max(env,policy): suggestion='Inspect minibatch construction, optimizer work and device kernels; Rust alone is unlikely to dominate.'
            elif policy>max(env,training): suggestion='Focus on batched policy inference across multiple environments.'
            elif env>max(policy,training): suggestion='Focus on Rust batch execution / CPU parallelism.'
            else: suggestion='No single clear bottleneck; inspect all timing categories.'
            lines.append(f'- **{arch}, {device}, {engine}:** {suggestion}')
    md='\n'.join(lines)+'\n'
    report=folder/'PROFILE_REPORT.md'
    report.write_text(md,encoding='utf-8')
    archive=folder/'UPLOAD_THIS_pipeline_profile.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        z.write(report,report.name)
        for name,_ in profiles:
            z.write(name,name.name)
    print(md)
    print(f'Upload package: {archive}')
    return archive


def main():
    p=argparse.ArgumentParser(description='Summarize per-model CNN training-pipeline profiles')
    p.add_argument('--folder',type=Path,required=True)
    a=p.parse_args()
    create_report(a.folder)


if __name__=='__main__':main()
