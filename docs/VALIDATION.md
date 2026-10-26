# Validation record

This is a sanitized summary of the local Apple Silicon validation. Machine paths,
raw installation logs, temporary profiles, archives and checkpoints are not part
of the public repository. The original detailed local reports remain on the
maintainer's machine; ignoring them does not delete them.

## Local macOS validation

- Hardware: Apple M5, native arm64; macOS 27.0.1.
- Python 3.13.15, PyTorch 2.14.1, NumPy 2.5.4, Pygame 2.6.1,
  Gymnasium 1.4.0; Rust/Cargo 1.99.1.
- Before publication changes: 98 pytest tests passed, no skips; 2 Rust tests passed.
- Publication candidate: 102 pytest tests passed locally with Metal access,
  including four publication-hygiene regressions; 2 Rust tests passed again.
- All five model architectures passed CPU/MPS initial-Q comparison and one Adam
  update with finite gradients (`rtol=1e-3`, `atol=1e-4` for Q comparison).
- CNN and Motion-CNN passed four 64-transition batch smoke runs per device
  (two architectures × Python/Rust), seven Adam updates per run, replay,
  evaluation, and checkpoint save/load.
- Native library was verified as an arm64 Mach-O dylib.
- Random and Motion-CNN/MPS viewers rendered actual macOS windows for two frames
  each and terminated successfully; regression tests also render SDL dummy windows.

## Deterministic engine parity

All 7 maps × 3 ghost counts × 2 ghost movement cadences × 30 episodes:
**501,488 identical simulation steps**, including observations, rewards, endings
and diagnostic counters. These are simulator checks, not learning experiments.

## Simulator benchmark

One short run, `large`, 16 worlds, 160,000 transitions, same action tape and pellet
checksum: Python 128,564 steps/s, Rust 292,711 steps/s, approximately **2.28×**.
This measures simulation and binding only, not end-to-end neural training.

## Short pipeline profiles

Single-world profiles include setup, first backend calls and synchronization.
Each used 64 transitions and seven Adam updates. They were run sequentially.

| Model | CPU/Python | CPU/Rust | MPS/Python | MPS/Rust |
|---|---:|---:|---:|---:|
| CNN | 49.07 | 146.28 | 41.15 | 313.29 |
| Motion-CNN | 146.98 | 130.96 | 307.28 | 136.04 |

Values are profiled environment steps/s. Their variability and cold-start effects
make these unsuitable for ranking architectures, engines or devices. No sustained
training speedup or learning improvement is claimed. Performance decisions need
warmup, repetitions, balanced order and a longer measurement window.

## CI hardware and limits

[GitHub documents](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
`macos-15` as arm64 standard hosted hardware. CI asserts `uname -m == arm64` and
prints toolchain/device information. It is not an M5 benchmark. Virtualized hosts
may not expose Metal; CI explicitly reports MPS skips and still runs full CPU
coverage, compiled Rust parity and bounded CPU smoke. Local MPS coverage above
must not be inferred from a CPU-only CI run.

The workflow pins Action commits, Rust and Python dependency versions. The macOS
image and the Python 3.13 patch may receive hosted-image updates; this is not a
bit-for-bit OS build. No binary artifacts or training weights are uploaded.
