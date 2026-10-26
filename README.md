# Pac-Man RL Lab

[![macOS CI](https://github.com/ka0n/pacman-rl-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ka0n/pacman-rl-lab/actions/workflows/ci.yml)
[![License: GPL v3 or later](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)

An independent, Pac-Man-inspired reinforcement-learning research environment for
**macOS Apple Silicon**, with PyTorch **MPS** acceleration and a deterministic
native **Rust** simulator. It studies how observation history, network structure
and training choices affect Double-DQN agents on a small semantic gridworld.

This is not an arcade emulator, a reproduction of the commercial game, or a claim
that general Pac-Man is solved. Agents perform well on the familiar training map;
existing evaluations have **not demonstrated successful generalization to the two
unseen maze layouts**. The cause of the Motion-CNN disadvantage remains open.

## Research results

The controlled v0.9.4 comparison used three training seeds and 1,000 common
held-out evaluation episodes per model on the familiar 11×13 map:

| Training seed | CNN win rate | Motion-CNN win rate | Motion − CNN |
|---|---:|---:|---:|
| 930042 | 89.3% | 75.4% | −13.9 pp |
| 930043 | 90.1% | 76.2% | −13.9 pp |
| 930044 | 89.2% | 77.1% | −12.1 pp |
| **Mean** | **89.53%** | **76.23%** | **−13.30 pp** |

Mean pellet collection was 97.1969% and 93.5177%, respectively. The disadvantage
appeared in all three seeds. Tests have not demonstrated an implementation error
in frame alignment, episode resets, replay or Double-DQN. Extra information can
still change optimization and state visitation; it does not guarantee better
finite-budget training. Three seeds do not establish a general architectural law.

See the [preserved results](docs/EXPERIMENT_RESULTS.md),
[original summary](docs/summary_620000.json), and detailed
[motion audit](docs/MACOS_MOTION_AUDIT.md). Original trained weights and run
folders are **not included**. These historical results were not retrained on M5.

## Architecture

Python implements the reference environment, observation history, replay,
training, evaluation and Pygame viewers. PyTorch runs inference and gradients on
CPU or MPS. A standard-library-only Rust `cdylib`, accessed through a `ctypes` C
ABI, provides single- and multi-world simulation with CPython-compatible seeded
randomness. There is no PyO3 dependency.

Observations are binary semantic planes: walls, remaining pellets, player and
current ghosts. They are not RGB screenshots. The batched trainer keeps each
world's history isolated and stores complete state/next-state transitions.

| Model | Input | Parameters on 11×13 | Learning |
|---|---|---:|---|
| `mlp_wide` | Current four planes | 688,388 | Double DQN |
| `cnn_single` | Current four planes | 4,976,708 | Double DQN |
| `cnn_history4` | Four stacked frames, oldest first | 4,983,620 | Double DQN |
| `cnn_gru` | CNN encoder and GRU temporal state | 5,371,460 | Recurrent Double DQN |
| `cnn_motion` | Current four planes + previous ghost plane | 4,977,284 | Double DQN |

Motion history repeats the initial ghost positions on reset. For a transition,
`next_state[4] == state[3]`. Matched-scratch initialization copies all shared CNN
weights and zeros only the extra channel, giving the same initial Q-function.
The extra channel receives gradients immediately; it is not frozen.

## Install on macOS

Requirements: native Apple Silicon macOS, Homebrew, Xcode Command Line Tools,
and Python 3.11–3.13. The validated setup uses Homebrew Python 3.13 and Apple M5.
Python 3.14 is outside the supported range.

```sh
git clone https://github.com/ka0n/pacman-rl-lab.git
cd pacman-rl-lab
# Install Command Line Tools if absent: xcode-select --install
./scripts/setup-macos.sh
./scripts/lab.sh doctor
```

The setup script installs missing Homebrew Python/Rust tools, creates `.venv`,
installs `.[dev,ml]`, builds/tests Rust and runs pytest. It starts no experiment.
For the exact validated Python dependency snapshot, use
`requirements/requirements-macos-py313.txt` before installing the project.

Manual equivalent:

```sh
brew install python@3.13 rust
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements-macos-py313.txt
python -m pip install --no-build-isolation --no-deps -e .
cargo test --release --locked --manifest-path native/pacman_native/Cargo.toml
cargo build --release --locked --manifest-path native/pacman_native/Cargo.toml
python -m pytest -q
```

MPS requires Metal GPU access. A sandbox or hosted VM may report it unavailable.
`--device auto` then selects CPU; explicit `--device mps` fails with a clear
message rather than pretending that GPU coverage occurred. CUDA remains an
optional backend in shared code; Windows launchers are no longer maintained.

## Play and inspect an agent

```sh
./scripts/lab.sh play --mode manual --maze large --ghosts 2
./scripts/lab.sh play --mode random --max-frames 120
./scripts/lab.sh watch --checkpoint /path/to/best.pt --maze large --device mps
```

Manual controls: arrow keys/WASD, Space to pause, R to reset, Escape to quit.
The CNN viewer adds S for one step and N for the next episode. Both viewers
support `--max-frames` for bounded GUI checks. CNN/Motion/GRU checkpoints use
`watch`; `play --mode dqn` is the legacy small-MLP viewer.

## Test, smoke and benchmark

```sh
./scripts/lab.sh test
./scripts/lab.sh verify
./scripts/lab.sh smoke --device cpu --engine both
./scripts/lab.sh smoke --device mps --engine both
./scripts/lab.sh bench --steps 160000
./scripts/lab.sh profile --architecture cnn_motion --device mps --engine rust
```

Smoke tests use fresh random models: five one-update gradient probes, then CNN
and Motion-CNN with 64 transitions per engine, replay, seven Adam updates,
evaluation and checkpoint roundtrip. The smoke command caps a run at 256
transitions and creates a new temporary output directory by default.
These are functional checks, not model-quality measurements.

Local validation before publication: **98 pytest tests, 2 Rust tests and 501,488
identical Python/Rust steps**, plus CPU/MPS and real macOS viewer smoke tests.
Publication hygiene tests are added separately. See [validation](docs/VALIDATION.md)
and the live CI badge for current coverage.

On M5, one 160,000-step simulator benchmark measured about **128,564 Python vs.
292,711 Rust steps/s (2.28×)** with identical action tape and checksum. This
measures simulation plus binding, not neural training. Very short pipeline
profiles varied substantially due to setup, kernel warmup and synchronization;
no sustained training-speed ranking is claimed. Always distinguish CPU/Python,
CPU/Rust, MPS/Python and MPS/Rust. See [profiling](PROFILE_README.md).

## Training and evaluation

A deliberately small trainer example, with real weight updates:

```sh
.venv/bin/python -m pacman_lab.cnn_train \
  --architecture cnn_single --maze large --engine rust --device mps \
  --worlds 4 --steps 64 --warmup 16 --batch-size 4 --replay-capacity 128 \
  --eval-every 64 --save-every 64 --eval-episodes 2 --seed 710093 \
  --output runs/example-cnn
```

This checks the basic trainer; it does not reproduce the historical one-million-
transition study or its study-specific −20 death penalty. The controlled study
protocol lives in `cnn_motion_study.py` and
[V094_MOTION_STUDY_README.md](V094_MOTION_STUDY_README.md). Its reuse path requires
a previously trained reference checkpoint that is not distributed here.
`./scripts/lab.sh study-plan` only plans; no launcher defaults to a long study.

For a compatible checkpoint, evaluate on a separately preregistered held-out
window (900000 below is an example seed base, not a new measured result):

```sh
.venv/bin/python -m pacman_lab.cnn_v09 evaluate \
  --checkpoint runs/example-cnn/best.pt --maze large --engine rust --device mps \
  --episodes 1000 --test-seed 900000 --output runs/example-audit.json
```

Use at least 1,000 common final episode seeds per model and multiple training
seeds for model comparisons; report wins and pellets. Do not tune repeatedly on
that final set. The historical 620000–620999 window has already been used.
Repeated 100-episode checkpoint-selection validation is not evidence of a model
improvement. Small smoke evaluations are not suitable for selecting a policy.

Seeds, configuration and selected model/target/optimizer states are recorded.
A resumed run is a **warm restart**: replay, environment and RNG state are not
restored. Bit-exact training across CPU, MPS and CUDA is not claimed. Existing
checkpoints/results are never to be overwritten. Use fresh output directories.
Unseen layouts must have dimensions compatible with the checkpoint's dense head.

## CI and development

[GitHub Actions](https://github.com/ka0n/pacman-rl-lab/actions) builds the Rust
engine, runs Rust/pytest tests, checks parity and executes bounded smoke tests on
`macos-15`. CI asserts arm64 and logs the actual environment; hosted hardware is
not an M5 performance benchmark. If Metal is unavailable, CI reports MPS skips
and runs CPU tests. Action commits, Rust and Python package versions are pinned.
See [contributing](CONTRIBUTING.md), [agent guidance](AGENTS.md) and
[third-party licensing](docs/THIRD_PARTY_LICENSES.md).

Older `V*_README.md` files document historical protocols and may contain obsolete
Windows commands. Use this README and `scripts/` for the current installation.
Raw local logs/profiles, environments, native binaries and weights are excluded
from version control. Sanitized results and source are published.

## License and trademarks

Original project code is **GPL-3.0-or-later**; see the unchanged full GPLv3 text in
[LICENSE](LICENSE) and the licensing/attribution scope in [NOTICE](NOTICE).
Contributors retain their copyrights. Dependencies and upstream portions retain
their own licenses; the project does not relicense third-party libraries.

Pac-Man is a trademark of Bandai Namco. This is an independent research project,
not affiliated with, endorsed by, or sponsored by Bandai Namco. No original
commercial artwork, sounds, ROMs, fonts or logos are included. The software
license grants no trademark rights.
