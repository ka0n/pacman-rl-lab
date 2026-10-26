#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then echo 'Run scripts/setup-macos.sh first.' >&2; exit 1; fi
export PATH="/opt/homebrew/bin:$PATH"
command_name="${1:-doctor}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
  doctor) exec .venv/bin/python -m pacman_lab.macos doctor "$@" ;;
  test) exec .venv/bin/python -m pytest -q "$@" ;;
  smoke) exec .venv/bin/python -m pacman_lab.macos smoke "$@" ;;
  play) exec .venv/bin/python -m pacman_lab.play "$@" ;;
  watch) exec .venv/bin/python -m pacman_lab.cnn_inspect watch "$@" ;;
  verify) exec .venv/bin/python -m pacman_lab.native_bench verify --episodes 4 "$@" ;;
  bench) exec .venv/bin/python -m pacman_lab.native_bench bench --steps 16000 --worlds 16 "$@" ;;
  profile) exec .venv/bin/python -m pacman_lab.profile_pipeline --steps 64 --warmup 16 --batch-size 4 --replay-capacity 128 "$@" ;;
  study-plan) exec .venv/bin/python -m pacman_lab.cnn_motion_study plan "$@" ;;
  *) echo 'Usage: scripts/lab.sh {doctor|test|smoke|play|watch|verify|bench|profile|study-plan} [options]' >&2; exit 2 ;;
esac
