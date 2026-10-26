#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This setup targets native Apple Silicon macOS (arm64).' >&2
  exit 1
fi
brew_bin="$(command -v brew || true)"
if [[ -z "$brew_bin" && -x /opt/homebrew/bin/brew ]]; then brew_bin=/opt/homebrew/bin/brew; fi
if [[ -z "$brew_bin" ]]; then echo 'Install Homebrew first: https://brew.sh' >&2; exit 1; fi
xcrun --find clang >/dev/null || { echo 'Install Command Line Tools: xcode-select --install' >&2; exit 1; }
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
brew_root="$($brew_bin --prefix)"
if [[ ! -x "$brew_root/bin/python3.13" ]]; then "$brew_bin" install python@3.13; fi
if [[ ! -x "$brew_root/bin/cargo" ]]; then "$brew_bin" install rust; fi
export PATH="$brew_root/bin:$PATH"
if [[ ! -d .venv ]]; then "$brew_root/bin/python3.13" -m venv .venv; fi
.venv/bin/python -c 'import platform, sys; assert platform.machine() == "arm64" and (3,11) <= sys.version_info[:2] < (3,14), "Use an arm64 Python 3.11–3.13 venv"'
.venv/bin/python -m pip install -e '.[dev,ml]'
cargo test --release --manifest-path native/pacman_native/Cargo.toml
cargo build --release --manifest-path native/pacman_native/Cargo.toml
.venv/bin/python -m pytest -q
.venv/bin/python -m pacman_lab.macos doctor
echo 'Ready. Run scripts/lab.sh smoke for the bounded MPS/Rust smoke test.'
