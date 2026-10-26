# SPDX-License-Identifier: GPL-3.0-or-later
"""The native parity tests execute whenever `cargo build --release` has run."""
import pytest
from pacman_lab.native import _load_library
from pacman_lab.native_bench import verify


def test_rust_python_episode_parity():
    try:
        _load_library()
    except FileNotFoundError:
        pytest.skip("Rust not compiled; run cargo build --release in native/pacman_native")
    verify(episodes=4)
