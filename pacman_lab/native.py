# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional std-only Rust engine via ctypes; existing Python engine remains default.

Build: cd native/pacman_native && cargo build --release
No dependence on Rust at Python package installation time.
"""
from __future__ import annotations

import ctypes as C
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .core import MAPS, StepResult, validate_batch_actions


class _Stats(C.Structure):
    _fields_ = [(key, C.c_uint32) for key in (
        "steps", "collected", "pellets_left", "initial_pellets",
        "wall_bumps", "stationary_steps", "stall_streak", "max_stall_streak",
        "reversals", "ghost_count")]
    _fields_ += [("won", C.c_uint8), ("done", C.c_uint8), ("_padding", C.c_uint8*2)]


def _load_library():
    filename = {"Windows": "pacman_native.dll", "Linux": "libpacman_native.so",
                "Darwin": "libpacman_native.dylib"}.get(platform.system())
    if not filename:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")
    root = Path(__file__).resolve().parents[1]
    paths = [root / "native" / "pacman_native" / "target" / "release" / filename,
             root / "native" / "pacman_native" / "target" / "debug" / filename]
    libpath = next((p for p in paths if p.exists()), None)
    if libpath is None:
        raise FileNotFoundError("Native Rust library not built. Run `cargo build --release` "
                                "from the native/pacman_native folder. "
                                f"Expected: {paths[0]}")
    lib = C.CDLL(str(libpath))
    lib.pacman_create.argtypes = [C.c_char_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint64]
    lib.pacman_create.restype = C.c_void_p
    lib.pacman_destroy.argtypes = [C.c_void_p]
    lib.pacman_destroy.restype = None
    lib.pacman_reset_one.argtypes = [C.c_void_p, C.c_uint32, C.c_uint64, C.c_uint32,
                                     C.c_uint32, C.c_void_p, C.POINTER(_Stats)]
    lib.pacman_reset_one.restype = C.c_int
    lib.pacman_observe.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(_Stats)]
    lib.pacman_observe.restype = C.c_int
    lib.pacman_step.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                C.c_void_p, C.c_void_p, C.POINTER(_Stats)]
    lib.pacman_step.restype = C.c_int
    return lib


@dataclass
class BatchOutcome:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    stats: list[dict]


class NativeBatch:
    """N independent games with one native step call; caller resets completed games.

    The return values are copies, safe to hold across further calls.
    Not thread-safe; do not step the same handle simultaneously.
    """
    def __init__(self, maze="small", count=16, ghosts=2, ghost_every=1,
                 max_steps=None, seed=42):
        if maze not in MAPS or count < 1 or ghosts not in (0, 1, 2) or ghost_every < 1:
            raise ValueError("Invalid maze / count / ghosts / ghost_every")
        self.maze = maze
        self.layout = MAPS[maze]
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        self.count = count
        self.ghost_count = ghosts
        self.ghost_every = ghost_every
        self.max_steps = max_steps or self.height * self.width * 8
        self.lib = _load_library()
        self._ptr = self.lib.pacman_create("\n".join(self.layout).encode(), count, ghosts,
                                           ghost_every, self.max_steps, seed)
        if not self._ptr:
            raise RuntimeError("Rust pacman_create failed (invalid maze or settings)")
        self._obs = np.zeros((count, 4, self.height, self.width), dtype=np.uint8)
        self._rewards = np.zeros(count, dtype=np.float64)
        self._terminated = np.zeros(count, dtype=np.uint8)
        self._truncated = np.zeros(count, dtype=np.uint8)
        self._stats = (_Stats * count)()
        self.observe()

    def close(self):
        if getattr(self, "_ptr", None):
            self.lib.pacman_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _info(s):
        return {"collected": int(s.collected), "pellets_left": int(s.pellets_left),
                "collected_fraction": s.collected/max(1, s.initial_pellets),
                "steps": int(s.steps), "wall_bumps": int(s.wall_bumps),
                "stationary_steps": int(s.stationary_steps),
                "stall_streak": int(s.stall_streak),
                "max_stall_streak": int(s.max_stall_streak),
                "reversals": int(s.reversals), "won": bool(s.won),
                "ghosts": int(s.ghost_count)}

    def observe(self):
        if not self._ptr:
            raise RuntimeError("NativeBatch already closed")
        rc = self.lib.pacman_observe(self._ptr, self._obs.ctypes.data,
                                     self._stats)
        if rc:
            raise RuntimeError(f"native observe failed ({rc})")
        return self._obs.copy()

    def reset_one(self, index, seed=None, ghosts=None, ghost_every=None):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if ghosts is not None:
            if ghosts not in (0,1,2): raise ValueError("ghosts must be 0, 1 or 2")
            if self.count != 1: raise ValueError("Changing ghost count requires a single-game batch")
        if ghost_every is not None:
            if ghost_every < 1: raise ValueError("ghost_every must be >= 1")
        if seed is not None and not 0 <= seed < 2**64-1:
            raise ValueError("Rust parity seeds must be integers between 0 and 2**64-2")
        frame = self._obs[index]
        info = _Stats()
        new_ghosts = self.ghost_count if ghosts is None else ghosts
        new_every = self.ghost_every if ghost_every is None else ghost_every
        rc = self.lib.pacman_reset_one(self._ptr, index, 2**64-1 if seed is None else seed,
                                       new_ghosts, new_every,
                                       frame.ctypes.data, C.byref(info))
        if rc:
            raise RuntimeError(f"native reset failed ({rc})")
        self.ghost_count, self.ghost_every = new_ghosts, new_every
        self._stats[index] = info
        return frame.copy(), self._info(info)

    def step(self, actions):
        arr = validate_batch_actions(actions, self.count)
        rc = self.lib.pacman_step(self._ptr, arr.ctypes.data, self._obs.ctypes.data,
                                  self._rewards.ctypes.data, self._terminated.ctypes.data,
                                  self._truncated.ctypes.data, self._stats)
        if rc: raise RuntimeError(f"native batch step failed ({rc}); reset completed episodes")
        return BatchOutcome(self._obs.copy(), self._rewards.copy(), self._terminated.copy(),
                            self._truncated.copy(), [self._info(i) for i in self._stats])


class NativePacmanGame:
    """Drop-in subset of PacmanGame used by training/evaluation."""
    def __init__(self, maze="small", ghosts=1, ghost_every=1, max_steps=None, seed=None):
        self.batch = NativeBatch(maze=maze, count=1, ghosts=ghosts,
                                 ghost_every=ghost_every, max_steps=max_steps,
                                 seed=seed if seed is not None else int.from_bytes(os.urandom(8),"little")%(2**64-1))
        self.height = self.batch.height
        self.width = self.batch.width
        self.maze_name = maze
        self.layout = self.batch.layout
        self.ghost_count = ghosts
        self.ghost_every = ghost_every
        self.max_steps = self.batch.max_steps
        self._last_obs = self.batch.observe()[0]
        self._last_info = self.batch._info(self.batch._stats[0])
        self.done = False
        self.won = False

    def reset(self, seed=None, ghosts=None, ghost_every=None):
        self._last_obs, self._last_info = self.batch.reset_one(0,seed,ghosts,ghost_every)
        self.ghost_count = self.batch.ghost_count
        self.ghost_every = self.batch.ghost_every
        self.done = self.won = False
        return self._last_obs.copy(), self._last_info.copy()

    def observe(self): return self._last_obs.copy()
    def info(self): return self._last_info.copy()

    @property
    def wall_bumps(self): return self._last_info["wall_bumps"]

    @property
    def stationary_steps(self): return self._last_info["stationary_steps"]

    @property
    def reversals(self): return self._last_info["reversals"]

    @property
    def steps(self): return self._last_info["steps"]

    @property
    def collected(self): return self._last_info["collected"]

    def step(self, action):
        if self.done: raise RuntimeError("episode finished; call reset() before another step")
        if int(action) not in (0,1,2,3): raise ValueError("action must be 0..3")
        result = self.batch.step([int(action)])
        self._last_obs = result.observations[0]
        self._last_info = result.stats[0]
        self.done = bool(result.terminated[0] or result.truncated[0])
        self.won = self._last_info["won"]
        return StepResult(self._last_obs.copy(), float(result.rewards[0]),
                          bool(result.terminated[0]),bool(result.truncated[0]),self._last_info.copy())

    def close(self): self.batch.close()
