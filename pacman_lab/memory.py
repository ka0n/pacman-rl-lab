# SPDX-License-Identifier: GPL-3.0-or-later
"""Short-term memory from observable successive game frames (no hidden game state).

Stacking observations, rather than an RNN, keeps DQN and experience replay simple.
The oldest frame is first; the most recent frame is last. An initial observation
is repeated on reset, so the net does NOT receive an invented initial velocity.
"""
from __future__ import annotations

from collections import deque
import numpy as np


class FrameHistory:
    def __init__(self, frames: int = 1):
        if frames < 1 or frames > 8:
            raise ValueError("frames must be from 1 to 8")
        self.frames = frames
        self.history: deque[np.ndarray] = deque(maxlen=frames)

    def reset(self, observation: np.ndarray) -> np.ndarray:
        self.history.clear()
        for _ in range(self.frames):
            self.history.append(observation.copy())
        return self.observe()

    def step(self, observation: np.ndarray) -> np.ndarray:
        if len(self.history) != self.frames:
            raise RuntimeError("Call reset before step")
        self.history.append(observation.copy())
        return self.observe()

    def observe(self) -> np.ndarray:
        if len(self.history) != self.frames:
            raise RuntimeError("FrameHistory is not initialized")
        return np.concatenate(tuple(self.history), axis=0)


def infer_frames(input_shape: tuple[int, ...]) -> int:
    if len(input_shape) != 3 or input_shape[0] % 4 or input_shape[0] < 4:
        raise ValueError(f"Unexpected observation shape: {input_shape}")
    return input_shape[0] // 4


def erase_ghost_motion(history: np.ndarray, frames: int) -> np.ndarray:
    """Ablation: replace past ghost planes with the current ghost positions.

    Preserves all current fields and player/pellet history. This synthetic
    observation is not necessarily a physically valid path; Q sensitivity is
    a diagnostic, not proof of prediction or anticipation.
    """
    if frames == 1:
        return history.copy()
    changed = history.copy()
    latest = history[(frames - 1) * 4 + 3]
    for index in range(frames - 1):
        changed[index * 4 + 3] = latest
    return changed
