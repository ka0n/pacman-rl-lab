# SPDX-License-Identifier: GPL-3.0-or-later
"""Gymnasium adapter: one reset/step API, no graphics during training."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .core import PacmanGame


class PacmanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maze: str = "small", ghosts: int = 1, ghost_every: int = 1,
                 max_steps: int | None = None) -> None:
        super().__init__()
        self.game = PacmanGame(maze=maze, ghosts=ghosts,
                              ghost_every=ghost_every, max_steps=max_steps)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(4, self.game.height, self.game.width), dtype=np.uint8,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        return self.game.reset(seed=seed, ghosts=options.get("ghosts"),
                               ghost_every=options.get("ghost_every"))

    def step(self, action: int):
        result = self.game.step(int(action))
        return result.observation, result.reward, result.terminated, result.truncated, result.info

    def close(self):
        return None
