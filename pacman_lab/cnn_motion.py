# SPDX-License-Identifier: GPL-3.0-or-later
"""Observable ghost-motion input and safe 4->5-channel CNN weight transfer.

The extra plane is the previous observation's *ghost occupancy* (not an
oracle for latent ghost directions). This does not reveal hidden RNG or the
simulator's `ghost_directions`. All paths use the exact same online observation.
"""
from __future__ import annotations

import numpy as np

from .memory import FrameHistory


class GhostMotionHistory:
    """Five channels: [current walls, pellets, player, ghosts, previous ghosts]."""
    frames = 1

    def __init__(self):
        self._last = None
        self._previous_ghosts = None

    @staticmethod
    def _checked(observation):
        if observation.ndim != 3 or observation.shape[0] != 4:
            raise ValueError('GhostMotionHistory requires a [4,H,W] observation')
        return np.asarray(observation, dtype=np.uint8)

    def reset(self, observation):
        obs = self._checked(observation)
        self._last = obs.copy()
        self._previous_ghosts = obs[3].copy()
        return self.observe()

    def step(self, observation):
        if self._last is None:
            raise RuntimeError('Call reset before step')
        obs = self._checked(observation)
        self._previous_ghosts = self._last[3].copy()
        self._last = obs.copy()
        return self.observe()

    def observe(self):
        if self._last is None:
            raise RuntimeError('Call reset before observe')
        return np.concatenate((self._last, self._previous_ghosts[None]), axis=0)


def observer_for(architecture: str, frames: int):
    return GhostMotionHistory() if architecture == 'cnn_motion' else FrameHistory(frames)


def current_frame(state: np.ndarray, architecture: str):
    # v0.8 stack order is oldest -> newest, four planes each.
    return state[:4] if architecture == 'cnn_motion' else state[-4:]


def transfer_cnn_single_to_motion(source_state: dict, destination_model):
    """Copy a pretrained CNN exactly, zero-initializing only the extra plane.

    Before further training, Q(s,previous_ghosts) == Q_source(s) in eval mode.
    The optimizer must be new because the first convolution's shape changed.
    """
    import torch
    key = '0.conv.0.weight'
    destination = destination_model.state_dict()
    if key not in source_state or key not in destination:
        raise ValueError('Expected matching CNN sequential encoder keys')
    src = source_state[key]
    dst = destination[key]
    if tuple(src.shape) != (64,4,3,3) or tuple(dst.shape) != (64,5,3,3):
        raise ValueError(f'Unexpected first convolution: {tuple(src.shape)} -> {tuple(dst.shape)}')
    if source_state.keys() != destination.keys():
        raise ValueError('Source/destination state dictionaries differ')
    new_state = {}
    for name, value in destination.items():
        old = source_state[name]
        if name == key:
            clone = torch.zeros_like(value)
            clone[:, :4] = old.to(device=clone.device, dtype=clone.dtype)
            new_state[name] = clone
        else:
            if old.shape != value.shape:
                raise ValueError(f'Parameter mismatch at {name}')
            new_state[name] = old.to(device=value.device, dtype=value.dtype).clone()
    destination_model.load_state_dict(new_state)
