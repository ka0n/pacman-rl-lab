# SPDX-License-Identifier: GPL-3.0-or-later
"""Large spatial and temporal Q-networks for the independent CNN marathon.

All four variants train with Double-DQN targets and the same game rewards.
Networks take semantic 4-plane observations, NOT RGB/Pac-Man arcade pixels.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Specification:
    architecture: str
    frames: int
    recurrent: bool


SPECS = {
    "mlp_wide": Specification("mlp_wide", 1, False),
    "cnn_single": Specification("cnn_single", 1, False),
    "cnn_motion": Specification("cnn_motion", 1, False),
    "cnn_history4": Specification("cnn_history4", 4, False),
    "cnn_gru": Specification("cnn_gru", 1, True),
}


def make_model(name: str, input_shape: tuple[int, int, int]):
    import math
    import torch.nn as nn
    if name not in SPECS:
        raise ValueError(f"Unknown CNN marathon architecture: {name}")
    channels, height, width = input_shape
    spec = SPECS[name]
    if height < 3 or width < 3 or channels != (5 if name == "cnn_motion" else 4 * spec.frames):
        raise ValueError(f"Shape {input_shape} incompatible with {name}")

    if name == "mlp_wide":
        return nn.Sequential(nn.Flatten(), nn.Linear(math.prod(input_shape), 512),
                             nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                             nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 4))

    class SpatialEncoder(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * height * width, 256), nn.ReLU())

        def forward(self, x):
            return self.conv(x)

    if not spec.recurrent:
        return nn.Sequential(SpatialEncoder(channels), nn.Linear(256, 256),
                             nn.ReLU(), nn.Linear(256, 4))

    class SpatialGRU(nn.Module):
        is_recurrent = True

        def __init__(self):
            super().__init__()
            self.encoder = SpatialEncoder(4)
            self.gru = nn.GRU(256, 256, batch_first=True)
            self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))

        def forward(self, observations, hidden=None):
            if observations.ndim != 5:
                raise ValueError("GRU expects [batch, time, 4, height, width]")
            b, t, c, h, w = observations.shape
            embeddings = self.encoder(observations.reshape(b * t, c, h, w))
            output, hidden = self.gru(embeddings.reshape(b, t, -1), hidden)
            return self.head(output), hidden

    return SpatialGRU()


def parameters_count(model) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
