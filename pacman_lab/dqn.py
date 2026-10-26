# SPDX-License-Identifier: GPL-3.0-or-later
"""Small MLP DQN, shared by trainer and viewer. Torch is imported lazily."""
from __future__ import annotations


def make_qnet(input_shape):
    import math
    import torch.nn as nn
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(math.prod(input_shape), 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )


def select_device(preference="auto"):
    import torch
    if preference != "auto":
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Check driver and cu126 PyTorch installation.")
        if preference == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS unavailable in this process. Check an arm64 PyTorch "
                               "installation and Metal GPU access; try running from macOS Terminal.")
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
