"""Behavior-cloning CNN policy for the insertion task (vision modality).

Image-in, action-out: top-view RGB (3, 64, 64) -> action (2-dim).  A small
convnet whose width (channels) is the capacity axis of the Tolerance Law
for vision policies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BCCNN(nn.Module):
    def __init__(self, act_dim: int = 2, channels: int = 16, depth: int = 3,
                 img_size: int = 64):
        super().__init__()
        ch = channels
        layers: list[nn.Module] = []
        cin = 3
        k = 5
        s = img_size
        for i in range(depth):
            layers.append(nn.Conv2d(cin, ch, k, padding=k // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            cin = ch
            ch *= 2
            s //= 2
        self.features = nn.Sequential(*layers)
        flat = cin * s * s
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )
        self.act_dim = act_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(img))
