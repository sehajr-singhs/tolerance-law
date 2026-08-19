"""Behavior-cloning MLP policy for the insertion task.

State-in, action-out: obs (8-dim) -> action (2-dim). Standard MLP with
ReLU activations; width and depth are the capacity axis of the Tolerance Law.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BCMLP(nn.Module):
    def __init__(self, obs_dim: int = 8, act_dim: int = 2, width: int = 64,
                 depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
