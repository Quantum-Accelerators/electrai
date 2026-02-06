from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):  # vanilla
    """
    Feature-wise Linear Modulation (FiLM):
    Given cond (B, cond_dim), produce per-channel scale/shift applied to a feature map.
    """

    def __init__(self, cond_dim: int, n_channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * n_channels)
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: (B,C,D,H,W), cond: (B,cond_dim)
        gb = self.mlp(cond)  # (B,2C)
        gamma, beta = gb.chunk(2, dim=-1)  # (B,C), (B,C)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        # (1 + gamma) is nice: starts close to identity if gamma ~ 0
        return h * (1.0 + gamma) + beta
