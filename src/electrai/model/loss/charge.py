from __future__ import annotations

import torch


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        mae = self.mae(output, target)
        nelec = torch.sum(target, axis=(-3, -2, -1))
        # Use .view() instead of [..., None, None, None] to avoid creating slice operations
        mae = mae / nelec.view(-1, 1, 1, 1)
        return torch.sum(mae)
