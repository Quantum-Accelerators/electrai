from __future__ import annotations

import torch


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        mae = self.mae(output, target)
        # Use keepdim=True to avoid view/reshape operations that create SliceBackward0 nodes
        nelec = torch.sum(target, dim=(-3, -2, -1), keepdim=True)
        mae = mae / nelec
        return torch.sum(mae)
