from __future__ import annotations

import torch


class MAE(torch.nn.Module):
    def __init__(self, elf: bool):
        super().__init__()
        self.elf = elf
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        if isinstance(output, torch.Tensor):
            return self._forward(output, target)

        losses = []
        for out, tar in zip(output, target, strict=False):
            losses.append(self._forward(out.unsqueeze(0), tar.unsqueeze(0)))
        return torch.stack(losses).mean()

    def _forward(self, output, target):
        mae = self.mae(output, target)
        if self.elf:
            return torch.mean(mae)
        nelec = torch.sum(target, axis=(-3, -2, -1))
        mae = mae / nelec[..., None, None, None]
        return torch.sum(mae)
