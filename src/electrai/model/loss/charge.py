from __future__ import annotations

import torch


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        nelec = torch.sum(target, axis=(-3, -2, -1))
        mae = mae / nelec[..., None, None, None]
        return torch.sum(mae)


class DensityWeightedNormMAE(torch.nn.Module):
    """NormMAE with density-dependent weighting to penalize errors at high-density voxels.

    Weight function: w(rho) = 1 + alpha * (rho / rho_max)^power
    where rho_max is the per-sample maximum density.
    """

    def __init__(self, alpha=1.0, power=1.0):
        super().__init__()
        self.alpha = alpha
        self.power = power
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
        rho_max = target.amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-8)
        weights = 1.0 + self.alpha * (target / rho_max) ** self.power
        weighted_mae = weights * mae
        nelec = torch.sum(target, dim=(-3, -2, -1))
        weighted_mae = weighted_mae / nelec[..., None, None, None]
        return torch.sum(weighted_mae)
