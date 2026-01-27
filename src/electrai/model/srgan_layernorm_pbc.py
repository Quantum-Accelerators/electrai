"""
adapted from https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, in_features, K=3, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_block = nn.Sequential(
            nn.Conv3d(
                in_features,
                in_features,
                kernel_size=K,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.InstanceNorm3d(in_features),
            nn.PReLU(),
            nn.Conv3d(
                in_features,
                in_features,
                kernel_size=K,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.InstanceNorm3d(in_features),
        )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return x + checkpoint(self.conv_block, x, use_reentrant=False)
        else:
            return x + self.conv_block(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        assert in_channels % (upscale_factor**3) == 0
        super().__init__()
        self.u = upscale_factor
        self.Cin = in_channels

    def forward(self, X):
        """
        assume X has shape of (Nbatch, Cin*u**3, H, W, D)
        """
        assert X.shape[1] == self.Cin
        u = self.u
        Cout = self.Cin // u**3
        out = X.reshape(-1, Cout, u, u, u, *X.shape[-3:])
        out = out.permute((0, 1, 5, 2, 6, 3, 7, 4))
        return out.reshape(-1, Cout, u * X.shape[-3], u * X.shape[-2], u * X.shape[-1])


class GeneratorResNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_residual_blocks=16,
        n_upscale_layers=2,
        n_channels=64,
        kernel_size1=5,
        kernel_size2=3,
        normalize=True,
        use_checkpoint=True,
    ):
        """
        This net upscales each axis by 2**n_upscale_layers
        n_channels = channel size in most of layers
        kernel_size1 = kernel size in the first and last layers
        kernel_size2 = kernel size in Res blocks
        use_checkpoint = enable gradient checkpointing to save memory
        """
        super().__init__()
        self.n_upscale_layers = n_upscale_layers
        self.normalize = normalize
        self.use_checkpoint = use_checkpoint

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                n_channels,
                kernel_size=kernel_size1,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = [
            ResidualBlock(n_channels, K=kernel_size2, use_checkpoint=use_checkpoint)
            for _ in range(n_residual_blocks)
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                n_channels,
                n_channels,
                kernel_size=kernel_size2,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.InstanceNorm3d(n_channels),
        )

        # Upsampling layers
        upsampling = []
        for _out_features in range(n_upscale_layers):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv3d(
                    n_channels,
                    n_channels * 8,
                    kernel_size=kernel_size2,
                    stride=1,
                    padding="same",
                    padding_mode="circular",
                ),
                nn.InstanceNorm3d(n_channels * 8),
                PixelShuffle3d(n_channels * 8, upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                n_channels,
                out_channels,
                kernel_size=kernel_size1,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self._forward(x)
        return [self._forward(xi.unsqueeze(0)).squeeze(0) for xi in x]

    def _forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        if self.normalize:
            upscale_factor = 8 ** (self.n_upscale_layers)
            out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
            out = (
                out
                * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]
                * upscale_factor
            )
        return out
