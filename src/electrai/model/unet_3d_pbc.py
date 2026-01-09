"""
3D U-Net architecture with periodic boundary conditions for materials science super-resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from electrai.model.srgan_layernorm_pbc import PixelShuffle3d


class ConvBlock3d(nn.Module):
    """Double convolution block with periodic boundary conditions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self.conv_block, x, use_reentrant=False)
        else:
            return self.conv_block(x)


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock followed by strided convolution for downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.conv_block = ConvBlock3d(
            in_channels, out_channels, kernel_size, use_checkpoint
        )
        # Use strided convolution for downsampling (MaxPool breaks periodic BC)
        # Manual circular padding needed for strided conv
        self.downsample = nn.Conv3d(
            out_channels, out_channels, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        features = self.conv_block(x)
        downsampled = self.downsample(features)
        return features, downsampled


class DecoderBlock(nn.Module):
    """Decoder block: Upsample, concatenate skip connection, then ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        # Conv to adjust channels after upsampling
        self.up_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            padding_mode="circular",
        )
        # ConvBlock after concatenation
        self.conv_block = ConvBlock3d(
            in_channels + skip_channels, out_channels, kernel_size, use_checkpoint
        )

    def forward(self, x, skip):
        # Trilinear upsampling (avoids checkerboard artifacts)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.up_conv(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class Bottleneck(nn.Module):
    """Bottleneck block at the deepest level of U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.conv_block = ConvBlock3d(
            in_channels, out_channels, kernel_size, use_checkpoint
        )

    def forward(self, x):
        return self.conv_block(x)


class GeneratorUNet(nn.Module):
    """3D U-Net generator with periodic boundary conditions for materials science."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_upscale_layers: int = 1,
        C: int = 64,
        depth: int = 4,
        K1: int = 5,
        K2: int = 3,
        channel_multiplier: tuple[int, ...] = (1, 2, 4, 8),
        _n_residual_blocks: int = 0,
        normalize: bool = True,
        use_checkpoint: bool = True,
    ):
        """
        3D U-Net with periodic boundary conditions.

        Args:
            in_channels: Input channels (typically 1 for charge density)
            out_channels: Output channels (typically 1)
            n_upscale_layers: Number of 2x upscaling for super-resolution
            C: Base channel count
            depth: Number of encoder/decoder levels
            K1: Kernel size for input/output convolutions
            K2: Kernel size for internal convolutions
            channel_multiplier: Tuple defining channel multiplier at each level
            _n_residual_blocks: Unused, kept for config compatibility
            normalize: Apply charge conservation normalization
            use_checkpoint: Enable gradient checkpointing
        """
        super().__init__()
        self.n_upscale_layers = n_upscale_layers
        self.normalize = normalize
        self.depth = depth

        # Ensure channel_multiplier matches depth
        if len(channel_multiplier) < depth:
            channel_multiplier = tuple(
                list(channel_multiplier)
                + [channel_multiplier[-1]] * (depth - len(channel_multiplier))
            )
        channel_multiplier = channel_multiplier[:depth]

        # Calculate channel counts at each level
        encoder_channels = [C * m for m in channel_multiplier]
        bottleneck_channels = encoder_channels[-1] * 2

        # Input convolution
        self.conv_in = nn.Sequential(
            nn.Conv3d(
                in_channels,
                C,
                kernel_size=K1,
                stride=1,
                padding="same",
                padding_mode="circular",
            ),
            nn.PReLU(),
        )

        # Encoder path
        self.encoders = nn.ModuleList()
        in_ch = C
        for i in range(depth):
            out_ch = encoder_channels[i]
            self.encoders.append(EncoderBlock(in_ch, out_ch, K2, use_checkpoint))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = Bottleneck(in_ch, bottleneck_channels, K2, use_checkpoint)

        # Decoder path
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_channels
        for i in range(depth - 1, -1, -1):
            skip_ch = encoder_channels[i]
            out_ch = encoder_channels[i]
            self.decoders.append(
                DecoderBlock(in_ch, skip_ch, out_ch, K2, use_checkpoint)
            )
            in_ch = out_ch

        # Super-resolution upsampling layers (reuse PixelShuffle3d pattern)
        upsampling = []
        for _ in range(n_upscale_layers):
            upsampling += [
                nn.Conv3d(
                    C,
                    C * 8,
                    kernel_size=K2,
                    stride=1,
                    padding="same",
                    padding_mode="circular",
                ),
                nn.InstanceNorm3d(C * 8),
                PixelShuffle3d(C * 8, upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv3d(
                C,
                out_channels,
                kernel_size=K1,
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
        original_input = x

        # Input convolution
        out = self.conv_in(x)

        # Encoder path - store skip connections
        skip_connections = []
        for encoder in self.encoders:
            skip, out = encoder(out)
            skip_connections.append(skip)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder path - use skip connections (reversed order)
        for decoder, skip in zip(
            self.decoders, reversed(skip_connections), strict=False
        ):
            out = decoder(out, skip)

        # Super-resolution upsampling
        out = self.upsampling(out)

        # Output convolution
        out = self.conv_out(out)

        # Charge conservation normalization
        if self.normalize:
            upscale_factor = 8 ** (self.n_upscale_layers)
            out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
            out = (
                out
                * torch.sum(original_input, axis=(-3, -2, -1))[..., None, None, None]
                * upscale_factor
            )

        return out
