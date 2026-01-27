from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, k, padding=k // 2, padding_mode="circular")
        self.norm1 = nn.InstanceNorm3d(cout)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv3d(cout, cout, k, padding=k // 2, padding_mode="circular")
        self.norm2 = nn.InstanceNorm3d(cout)

        if cin != cout:
            self.skip = nn.Conv3d(cin, cout, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,
        depth,
        n_residual_blocks,
        kernel_size,
    ):
        super().__init__()

        self.in_conv = ResBlock3D(in_channels, n_channels, kernel_size)

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = n_channels
        for _ in range(depth):
            self.enc_blocks.append(
                nn.Sequential(
                    *[ResBlock3D(ch, ch, kernel_size) for _ in range(n_residual_blocks)]
                )
            )
            self.downs.append(downsample(ch, 2 * ch))
            ch *= 2

        # -------- Bottleneck --------
        self.mid = nn.Sequential(
            *[ResBlock3D(ch, ch, kernel_size) for _ in range(2 * n_residual_blocks)]
        )

        # -------- Decoder --------
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for _ in range(depth):
            self.ups.append(upsample(ch, ch // 2))
            ch //= 2
            self.dec_blocks.append(
                nn.Sequential(
                    *[
                        ResBlock3D(2 * ch, ch, kernel_size)
                        for _ in range(n_residual_blocks)
                    ]
                )
            )

        # -------- Output --------
        self.out_conv = nn.Conv3d(n_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.in_conv(x)

        for enc, down in zip(self.enc_blocks, self.downs, strict=False):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.mid(x)

        for up, dec in zip(self.ups, self.dec_blocks, strict=False):
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = dec(x)

        return self.out_conv(x)


def downsample(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=2, padding=1, padding_mode="circular"),
        nn.InstanceNorm3d(cout),
        nn.PReLU(),
    )


def upsample(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2),
        nn.InstanceNorm3d(cout),
        nn.PReLU(),
    )
