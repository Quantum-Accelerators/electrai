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
        normalize,
    ):
        super().__init__()

        self.in_conv = ResBlock3D(in_channels, n_channels, kernel_size)

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.normalize = normalize

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
        out = self.in_conv(x)

        for enc, down in zip(self.enc_blocks, self.downs, strict=False):
            out = enc(out)
            skips.append(out)
            out = down(out)
        out = self.mid(out)

        for up, dec in zip(self.ups, self.dec_blocks, strict=False):
            out = up(out)
            out = torch.cat([out, skips.pop()], dim=1)
            out = dec(out)
        out = self.out_conv(out)
        if self.normalize:
            out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
            out = out * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]
        return out


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
