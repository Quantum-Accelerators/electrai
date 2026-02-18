from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, k, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_block = nn.Sequential(
            nn.Conv3d(cin, cout, k, padding=k // 2, padding_mode="circular"),
            nn.InstanceNorm3d(cout),
            nn.PReLU(),
            nn.Conv3d(cout, cout, k, padding=k // 2, padding_mode="circular"),
            nn.InstanceNorm3d(cout),
        )
        self.act = nn.PReLU()

        if cin != cout:
            self.skip = nn.Conv3d(cin, cout, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return self.act(
                checkpoint(self.conv_block, x, use_reentrant=False) + self.skip(x)
            )
        else:
            return self.act(self.conv_block(x) + self.skip(x))


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,
        depth,
        n_residual_blocks,
        kernel_size,
        use_checkpoint=True,
    ):
        super().__init__()
        self.in_conv = ResBlock3D(
            in_channels, n_channels, kernel_size, use_checkpoint=use_checkpoint
        )

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = n_channels
        for _ in range(depth):
            self.enc_blocks.append(
                nn.Sequential(
                    *[
                        ResBlock3D(ch, ch, kernel_size, use_checkpoint=use_checkpoint)
                        for _ in range(n_residual_blocks)
                    ]
                )
            )
            self.downs.append(downsample(ch, 2 * ch))
            ch *= 2

        # -------- Bottleneck --------
        self.mid = nn.Sequential(
            *[
                ResBlock3D(ch, ch, kernel_size, use_checkpoint=use_checkpoint)
                for _ in range(2 * n_residual_blocks)
            ]
        )

        # -------- Decoder --------
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for _ in range(depth):
            self.ups.append(PeriodicUpsampleConv3d(ch, ch // 2))
            ch //= 2
            self.dec_blocks.append(
                nn.Sequential(
                    *[
                        ResBlock3D(
                            2 * ch, ch, kernel_size, use_checkpoint=use_checkpoint
                        )
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
        out = out / (torch.sum(out, axis=(-3, -2, -1))[..., None, None, None] + 1e-8)
        return out * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]


class PeriodicUpsampleConv3d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Conv3d(cin, cout, 3, padding=1, padding_mode="circular")
        self.norm = nn.InstanceNorm3d(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
        x = self.up(x)
        x = x[..., 2:-2, 2:-2, 2:-2]
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


def downsample(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=2, padding=1, padding_mode="circular"),
        nn.InstanceNorm3d(cout),
        nn.PReLU(),
    )
