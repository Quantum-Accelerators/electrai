from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
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


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, k, use_conditioning: bool = False, cond_dim: int = 0):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, k, padding=k // 2, padding_mode="circular")
        self.norm1 = nn.InstanceNorm3d(cout)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv3d(cout, cout, k, padding=k // 2, padding_mode="circular")
        self.norm2 = nn.InstanceNorm3d(cout)
        self.use_conditioning = use_conditioning

        if self.use_conditioning:
            self.film1 = FiLM(cond_dim, cout)
            self.film2 = FiLM(cond_dim, cout)

        if cin != cout:
            self.skip = nn.Conv3d(cin, cout, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_conditioning and cond is None:
            raise ValueError("Conditioning is enabled but cond is None.")
        h = self.conv1(x)
        h = self.norm1(h)
        if self.use_conditioning:
            h = self.film1(h, cond)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        if self.use_conditioning:
            h = self.film2(h, cond)

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
        normalize: bool = True,
        use_conditioning: bool = False,
        cond_dim: int = 6,
    ):
        super().__init__()
        self.normalize = normalize
        self.use_conditioning = use_conditioning
        self.cond_dim = cond_dim

        self.in_conv = ResBlock3D(
            in_channels,
            n_channels,
            kernel_size,
            use_conditioning=self.use_conditioning,
            cond_dim=self.cond_dim,
        )

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = n_channels
        for _ in range(depth):
            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock3D(
                            ch,
                            ch,
                            kernel_size,
                            use_conditioning=use_conditioning,
                            cond_dim=cond_dim,
                        )
                        for _ in range(n_residual_blocks)
                    ]
                )
            )
            self.downs.append(downsample(ch, 2 * ch))
            ch *= 2

        # -------- Bottleneck --------
        self.mid = nn.ModuleList(
            [
                ResBlock3D(
                    ch,
                    ch,
                    kernel_size,
                    use_conditioning=use_conditioning,
                    cond_dim=cond_dim,
                )
                for _ in range(2 * n_residual_blocks)
            ]
        )

        # -------- Decoder --------
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for _ in range(depth):
            self.ups.append(upsample(ch, ch // 2))
            ch //= 2
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock3D(
                            2 * ch,
                            ch,
                            kernel_size,
                            use_conditioning=use_conditioning,
                            cond_dim=cond_dim,
                        )
                        for _ in range(n_residual_blocks)
                    ]
                )
            )

        # -------- Output --------
        self.out_conv = nn.Conv3d(n_channels, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_conditioning and cond is None:
            raise ValueError("Model expects cond but got None.")
        if cond is not None and cond.ndim != 2:
            raise ValueError(
                f"cond must be (B,cond_dim). Got shape {tuple(cond.shape)}"
            )

        skips = []
        out = self.in_conv(x, cond)

        for blocks, down in zip(self.enc_blocks, self.downs, strict=False):
            for blk in blocks:
                out = blk(out, cond)
            skips.append(out)
            out = down(out)

        for blk in self.mid:
            out = blk(out, cond)

        for up, blocks in zip(self.ups, self.dec_blocks, strict=False):
            out = up(out)
            out = torch.cat([out, skips.pop()], dim=1)
            for blk in blocks:
                out = blk(out, cond)

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
