# multihead_readout.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _mass_normalize_like_input(pred: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):  # noqa ARG001
    """
    Match the total "mass" (integral over grid) of pred to x, per-sample.
    pred: (B, C, D, H, W)
    x:    (B, C_in, D, H, W)  (can be different C; we just use total over grid)
    """
    pred = pred / torch.sum(pred, axis=(-3, -2, -1))[..., None, None, None]
    return pred * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]


class MultiHeadReadout3D(nn.Module):
    def __init__(
        self, feat_channels: int, out_channels: int, num_heads: int, fid_offset: int = 1
    ):
        super().__init__()
        self.fid_offset = int(fid_offset)  # if your IDs start at 1, set fid_offset=1
        self.heads = nn.ModuleList(
            [
                nn.Conv3d(feat_channels, out_channels, kernel_size=1)
                for _ in range(num_heads)
            ]
        )

    def forward(
        self, h: torch.Tensor, x_in: torch.Tensor, fid: torch.Tensor
    ) -> torch.Tensor:
        """
        h:   (B, F, D, H, W)
        fid: (B,) integer functional ids (can be 1,2,...). We map to 0..num_heads-1 using fid_offset.
        returns: (B, out_channels, D, H, W)
        """
        if fid is None:
            raise ValueError("fid must be provided to compute only the relevant head.")

        fid = fid.to(device=h.device).long() - self.fid_offset  # map to 0-based
        if (fid < 0).any() or (fid >= len(self.heads)).any():
            raise ValueError(
                f"fid out of range after offset. Got min={fid.min().item()} max={fid.max().item()}"
            )

        # B = h.shape[0]
        # out = torch.empty(
        #     (B, self.heads[0].out_channels, *h.shape[-3:]),
        #     device=h.device,
        #     dtype=h.dtype,
        # )

        # compute only heads that appear in the batch
        # for u in torch.unique(fid):
        #     idx = (fid == u).nonzero(as_tuple=True)[0]  # indices for this head
        #     out[idx] = self.heads[int(u.item())](
        #         h[idx]
        #     )  # run only that head on that subset
        out = self.heads[fid.item()](h)
        return _mass_normalize_like_input(out, x_in)


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


class ResUNet3DBackbone(nn.Module):
    """
    Same as your ResUNet3D, but returns the shared feature map BEFORE out_conv.
    That feature map is what we feed into the two-head readout.

    Output shape: (B, n_channels, D, H, W)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,
        depth,
        n_residual_blocks,
        kernel_size,
        use_checkpoint=True,
        use_multihead=False,
    ):
        super().__init__()
        self.use_multihead = use_multihead

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
            blocks = [
                ResBlock3D(2 * ch, ch, kernel_size, use_checkpoint=use_checkpoint)
            ]
            blocks.extend(
                [
                    ResBlock3D(ch, ch, kernel_size, use_checkpoint=use_checkpoint)
                    for _ in range(n_residual_blocks - 1)
                ]
            )
            self.dec_blocks.append(nn.Sequential(*blocks))
        self.out_feat_channels = ch
        if not use_multihead:
            # -------- Output --------
            self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=1)

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
        if not self.use_multihead:
            out = self.out_conv(out)
            out = _mass_normalize_like_input(out, x)
            # out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
            # return out * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]
        return out


class ResUNet3D(nn.Module):
    """
    Full model: backbone -> two-head readout.
    Returns dict: {"low": ..., "high": ...}
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,
        depth,
        n_residual_blocks,
        kernel_size,
        use_checkpoint=True,
        head_hidden: int = 32,  # noqa ARG002
        use_multihead=False,
        num_heads=1,
    ):
        super().__init__()
        self.use_multihead = use_multihead
        self.backbone = ResUNet3DBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            n_channels=n_channels,
            depth=depth,
            n_residual_blocks=n_residual_blocks,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint,
            use_multihead=use_multihead,
        )
        if use_multihead:
            self.readout = MultiHeadReadout3D(
                feat_channels=self.backbone.out_feat_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                fid_offset=1,
            )

    def forward(self, x, fid=None):
        h = self.backbone(x)
        if self.use_multihead:
            return self.readout(h, x, fid)
        return h
