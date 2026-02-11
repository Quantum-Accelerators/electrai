from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.electrai.model.utils import (
    FourierPositionalEmbedding,
    GaussianRadialBasis,
    PositionalEmbedding,
)


class LatticeConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding_mode="circular",
        stride=1,
        dilation=1,
        use_lattice_conv=False,
        use_radial_embedding=False,
        use_positional_embedding=False,
        num_gaussians=16,
        pos_embed_dim=16,
        pos_embed_type="learnable",
        r_max=5.0,
        hidden_dim=64,
    ):
        super().__init__()
        padding = kernel_size // 2  # - 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        )
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.use_lattice_conv = use_lattice_conv
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=stride,
            dilation=dilation,
            bias=True,
        )
        w = self.conv.weight.detach().clone()
        del self.conv._parameters["weight"]
        self.conv.register_buffer("weight", w)
        self.use_radial_embedding = use_radial_embedding
        self.use_positional_embedding = use_positional_embedding

        if use_lattice_conv:
            if use_radial_embedding:
                self.gaussian_smear = GaussianRadialBasis(
                    num_gaussians=num_gaussians, r_min=0.0, r_max=r_max, trainable=True
                )

            if use_positional_embedding:
                if pos_embed_type == "learnable":
                    self.pos_embedding = PositionalEmbedding(
                        embed_dim=pos_embed_dim, max_kernel_size=max(self.kernel_size)
                    )
                elif pos_embed_type == "fourier":
                    self.pos_embedding = FourierPositionalEmbedding(
                        embed_dim=pos_embed_dim, max_freq=10
                    )
                else:
                    raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")
                self.pos_embed_type = pos_embed_type

            if use_radial_embedding or use_positional_embedding:
                input_size = 0
                if use_radial_embedding:
                    input_size += num_gaussians
                if use_positional_embedding:
                    input_size += pos_embed_dim
                self.filter_network = nn.Sequential(
                    nn.Linear(input_size, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, in_channels * out_channels),
                )

            # if self.use_radial_embedding or self.use_positional_embedding:
            #     self.mix_weight = nn.Parameter(torch.tensor(0.1))

    def _apply_padding(self, x):
        if self.padding_mode == "circular" and any(p > 0 for p in self.padding):
            pad_3d = (
                self.padding[2],
                self.padding[2],
                self.padding[1],
                self.padding[1],
                self.padding[0],
                self.padding[0],
            )
            return F.pad(x, pad_3d, mode="circular")
        elif self.padding_mode in ["zeros", "reflect", "replicate"] and any(
            p > 0 for p in self.padding
        ):
            pad_3d = (
                self.padding[2],
                self.padding[2],
                self.padding[1],
                self.padding[1],
                self.padding[0],
                self.padding[0],
            )
            return F.pad(x, pad_3d, mode=self.padding_mode)
        else:
            return x

    def compute_geometric_kernel(self, lattice_vectors):
        if lattice_vectors.dim() == 2:
            lattice_vectors = lattice_vectors.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B = lattice_vectors.shape[0]
        kz, ky, kx = self.kernel_size
        device = lattice_vectors.device

        z = torch.arange(kz, device=device) - kz // 2
        y = torch.arange(ky, device=device) - ky // 2
        x = torch.arange(kx, device=device) - kx // 2

        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        frac_coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).float()

        cart_coords = torch.einsum("ijkl,bml->bijkm", frac_coords, lattice_vectors)
        distances = torch.norm(cart_coords, dim=-1)

        if self.use_radial_embedding:
            radial_features = self.gaussian_smear(distances)
            radial_flat = rearrange(radial_features, "b kz ky kx n -> b (kz ky kx) n")
        if self.use_positional_embedding:
            if self.pos_embed_type == "learnable":
                pos_features = self.pos_embedding(frac_coords, self.kernel_size)
                pos_features = pos_features.unsqueeze(0).expand(B, -1, -1, -1, -1)
            else:
                pos_features = self.pos_embedding(frac_coords)
                pos_features = pos_features.unsqueeze(0).expand(B, -1, -1, -1, -1)
            pos_flat = rearrange(pos_features, "b kz ky kx n -> b (kz ky kx) n")

        if self.use_radial_embedding and self.use_positional_embedding:
            features = torch.cat([radial_flat, pos_flat], dim=-1)
        elif self.use_radial_embedding:
            features = torch.cat([radial_flat], dim=-1)
        else:
            features = torch.cat([pos_flat], dim=-1)

        kernel_flat = self.filter_network(features)
        kernel = rearrange(
            kernel_flat,
            "b (kz ky kx) (o i) -> b o i kz ky kx",
            o=self.out_channels,
            i=self.in_channels,
            kz=kz,
            ky=ky,
            kx=kx,
        )

        if squeeze_batch:
            kernel = kernel.squeeze(0)

        return kernel

    def forward(self, x, lattice_vectors=None):
        if not self.use_lattice_conv or (
            not self.use_radial_embedding and not self.use_positional_embedding
        ):
            return self.conv(x)

        B = x.shape[0]
        if lattice_vectors.dim() == 2:
            x_padded = self._apply_padding(x)
            geometric_kernel = self.compute_geometric_kernel(lattice_vectors)
            alpha = 1  # self.mix_weight
            kernel = alpha * geometric_kernel

            return F.conv3d(
                x_padded,
                kernel,
                self.conv.bias,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
            )

        else:
            geometric_kernels = self.compute_geometric_kernel(lattice_vectors)
            alpha = 1  # 0.1  # self.mix_weight
            # self.register_buffer("base_weight", w)  # saved + moved with .to(device), not trained

            # base_kernel = self.conv.weight.unsqueeze(0)
            kernels = alpha * geometric_kernels  # + (1 - alpha) * base_kernel
            x_grouped = x.reshape(1, B * self.in_channels, *x.shape[2:])
            x_grouped = self._apply_padding(x_grouped)
            kernels_grouped = kernels.reshape(
                B * self.out_channels, self.in_channels, *self.kernel_size
            )
            out = F.conv3d(
                x_grouped,
                kernels_grouped,
                bias=None,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=B,
            )

            out = out.reshape(B, self.out_channels, *out.shape[2:])

            if self.conv.bias is not None:
                out = out + self.conv.bias.view(1, -1, 1, 1, 1)
            return out
