from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.electrai.model.utils import CartesianFourierEmbedding, GaussianRadialBasis


class LatticeConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding_mode: str = "circular",
        stride: int = 1,
        dilation: int = 1,
        use_lattice_conv: bool = False,
        mix_weight: float = 0.1,
        use_radial_embedding: bool = False,
        use_positional_embedding: bool = False,
        trainable_gaussian_params: bool = False,
        num_gaussians: int = 16,
        pos_embed_dim: int = 16,
        r_max: float = 5.0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        padding = kernel_size // 2

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
        self.use_radial_embedding = use_radial_embedding
        self.use_positional_embedding = use_positional_embedding
        self.trainable_gaussian_params = trainable_gaussian_params

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

        if use_lattice_conv:
            if use_radial_embedding:
                self.gaussian_smear = GaussianRadialBasis(
                    num_gaussians=num_gaussians,
                    r_min=0.0,
                    r_max=r_max,
                    trainable=self.trainable_gaussian_params,
                )

            if use_positional_embedding:
                self.pos_embedding = CartesianFourierEmbedding(num_freqs=60)  # 6)

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
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, in_channels * out_channels),
                )

                # Optional stabilization (uncomment if desired)
                nn.init.zeros_(self.filter_network[-1].weight)
                nn.init.zeros_(self.filter_network[-1].bias)

            # Learnable mixing weight (scalar)
            # self.mix_weight = nn.Parameter(torch.tensor(float(mix_weight)))
            # Alternative per-out-channel alpha:
            # self.mix_weight = nn.Parameter(torch.full((out_channels,), float(mix_weight)))
            self.mix_weight = nn.Parameter(
                torch.full((out_channels,), float(mix_weight))
            )

    def _apply_padding(self, x: torch.Tensor) -> torch.Tensor:
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

        if self.padding_mode in ["zeros", "reflect", "replicate"] and any(
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

        return x

    def compute_geometric_kernel(self, lattice_vectors: torch.Tensor) -> torch.Tensor:
        """
        lattice_vectors: (B, 3, 3) or (3, 3) in voxel units (z,y,x ordering upstream)
        returns: (B, out, in, kz, ky, kx) or (out, in, kz, ky, kx) if input was (3,3)
        """
        if lattice_vectors.dim() == 2:
            lattice_vectors = lattice_vectors.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B = lattice_vectors.shape[0]  # noqa: F841
        kz, ky, kx = self.kernel_size
        device = lattice_vectors.device

        z = torch.arange(kz, device=device) - kz // 2
        y = torch.arange(ky, device=device) - ky // 2
        x = torch.arange(kx, device=device) - kx // 2

        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        frac_coords = torch.stack(
            [grid_z, grid_y, grid_x], dim=-1
        ).float()  # (kz,ky,kx,3)

        cart_coords = torch.einsum(
            "ijkl,bml->bijkm", frac_coords, lattice_vectors
        )  # (B,kz,ky,kx,3)
        distances = torch.norm(cart_coords, dim=-1)  # (B,kz,ky,kx)

        if self.use_radial_embedding:
            radial_features = self.gaussian_smear(distances)  # (B,kz,ky,kx,Ng)
            radial_flat = rearrange(radial_features, "b kz ky kx n -> b (kz ky kx) n")

        if self.use_positional_embedding:
            pos_features = self.pos_embedding(cart_coords)  # (B,kz,ky,kx,Np)
            pos_flat = rearrange(pos_features, "b kz ky kx n -> b (kz ky kx) n")

        if self.use_radial_embedding and self.use_positional_embedding:
            features = torch.cat([radial_flat, pos_flat], dim=-1)
        elif self.use_radial_embedding:
            features = radial_flat
        else:
            features = pos_flat

        kernel_flat = self.filter_network(features)  # (B, kz*ky*kx, out*in)
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

    def forward(
        self, x: torch.Tensor, lattice_vectors: torch.Tensor | None = None
    ) -> torch.Tensor:
        base_kernel = self.conv.weight.unsqueeze(0)  # (1,out,in,kz,ky,kx)

        if (not self.use_lattice_conv) or (
            (not self.use_radial_embedding) and (not self.use_positional_embedding)
        ):
            with torch.no_grad():
                b_rms = base_kernel.pow(2).mean().sqrt()
                self.kernel_stats = {"base_rms": b_rms.item()}
            return self.conv(x)

        if lattice_vectors is None:
            raise ValueError(
                "lattice_vectors must be provided when use_lattice_conv=True"
            )

        D, H, W = x.shape[-3:]

        # lattice_vectors assumed (B,3,3) in Å; convert to voxel units and reorder to (z,y,x)
        a = lattice_vectors[:, 0, :]
        b = lattice_vectors[:, 1, :]
        c = lattice_vectors[:, 2, :]
        lv_voxel = torch.stack(
            [c / D, b / H, a / W], dim=1
        )  # (B,3,3) in voxel units, z/y/x order

        B = x.shape[0]
        geometric_kernels = self.compute_geometric_kernel(
            lv_voxel
        )  # (B,out,in,kz,ky,kx)

        # Optional global RMS match (comment out if you don't want this constraint)
        # g_rms = geometric_kernels.pow(2).mean().sqrt().clamp(min=1e-8)
        # b_rms = base_kernel.pow(2).mean().sqrt().clamp(min=1e-8)
        # geometric_kernels = geometric_kernels * (b_rms / g_rms)

        # alpha = torch.sigmoid(self.mix_weight)  # scalar
        alpha = torch.sigmoid(self.mix_weight).view(1, self.out_channels, 1, 1, 1, 1)

        with torch.no_grad():
            g_rms2 = geometric_kernels.pow(2).mean().sqrt()
            b_rms2 = base_kernel.pow(2).mean().sqrt()
            self.kernel_stats = {
                "geo_rms": g_rms2.item(),
                "base_rms": b_rms2.item(),
                "ratio": (g_rms2 / (b_rms2 + 1e-8)).item(),
                "alpha": float(alpha.mean().item())
                if alpha.numel() > 1
                else float(alpha.item()),
            }

        # Current mixing rule (as in your snippet):
        mod = torch.tanh(geometric_kernels)  # bounded
        kernels = base_kernel * (1 + alpha * mod)
        # kernels = (
        #     alpha * geometric_kernels + (1 - alpha) * base_kernel
        # )  # (B,out,in,kz,ky,kx)

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
