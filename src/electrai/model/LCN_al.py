from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.electrai.model.utils import DistanceTriangleNet


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
        mix_weight=0.1,  # noqa: ARG002
        use_radial_embedding=False,
        use_positional_embedding=False,
        trainable_gaussian_params=False,
        num_gaussians=16,  # noqa: ARG002
        pos_embed_dim=16,  # noqa: ARG002
        # pos_embed_type="learnable",
        r_max=5.0,  # noqa: ARG002
        hidden_dim=64,  # noqa: ARG002
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
        self.use_radial_embedding = use_radial_embedding
        self.use_positional_embedding = use_positional_embedding

        if use_lattice_conv:
            self.dist_triangle = DistanceTriangleNet(
                d=128,
                n_heads=4,
                n_blocks=2,
                rbf_num=32,
                r_max=16.0,
                dropout=0.0,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            # if use_radial_embedding:
            #     self.gaussian_smear = GaussianRadialBasis(
            #         num_gaussians=num_gaussians,
            #         r_min=0.0,
            #         r_max=r_max,
            #         trainable=self.trainable_gaussian_params,
            #     )

            # if use_positional_embedding:
            #     self.pos_embedding = CartesianFourierEmbedding(num_freqs=6)
            #     # if pos_embed_type == "learnable":
            #     #     self.pos_embedding = PositionalEmbedding(
            #     #         embed_dim=pos_embed_dim, max_kernel_size=max(self.kernel_size)
            #     #     )
            #     # elif pos_embed_type == "fourier":
            #     #     self.pos_embedding = FourierPositionalEmbedding(
            #     #         embed_dim=pos_embed_dim, max_freq=10
            #     #     )
            #     # else:
            #     #     raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")
            #     # self.pos_embed_type = pos_embed_type

            # if use_radial_embedding or use_positional_embedding:
            #     input_size = 0
            #     if use_radial_embedding:
            #         input_size += num_gaussians
            #     if use_positional_embedding:
            #         input_size += pos_embed_dim
            #     self.filter_network = nn.Sequential(
            #         nn.Linear(input_size, hidden_dim),
            #         nn.LayerNorm(hidden_dim),
            #         nn.SiLU(),
            #         # nn.Dropout(0.1),
            #         nn.Linear(hidden_dim, hidden_dim),
            #         nn.LayerNorm(hidden_dim),
            #         nn.SiLU(),
            #         # nn.Dropout(0.1),
            #         nn.Linear(hidden_dim, in_channels * out_channels),
            #     )
            #     # nn.init.zeros_(self.filter_network[-1].weight)
            #     # nn.init.zeros_(self.filter_network[-1].bias)

            # # self.register_buffer("mix_weight", torch.tensor(float(mix_weight)))
            # self.mix_weight = nn.Parameter(torch.tensor(float(mix_weight)))
            # # self.mix_weight = nn.Parameter(torch.zeros(out_channels))

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
            squeeze_batch = False  # noqa: F841

        B = lattice_vectors.shape[0]
        kz, ky, kx = self.kernel_size
        device = lattice_vectors.device

        z = torch.arange(kz, device=device) - kz // 2
        y = torch.arange(ky, device=device) - ky // 2
        x = torch.arange(kx, device=device) - kx // 2

        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        frac_coords = torch.stack([grid_z, grid_y, grid_x], dim=-1).float()

        cart_coords = torch.einsum("ijkl,bml->bijkm", frac_coords, lattice_vectors)
        return cart_coords.reshape(B, -1, 3)

        # distances = torch.norm(cart_coords, dim=-1)

        # # debug_stats = {}
        # # debug_stats["distances"] = {
        # #     "min": distances.min().item(),
        # #     "max": distances.max().item(),
        # #     "mean": distances.mean().item(),
        # #     "has_nan": torch.isnan(distances).any().item(),
        # #     "has_inf": torch.isinf(distances).any().item(),
        # # }

        # if self.use_radial_embedding:
        #     radial_features = self.gaussian_smear(distances)
        #     radial_flat = rearrange(radial_features, "b kz ky kx n -> b (kz ky kx) n")
        # if self.use_positional_embedding:
        #     pos_features = self.pos_embedding(cart_coords)
        #     # pos_features = pos_features.unsqueeze(0).expand(B, -1, -1, -1, -1)
        #     # if self.pos_embed_type == "learnable":
        #     #     pos_features = self.pos_embedding(frac_coords, self.kernel_size)
        #     #     pos_features = pos_features.unsqueeze(0).expand(B, -1, -1, -1, -1)
        #     # else:
        #     #     pos_features = self.pos_embedding(frac_coords)
        #     #     pos_features = pos_features.unsqueeze(0).expand(B, -1, -1, -1, -1)
        #     pos_flat = rearrange(pos_features, "b kz ky kx n -> b (kz ky kx) n")

        # if self.use_radial_embedding and self.use_positional_embedding:
        #     features = torch.cat([radial_flat, pos_flat], dim=-1)
        # elif self.use_radial_embedding:
        #     features = torch.cat([radial_flat], dim=-1)
        # else:
        #     features = torch.cat([pos_flat], dim=-1)

        # kernel_flat = self.filter_network(features)
        # kernel = rearrange(
        #     kernel_flat,
        #     "b (kz ky kx) (o i) -> b o i kz ky kx",
        #     o=self.out_channels,
        #     i=self.in_channels,
        #     kz=kz,
        #     ky=ky,
        #     kx=kx,
        # )
        # # kernel_norm = torch.linalg.vector_norm(
        # #     kernel, dim=(-4, -3, -2, -1), keepdim=True
        # # ).clamp(min=1e-8)
        # # kernel = (
        # #     kernel / kernel_norm * (1.0 / (3 * self.in_channels * kz * ky * kx) ** 0.5)
        # # )
        # # after kernel computed
        # if squeeze_batch:
        #     kernel = kernel.squeeze(0)

        # return kernel

    def forward(self, x, lattice_vectors=None):
        base_kernel = self.conv.weight.unsqueeze(0)
        if not self.use_lattice_conv or (
            not self.use_radial_embedding and not self.use_positional_embedding
        ):
            with torch.no_grad():
                b_rms = base_kernel.pow(2).mean().sqrt()
                self.kernel_stats = {"base_rms": b_rms.item()}
            return self.conv(x)
        D, H, W = x.shape[-3:]
        # scale = x.new_tensor([D, H, W]).view(1, 3, 1)  # (1,3,1)
        a = lattice_vectors[:, 0, :]
        b = lattice_vectors[:, 1, :]
        c = lattice_vectors[:, 2, :]
        lv_voxel = torch.stack([c / D, b / H, a / W], dim=1)  # (z,y,x)
        # lv_voxel = lattice_vectors / scale  # (B,3,3)

        B = x.shape[0]
        # geometric_kernels = self.compute_geometric_kernel(lv_voxel)
        # g_rms = geometric_kernels.pow(2).mean().sqrt().clamp(min=1e-8)
        # b_rms = base_kernel.pow(2).mean().sqrt().clamp(min=1e-8)
        # geometric_kernels = geometric_kernels * (b_rms / g_rms)
        # alpha = torch.sigmoid(self.mix_weight)

        # with torch.no_grad():
        #     g_rms = geometric_kernels.pow(2).mean().sqrt()
        #     b_rms = base_kernel.pow(2).mean().sqrt()
        #     self.kernel_stats = {
        #         "geo_rms": g_rms.item(),
        #         "base_rms": b_rms.item(),
        #         "ratio": (g_rms / (b_rms + 1e-8)).item(),
        #         "alpha": alpha.item(),
        #     }

        # kernels = alpha * geometric_kernels + base_kernel  # (1 - alpha) * base_kernel
        pos = self.compute_geometric_kernel(lv_voxel)
        kernels = base_kernel + self.dist_triangle(pos, self.kernel_size[0])
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
