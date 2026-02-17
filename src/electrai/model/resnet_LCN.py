from __future__ import annotations

import torch
import torch.nn as nn
from src.electrai.model.LCN import LatticeConv3d
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        K=3,
        use_checkpoint=True,
        use_lattice_conv=False,
        use_radial_embedding=False,
        use_positional_embedding=False,
        trainable_gaussian_params=False,
        num_gaussians=16,
        pos_embed_dim=16,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv1 = LatticeConv3d(
            in_features,
            in_features,
            kernel_size=K,
            padding_mode="circular",
            stride=1,
            dilation=1,
            use_lattice_conv=use_lattice_conv,
            use_radial_embedding=use_radial_embedding,
            use_positional_embedding=use_positional_embedding,
            trainable_gaussian_params=trainable_gaussian_params,
            num_gaussians=num_gaussians,
            pos_embed_dim=pos_embed_dim,
        )
        self.norm1 = nn.InstanceNorm3d(in_features)
        self.act1 = nn.PReLU()
        self.conv2 = LatticeConv3d(
            in_features,
            in_features,
            kernel_size=K,
            padding_mode="circular",
            stride=1,
            dilation=1,
            use_lattice_conv=use_lattice_conv,
            use_radial_embedding=use_radial_embedding,
            use_positional_embedding=use_positional_embedding,
            trainable_gaussian_params=trainable_gaussian_params,
            num_gaussians=num_gaussians,
            pos_embed_dim=pos_embed_dim,
        )
        self.norm2 = nn.InstanceNorm3d(in_features)

    def forward(self, x, lattice_vectors=None):
        if self.use_checkpoint and self.training:
            return x + checkpoint(
                self._forward, x, lattice_vectors, use_reentrant=False
            )
        else:
            return x + self._forward(x, lattice_vectors)

    def _forward(self, x, lattice_vectors):
        out = self.conv1(x, lattice_vectors)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out, lattice_vectors)
        return self.norm2(out)


class GeneratorResNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_residual_blocks=16,
        n_channels=64,
        kernel_size1=5,
        kernel_size2=3,
        normalize=True,
        use_checkpoint=True,
        use_lattice_conv=False,
        use_radial_embedding=False,
        use_positional_embedding=False,
        trainable_gaussian_params=False,
        num_gaussians=16,
        pos_embed_dim=16,
    ):
        super().__init__()
        self.normalize = normalize
        self.use_checkpoint = use_checkpoint
        self.use_lattice_conv = use_lattice_conv

        # First layer
        self.conv1 = LatticeConv3d(
            in_channels,
            n_channels,
            kernel_size=kernel_size1,
            padding_mode="circular",
            stride=1,
            dilation=1,
            use_lattice_conv=use_lattice_conv,
            use_radial_embedding=use_radial_embedding,
            use_positional_embedding=use_positional_embedding,
            trainable_gaussian_params=trainable_gaussian_params,
            num_gaussians=num_gaussians,
            pos_embed_dim=pos_embed_dim,
        )
        self.act1 = nn.PReLU()

        # Residual blocks
        res_blocks = [
            ResidualBlock(
                n_channels,
                K=kernel_size2,
                use_checkpoint=use_checkpoint,
                use_lattice_conv=use_lattice_conv,
            )
            for _ in range(n_residual_blocks)
        ]
        self.res_blocks = nn.ModuleList(res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = LatticeConv3d(
            n_channels,
            n_channels,
            kernel_size=kernel_size2,
            padding_mode="circular",
            stride=1,
            dilation=1,
            use_lattice_conv=use_lattice_conv,
            use_radial_embedding=use_radial_embedding,
            use_positional_embedding=use_positional_embedding,
            trainable_gaussian_params=trainable_gaussian_params,
            num_gaussians=num_gaussians,
            pos_embed_dim=pos_embed_dim,
        )
        self.norm = nn.InstanceNorm3d(n_channels)

        # Final output layer
        self.conv3 = LatticeConv3d(
            n_channels,
            out_channels,
            kernel_size=kernel_size1,
            padding_mode="circular",
            stride=1,
            dilation=1,
            use_lattice_conv=use_lattice_conv,
            use_radial_embedding=use_radial_embedding,
            use_positional_embedding=use_positional_embedding,
            trainable_gaussian_params=trainable_gaussian_params,
            num_gaussians=num_gaussians,
            pos_embed_dim=pos_embed_dim,
        )
        self.act2 = nn.ReLU()

    def forward(self, x, lattice_vectors):
        if isinstance(x, torch.Tensor):
            return self._forward(x, lattice_vectors)
        return [self._forward(xi.unsqueeze(0), lattice_vectors).squeeze(0) for xi in x]

    def _forward(self, x, lattice_vectors=None):
        out1 = self.conv1(x, lattice_vectors)
        out1 = self.act1(out1)
        out = out1
        for block in self.res_blocks:
            out = block(out, lattice_vectors)
        out2 = self.conv2(out, lattice_vectors)
        out2 = self.norm(out2)
        out = torch.add(out1, out2)
        out = self.conv3(out, lattice_vectors)
        out = self.act2(out)

        if self.normalize:
            out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
            out = out * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]
        return out
