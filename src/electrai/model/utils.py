from __future__ import annotations

import torch
import torch.nn as nn


class GaussianRadialBasis(nn.Module):
    def __init__(self, num_gaussians=50, r_min=0.0, r_max=5.0, trainable=False):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.r_min = r_min
        self.r_max = r_max

        # Initialize Gaussian centers uniformly between r_min and r_max
        centers = torch.linspace(r_min, r_max, num_gaussians)

        # Initialize widths based on spacing
        # Common choice: width = distance between adjacent centers
        spacing = (r_max - r_min) / (num_gaussians - 1) if num_gaussians > 1 else 1.0
        widths = torch.ones(num_gaussians) * spacing

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, distances):
        """
        Expand distances using Gaussian basis functions.

        Args:
            distances: Tensor of shape [...] containing interatomic distances

        Returns:
            Tensor of shape [..., num_gaussians] with Gaussian features
        """
        # distances: [...] -> [..., 1]
        # centers: [num_gaussians] -> [1, ..., 1, num_gaussians]
        distances = distances.unsqueeze(-1)  # [..., 1]
        centers = self.centers.view(
            *([1] * (distances.dim() - 1)), -1
        )  # [1, ..., num_gaussians]
        widths = self.widths.view(*([1] * (distances.dim() - 1)), -1)

        # Gaussian RBF: exp(-(d - c)^2 / (2 * w^2))
        diff = distances - centers
        gamma = 1.0 / (2 * widths**2)
        return torch.exp(-gamma * diff**2)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding for kernel positions.
    Similar to positional encodings in Transformers but learnable.
    """

    def __init__(self, embed_dim=32, max_kernel_size=7):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable embeddings for each coordinate dimension
        # Range: [-max_kernel_size//2, max_kernel_size//2]
        self.z_embed = nn.Embedding(max_kernel_size, embed_dim)
        self.y_embed = nn.Embedding(max_kernel_size, embed_dim)
        self.x_embed = nn.Embedding(max_kernel_size, embed_dim)

        # Optional: learnable way to combine the three directions
        self.combine = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, frac_coords, kernel_size):
        """
        Args:
            frac_coords: [kz, ky, kx, 3] fractional coordinates (z, y, x offsets)
            kernel_size: (kz, ky, kx) tuple

        Returns:
            embeddings: [kz, ky, kx, embed_dim]
        """
        kz, ky, kx = kernel_size

        # Convert coordinates to indices (shift from [-k//2, k//2] to [0, k])
        z_idx = (frac_coords[..., 0] + kz // 2).long()
        y_idx = (frac_coords[..., 1] + ky // 2).long()
        x_idx = (frac_coords[..., 2] + kx // 2).long()

        # Get embeddings for each dimension
        z_emb = self.z_embed(z_idx)  # [kz, ky, kx, embed_dim]
        y_emb = self.y_embed(y_idx)  # [kz, ky, kx, embed_dim]
        x_emb = self.x_embed(x_idx)  # [kz, ky, kx, embed_dim]

        # Combine (can use addition, concatenation + linear, etc.)
        combined = torch.cat([z_emb, y_emb, x_emb], dim=-1)  # [kz, ky, kx, 3*embed_dim]
        return self.combine(combined)  # [kz, ky, kx, embed_dim]


class FourierPositionalEmbedding(nn.Module):
    """
    Fourier features for positional encoding (non-learnable but more expressive).
    Similar to what's used in NeRF.
    """

    def __init__(self, embed_dim=32, max_freq=10):
        super().__init__()
        self.embed_dim = embed_dim

        # Number of frequency bands
        num_freqs = embed_dim // 6  # 6 because we have sin+cos for each of 3 coords

        # Logarithmically spaced frequencies
        freq_bands = 2.0 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, frac_coords):
        """
        Args:
            frac_coords: [..., 3] fractional coordinates

        Returns:
            embeddings: [..., embed_dim]
        """
        # frac_coords: [..., 3]
        # freq_bands: [num_freqs]

        coords_expanded = frac_coords.unsqueeze(-2)  # [..., 1, 3]
        freqs_expanded = self.freq_bands.unsqueeze(-1)  # [num_freqs, 1]

        # Compute sin and cos for each frequency and coordinate
        angles = coords_expanded * freqs_expanded  # [..., num_freqs, 3]

        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        # Concatenate all features
        fourier_features = torch.cat(
            [sin_features, cos_features], dim=-2
        )  # [..., 2*num_freqs, 3]
        return fourier_features.reshape(
            *frac_coords.shape[:-1], -1
        )  # [..., 6*num_freqs]
