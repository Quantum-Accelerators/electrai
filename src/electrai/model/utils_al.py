from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def pairwise_dist(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    R: (B, N, 3) or (N, 3)
    returns D: (B, N, N) or (N, N)
    """
    if R.ndim == 2:
        R = R[None, ...]
        squeeze = True
    else:
        squeeze = False

    diff = R[:, :, None, :] - R[:, None, :, :]  # (B,N,N,3)
    D = torch.linalg.norm(diff, dim=-1).clamp_min(eps)
    if squeeze:
        D = D[0]
    return D


# 2) RBF embedder for distances (recommended)


class RBF(nn.Module):
    def __init__(self, num=32, r_max=10.0, trainable=False):
        super().__init__()
        centers = torch.linspace(0.0, r_max, num)
        widths = torch.full((num,), (r_max / num))
        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, D):
        """
        D: (B,N,N) distances
        returns: (B,N,N,num)
        """
        # (B,N,N,1) - (num,) -> (B,N,N,num)
        x = D[..., None] - self.centers
        return torch.exp(-0.5 * (x / self.widths).pow(2))


class TriangleAttention(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.0, starting=True):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dk = d // n_heads
        self.starting = starting

        self.ln = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.to_out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, P, mask=None):
        """
        P: (B,N,N,d)
        mask: (B,N) with 1 for real atoms, 0 for padding
        """
        B, N, _, d = P.shape
        x = self.ln(P)

        if self.starting:
            # fixed i: update row i using keys from (i,k)
            q = x.reshape(B * N, N, d)
            k = x.reshape(B * N, N, d)
            v = x.reshape(B * N, N, d)
        else:
            # fixed j: update col j using keys from (k,j)
            xt = x.transpose(1, 2)
            q = xt.reshape(B * N, N, d)
            k = xt.reshape(B * N, N, d)
            v = xt.reshape(B * N, N, d)

        q = self.to_q(q).view(B * N, N, self.h, self.dk).transpose(1, 2)  # (BN,h,N,dk)
        k = self.to_k(k).view(B * N, N, self.h, self.dk).transpose(1, 2)
        v = self.to_v(v).view(B * N, N, self.h, self.dk).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) / (self.dk**0.5)  # (BN,h,N,N)

        if mask is not None:
            kv_mask = mask.repeat_interleave(N, dim=0)  # (BN,N)
            attn = attn.masked_fill(kv_mask[:, None, None, :] == 0, float("-inf"))

        w = self.drop(F.softmax(attn, dim=-1))
        out = (w @ v).transpose(1, 2).contiguous().view(B * N, N, d)
        out = self.to_out(out).view(B, N, N, d)

        if not self.starting:
            out = out.transpose(1, 2)

        return P + out


class DistanceTriangleNet(nn.Module):
    def __init__(
        self,
        d=128,
        n_heads=4,
        n_blocks=2,
        rbf_num=32,
        r_max=10.0,
        dropout=0.0,
        in_channels=32,
        out_channels=32,
    ):
        super().__init__()
        self.rbf = RBF(num=rbf_num, r_max=r_max, trainable=False)

        self.pair_in = nn.Sequential(nn.Linear(rbf_num, d), nn.SiLU(), nn.Linear(d, d))

        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        TriangleAttention(
                            d, n_heads=n_heads, dropout=dropout, starting=True
                        ),
                        TriangleAttention(
                            d, n_heads=n_heads, dropout=dropout, starting=False
                        ),
                        nn.LayerNorm(d),
                        nn.Sequential(
                            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
                        ),
                    ]
                )
            )

        self.node_ln = nn.LayerNorm(d)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_head = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, self.in_channels * self.out_channels),
        )

    def forward(self, R, k: int = 5, mask=None):
        """
        R: (B,N,3) coordinates
        mask: (B,N) optional
        returns y: (B,N) per-atom values
        """
        N = R.shape[0]  # noqa: F841
        D = pairwise_dist(R)  # (B,N,N)
        # Optional: zero-out padded interactions
        if mask is not None:
            m2 = mask[:, :, None] * mask[:, None, :]  # (B,N,N)
            D = D * m2 + (1.0 - m2) * D.max().detach()  # keep padded far away

        P = self.pair_in(self.rbf(D))  # (B,N,N,d)

        for tri_s, tri_e, ln, ff in self.blocks:
            P = tri_s(P, mask=mask)
            P = tri_e(P, mask=mask)
            P = P + ff(ln(P))

        # pair -> node (row/col pooling)
        row = P.mean(dim=2)  # (B,N,d)
        col = P.mean(dim=1)  # (B,N,d)
        H = 0.5 * (row + col)

        y = self.node_head(self.node_ln(H))  # (B,N)
        y = rearrange(
            y,
            "B (k1 k2 k3) (o i)-> B o i k1 k2 k3",
            k1=k,
            k2=k,
            k3=k,
            o=self.out_channels,
            i=self.in_channels,
        )
        if mask is not None:
            y = y * mask
        return y


class GaussianRadialBasis(nn.Module):
    def __init__(self, num_gaussians=50, r_min=0.0, r_max=5.0, trainable=False):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.r_min = r_min
        self.r_max = r_max
        centers = torch.linspace(r_min, r_max, num_gaussians)
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
        distances = distances.unsqueeze(-1)
        centers = self.centers.view(*([1] * (distances.dim() - 1)), -1)
        widths = self.widths.view(*([1] * (distances.dim() - 1)), -1)

        diff = distances - centers
        gamma = 1.0 / (2 * widths**2)
        return torch.exp(-gamma * diff**2)


class CartesianFourierEmbedding(nn.Module):
    """
    Fourier features of real displacement vector (cartesian).
    Much more meaningful than index or fractional embedding.
    """

    def __init__(self, num_freqs=6, include_radius=True):
        super().__init__()

        # smooth low frequencies — not exponential like NeRF
        freqs = torch.linspace(0.5, 3.0, num_freqs)
        self.register_buffer("freqs", freqs)

        self.include_radius = include_radius
        self.out_dim = 2 * num_freqs * 3 + (2 if include_radius else 0)

    def forward(self, cart_coords):
        """
        cart_coords: [B, kz, ky, kx, 3] real displacement vectors (Å)
        returns: positional features
        """

        # normalize scale for stability
        # prevents huge lattice from exploding features
        scale = (
            cart_coords.norm(dim=-1, keepdim=True).mean(dim=(1, 2, 3, 4), keepdim=True)
            + 1e-6
        )
        v = cart_coords / scale

        # project onto frequencies
        # [B,kz,ky,kx,3] -> [B,kz,ky,kx,F,3]
        angles = v.unsqueeze(-2) * self.freqs.view(1, 1, 1, 1, -1, 1)

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        feat = torch.cat([sin, cos], dim=-2).reshape(*v.shape[:-1], -1)

        if self.include_radius:
            r = torch.norm(v, dim=-1, keepdim=True)
            feat = torch.cat([feat, r, r**2], dim=-1)

        return feat


# class PositionalEmbedding(nn.Module):
#     """
#     Learnable positional embedding for kernel positions.
#     Similar to positional encodings in Transformers but learnable.
#     """

#     def __init__(self, embed_dim=32, max_kernel_size=7):
#         super().__init__()
#         self.embed_dim = embed_dim

#         # Learnable embeddings for each coordinate dimension
#         # Range: [-max_kernel_size//2, max_kernel_size//2]
#         self.z_embed = nn.Embedding(max_kernel_size, embed_dim)
#         self.y_embed = nn.Embedding(max_kernel_size, embed_dim)
#         self.x_embed = nn.Embedding(max_kernel_size, embed_dim)

#         # Optional: learnable way to combine the three directions
#         self.combine = nn.Linear(3 * embed_dim, embed_dim)

#     def forward(self, frac_coords, kernel_size):
#         """
#         Args:
#             frac_coords: [kz, ky, kx, 3] fractional coordinates (z, y, x offsets)
#             kernel_size: (kz, ky, kx) tuple

#         Returns:
#             embeddings: [kz, ky, kx, embed_dim]
#         """
#         kz, ky, kx = kernel_size

#         # Convert coordinates to indices (shift from [-k//2, k//2] to [0, k])
#         z_idx = (frac_coords[..., 0] + kz // 2).long()
#         y_idx = (frac_coords[..., 1] + ky // 2).long()
#         x_idx = (frac_coords[..., 2] + kx // 2).long()

#         # Get embeddings for each dimension
#         z_emb = self.z_embed(z_idx)  # [kz, ky, kx, embed_dim]
#         y_emb = self.y_embed(y_idx)  # [kz, ky, kx, embed_dim]
#         x_emb = self.x_embed(x_idx)  # [kz, ky, kx, embed_dim]

#         # Combine (can use addition, concatenation + linear, etc.)
#         combined = torch.cat([z_emb, y_emb, x_emb], dim=-1)  # [kz, ky, kx, 3*embed_dim]
#         return self.combine(combined)  # [kz, ky, kx, embed_dim]


# class FourierPositionalEmbedding(nn.Module):
#     """
#     Fourier features for positional encoding (non-learnable but more expressive).
#     Similar to what's used in NeRF.
#     """

#     def __init__(self, embed_dim=32, max_freq=10):
#         super().__init__()
#         self.embed_dim = embed_dim

#         # Number of frequency bands
#         num_freqs = embed_dim // 6  # 6 because we have sin+cos for each of 3 coords

#         # Logarithmically spaced frequencies
#         freq_bands = 2.0 ** torch.linspace(0, max_freq, num_freqs)
#         self.register_buffer("freq_bands", freq_bands)

#     def forward(self, frac_coords):
#         """
#         Args:
#             frac_coords: [..., 3] fractional coordinates

#         Returns:
#             embeddings: [..., embed_dim]
#         """
#         # frac_coords: [..., 3]
#         # freq_bands: [num_freqs]

#         coords_expanded = frac_coords.unsqueeze(-2)  # [..., 1, 3]
#         freqs_expanded = self.freq_bands.unsqueeze(-1)  # [num_freqs, 1]

#         # Compute sin and cos for each frequency and coordinate
#         angles = coords_expanded * freqs_expanded  # [..., num_freqs, 3]

#         sin_features = torch.sin(angles)
#         cos_features = torch.cos(angles)

#         # Concatenate all features
#         fourier_features = torch.cat(
#             [sin_features, cos_features], dim=-2
#         )  # [..., 2*num_freqs, 3]
#         return fourier_features.reshape(
#             *frac_coords.shape[:-1], -1
#         )  # [..., 6*num_freqs]
