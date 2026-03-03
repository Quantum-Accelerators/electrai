from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.electrai.model.LCN import LatticeConv3d


class LatticeSequential(nn.Sequential):
    def forward(self, x, lattice_vectors):
        for module in self:
            x = module(x, lattice_vectors)
        return x


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, k, **lcn_kwargs):
        super().__init__()

        self.conv1 = LatticeConv3d(
            cin, cout, kernel_size=k, padding_mode="circular", **lcn_kwargs
        )
        self.norm1 = nn.InstanceNorm3d(cout)
        self.act1 = nn.PReLU()

        self.conv2 = LatticeConv3d(
            cout, cout, kernel_size=k, padding_mode="circular", **lcn_kwargs
        )
        self.norm2 = nn.InstanceNorm3d(cout)
        self.act2 = nn.PReLU()

        self.skip = (
            LatticeConv3d(cin, cout, kernel_size=1, **lcn_kwargs)
            if cin != cout
            else nn.Identity()
        )

    def forward(self, x, lattice_vectors):
        h = self.act1(self.norm1(self.conv1(x, lattice_vectors)))
        h = self.norm2(self.conv2(h, lattice_vectors))
        skip_out = (
            self.skip(x, lattice_vectors)
            if isinstance(self.skip, LatticeConv3d)
            else self.skip(x)
        )
        return self.act2(h + skip_out)


class DownsampleBlock(nn.Module):
    def __init__(self, cin, cout, **lcn_kwargs):
        super().__init__()
        self.conv = LatticeConv3d(
            cin, cout, 3, stride=2, padding_mode="circular", **lcn_kwargs
        )
        self.norm = nn.InstanceNorm3d(cout)
        self.act = nn.PReLU()

    def forward(self, x, lattice_vectors):
        return self.act(self.norm(self.conv(x, lattice_vectors)))


class PeriodicUpsampleConv3d(nn.Module):
    def __init__(self, cin, cout, **lcn_kwargs):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = LatticeConv3d(
            cin,
            cout,
            3,
            **lcn_kwargs,
            # padding=1,
            # padding_mode="circular",  # , use_lattice_conv=False
        )
        self.norm = nn.InstanceNorm3d(cout)
        self.act = nn.PReLU()

    def forward(self, x, lattice_vectors):
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
        x = self.up(x)
        x = x[..., 2:-2, 2:-2, 2:-2]
        x = self.conv(x, lattice_vectors)
        x = self.norm(x)
        return self.act(x)


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels,
        depth,
        n_residual_blocks,
        kernel_size,
        use_lattice_conv=False,
        mix_weight=0.1,
        use_radial_embedding=False,
        use_positional_embedding=False,
        trainable_gaussian_params=False,
        num_gaussians=16,
        pos_embed_dim=16,
        hidden_dim=64,
    ):
        super().__init__()

        lcn_kwargs = {
            "use_lattice_conv": use_lattice_conv,
            "mix_weight": mix_weight,
            "use_radial_embedding": use_radial_embedding,
            "use_positional_embedding": use_positional_embedding,
            "trainable_gaussian_params": trainable_gaussian_params,
            "num_gaussians": num_gaussians,
            "pos_embed_dim": pos_embed_dim,
            "hidden_dim": hidden_dim,
        }

        self.in_conv = ResBlock3D(in_channels, n_channels, kernel_size, **lcn_kwargs)

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = n_channels
        for _ in range(depth):
            self.enc_blocks.append(
                LatticeSequential(
                    *[
                        ResBlock3D(ch, ch, kernel_size, **lcn_kwargs)
                        for _ in range(n_residual_blocks)
                    ]
                )
            )
            self.downs.append(DownsampleBlock(ch, 2 * ch, **lcn_kwargs))
            ch *= 2

        # -------- Bottleneck --------
        self.mid = LatticeSequential(
            *[
                ResBlock3D(ch, ch, kernel_size, **lcn_kwargs)
                for _ in range(2 * n_residual_blocks)
            ]
        )

        # -------- Decoder --------
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        # for _ in range(depth):
        #     self.ups.append(PeriodicUpsampleConv3d(ch, ch // 2))
        #     ch //= 2

        #     blocks = [ResBlock3D(2 * ch, ch, kernel_size, **lcn_kwargs)]
        #     blocks.extend(
        #         [
        #             ResBlock3D(ch, ch, kernel_size, **lcn_kwargs)
        #             for _ in range(n_residual_blocks - 1)
        #         ]
        #     )
        #     self.dec_blocks.append(LatticeSequential(*blocks))
        for _ in range(depth):
            self.ups.append(PeriodicUpsampleConv3d(ch, ch // 2, **lcn_kwargs))
            ch //= 2

            blocks = [ResBlock3D(2 * ch, ch, kernel_size, **lcn_kwargs)]
            blocks.extend(
                [
                    ResBlock3D(ch, ch, kernel_size, **lcn_kwargs)
                    for _ in range(n_residual_blocks - 1)
                ]
            )
            self.dec_blocks.append(LatticeSequential(*blocks))

        # -------- Output --------
        # self.out_conv = nn.Conv3d(n_channels, out_channels, kernel_size=1)
        self.out_conv = LatticeConv3d(
            n_channels, out_channels, kernel_size=1, **lcn_kwargs
        )

    def forward(self, x, lattice_vectors):
        skips = []
        out = self.in_conv(x, lattice_vectors)

        for enc, down in zip(self.enc_blocks, self.downs, strict=False):
            out = enc(out, lattice_vectors)
            skips.append(out)
            out = down(out, lattice_vectors)

        out = self.mid(out, lattice_vectors)

        for up, dec in zip(self.ups, self.dec_blocks, strict=False):
            out = up(out, lattice_vectors)
            out = torch.cat([out, skips.pop()], dim=1)
            out = dec(out, lattice_vectors)

        out = self.out_conv(out, lattice_vectors)
        out = out / torch.sum(out, dim=(-3, -2, -1), keepdim=True)
        return out * torch.sum(x, dim=(-3, -2, -1), keepdim=True)


# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.electrai.model.LCN import LatticeConv3d


# class ResBlock3D(nn.Module):
#     def __init__(
#         self,
#         cin,
#         cout,
#         k,
#         use_lattice_conv=False,
#         mix_weight=0.1,
#         use_radial_embedding=False,
#         use_positional_embedding=False,
#         trainable_gaussian_params=False,
#         num_gaussians=16,
#         pos_embed_dim=16,
#         hidden_dim=64,
#     ):
#         super().__init__()
#         self.conv1 = LatticeConv3d(
#             cin,
#             cout,
#             kernel_size=k,
#             padding_mode="circular",
#             padding=k // 2,
#             stride=1,
#             dilation=1,
#             use_lattice_conv=use_lattice_conv,
#             mix_weight=mix_weight,
#             use_radial_embedding=use_radial_embedding,
#             use_positional_embedding=use_positional_embedding,
#             trainable_gaussian_params=trainable_gaussian_params,
#             num_gaussians=num_gaussians,
#             pos_embed_dim=pos_embed_dim,
#             hidden_dim=hidden_dim,
#         )
#         self.norm1 = nn.InstanceNorm3d(cout)
#         self.act1 = nn.PReLU()
#         self.conv2 = LatticeConv3d(
#             cout,
#             cout,
#             kernel_size=k,
#             padding_mode="circular",
#             padding=k // 2,
#             stride=1,
#             dilation=1,
#             use_lattice_conv=use_lattice_conv,
#             mix_weight=mix_weight,
#             use_radial_embedding=use_radial_embedding,
#             use_positional_embedding=use_positional_embedding,
#             trainable_gaussian_params=trainable_gaussian_params,
#             num_gaussians=num_gaussians,
#             pos_embed_dim=pos_embed_dim,
#             hidden_dim=hidden_dim,
#         )
#         self.norm2 = nn.InstanceNorm3d(cout)
#         self.act2 = nn.PReLU()
#         # self.conv1 = nn.Conv3d(cin, cout, k, padding=k // 2, padding_mode="circular")
#         # self.norm1 = nn.InstanceNorm3d(cout)
#         # self.act = nn.PReLU()
#         # self.conv2 = nn.Conv3d(cout, cout, k, padding=k // 2, padding_mode="circular")
#         # self.norm2 = nn.InstanceNorm3d(cout)

#         if cin != cout:
#             self.skip = LatticeConv3d(
#                 cin,
#                 cout,
#                 kernel_size=1,
#                 use_lattice_conv=use_lattice_conv,
#                 mix_weight=mix_weight,
#                 use_radial_embedding=use_radial_embedding,
#                 use_positional_embedding=use_positional_embedding,
#                 trainable_gaussian_params=trainable_gaussian_params,
#                 num_gaussians=num_gaussians,
#                 pos_embed_dim=pos_embed_dim,
#                 hidden_dim=hidden_dim,
#             )
#             # self.skip = nn.Conv3d(cin, cout, 1)
#         else:
#             self.skip = nn.Identity()

#     def forward(self, x, lattice_vectors):
#         h = self.act1(self.norm1(self.conv1(x, lattice_vectors)))
#         h = self.norm2(self.conv2(h, lattice_vectors))
#         return self.act2(h + self.skip(x))


# class ResUNet3D(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n_channels,
#         depth,
#         n_residual_blocks,
#         kernel_size,
#         use_lattice_conv=False,
#         mix_weight=0.1,
#         use_radial_embedding=False,
#         use_positional_embedding=False,
#         trainable_gaussian_params=False,
#         num_gaussians=16,
#         pos_embed_dim=16,
#         hidden_dim=64,
#     ):
#         super().__init__()
#         self.in_conv = ResBlock3D(
#             in_channels,
#             n_channels,
#             kernel_size,
#             use_lattice_conv=use_lattice_conv,
#             mix_weight=mix_weight,
#             use_radial_embedding=use_radial_embedding,
#             use_positional_embedding=use_positional_embedding,
#             trainable_gaussian_params=trainable_gaussian_params,
#             num_gaussians=num_gaussians,
#             pos_embed_dim=pos_embed_dim,
#             hidden_dim=hidden_dim,
#         )
#         # self.in_conv = ResBlock3D(in_channels, n_channels, kernel_size)

#         # -------- Encoder --------
#         self.enc_blocks = nn.ModuleList()
#         self.downs = nn.ModuleList()

#         ch = n_channels
#         for _ in range(depth):
#             self.enc_blocks.append(
#                 nn.Sequential(
#                     *[
#                         ResBlock3D(
#                             ch,
#                             ch,
#                             kernel_size,
#                             use_lattice_conv=use_lattice_conv,
#                             mix_weight=mix_weight,
#                             use_radial_embedding=use_radial_embedding,
#                             use_positional_embedding=use_positional_embedding,
#                             trainable_gaussian_params=trainable_gaussian_params,
#                             num_gaussians=num_gaussians,
#                             pos_embed_dim=pos_embed_dim,
#                             hidden_dim=hidden_dim,
#                         )
#                         for _ in range(n_residual_blocks)
#                     ]
#                 )
#             )
#             # self.enc_blocks.append(
#             #     nn.Sequential(
#             #         *[ResBlock3D(ch, ch, kernel_size,) for _ in range(n_residual_blocks)]
#             #     )
#             # )
#             self.downs.append(
#                 downsample(
#                     ch,
#                     2 * ch,
#                     use_lattice_conv=use_lattice_conv,
#                     mix_weight=mix_weight,
#                     use_radial_embedding=use_radial_embedding,
#                     use_positional_embedding=use_positional_embedding,
#                     trainable_gaussian_params=trainable_gaussian_params,
#                     num_gaussians=num_gaussians,
#                     pos_embed_dim=pos_embed_dim,
#                     hidden_dim=hidden_dim,
#                 )
#             )
#             ch *= 2

#         # -------- Bottleneck --------
#         self.mid = nn.Sequential(
#             *[
#                 ResBlock3D(
#                     ch,
#                     ch,
#                     kernel_size,
#                     use_lattice_conv=use_lattice_conv,
#                     mix_weight=mix_weight,
#                     use_radial_embedding=use_radial_embedding,
#                     use_positional_embedding=use_positional_embedding,
#                     trainable_gaussian_params=trainable_gaussian_params,
#                     num_gaussians=num_gaussians,
#                     pos_embed_dim=pos_embed_dim,
#                     hidden_dim=hidden_dim,
#                 )
#                 for _ in range(2 * n_residual_blocks)
#             ]
#         )
#         # self.mid = nn.Sequential(
#         #     *[ResBlock3D(ch, ch, kernel_size) for _ in range(2 * n_residual_blocks)]
#         # )

#         # -------- Decoder --------
#         self.ups = nn.ModuleList()
#         self.dec_blocks = nn.ModuleList()

#         for _ in range(depth):
#             self.ups.append(PeriodicUpsampleConv3d(ch, ch // 2))
#             ch //= 2
#             self.dec_blocks.append(
#                 nn.Sequential(
#                     *[
#                         ResBlock3D(
#                             2 * ch,
#                             ch,
#                             kernel_size,
#                             use_lattice_conv=use_lattice_conv,
#                             mix_weight=mix_weight,
#                             use_radial_embedding=use_radial_embedding,
#                             use_positional_embedding=use_positional_embedding,
#                             trainable_gaussian_params=trainable_gaussian_params,
#                             num_gaussians=num_gaussians,
#                             pos_embed_dim=pos_embed_dim,
#                             hidden_dim=hidden_dim,
#                         )
#                         for _ in range(n_residual_blocks)
#                     ]
#                 )
#             )
#             # self.dec_blocks.append(
#             #     nn.Sequential(
#             #         *[
#             #             ResBlock3D(2 * ch, ch, kernel_size)
#             #             for _ in range(n_residual_blocks)
#             #         ]
#             #     )
#             # )

#         # -------- Output --------
#         self.out_conv = LatticeConv3d(
#             n_channels,
#             out_channels,
#             kernel_size=1,
#             use_lattice_conv=use_lattice_conv,
#             mix_weight=mix_weight,
#             use_radial_embedding=use_radial_embedding,
#             use_positional_embedding=use_positional_embedding,
#             trainable_gaussian_params=trainable_gaussian_params,
#             num_gaussians=num_gaussians,
#             pos_embed_dim=pos_embed_dim,
#             hidden_dim=hidden_dim,
#         )
#         # self.out_conv = nn.Conv3d(n_channels, out_channels, kernel_size=1)

#     def forward(self, x, lattice_vectors):
#         skips = []
#         out = self.in_conv(x, lattice_vectors)

#         for enc, down in zip(self.enc_blocks, self.downs, strict=False):
#             out = enc(out)
#             skips.append(out)
#             out = down(out)
#         out = self.mid(out)

#         for up, dec in zip(self.ups, self.dec_blocks, strict=False):
#             out = up(out)
#             out = torch.cat([out, skips.pop()], dim=1)
#             out = dec(out)
#         out = self.out_conv(out)
#         out = out / torch.sum(out, axis=(-3, -2, -1))[..., None, None, None]
#         return out * torch.sum(x, axis=(-3, -2, -1))[..., None, None, None]


# class PeriodicUpsampleConv3d(nn.Module):
#     def __init__(
#         self,
#         cin,
#         cout,
#         use_lattice_conv=False,
#         mix_weight=0.1,
#         use_radial_embedding=False,
#         use_positional_embedding=False,
#         trainable_gaussian_params=False,
#         num_gaussians=16,
#         pos_embed_dim=16,
#         hidden_dim=64,
#     ):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
#         self.conv = LatticeConv3d(
#             cin,
#             cout,
#             3,
#             stride=2,
#             padding=1,
#             padding_mode="circular",
#             use_lattice_conv=use_lattice_conv,
#             mix_weight=mix_weight,
#             use_radial_embedding=use_radial_embedding,
#             use_positional_embedding=use_positional_embedding,
#             trainable_gaussian_params=trainable_gaussian_params,
#             num_gaussians=num_gaussians,
#             pos_embed_dim=pos_embed_dim,
#             hidden_dim=hidden_dim,
#         )
#         # self.conv = nn.Conv3d(cin, cout, 3, padding=1, padding_mode="circular")
#         self.norm = nn.InstanceNorm3d(cout)
#         self.act = nn.PReLU()

#     def forward(self, x, lattice_vectors):
#         x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
#         x = self.up(x)
#         x = x[..., 2:-2, 2:-2, 2:-2]
#         x = self.conv(x, lattice_vectors)
#         x = self.norm(x)
#         return self.act(x)


# def downsample(
#     cin,
#     cout,
#     use_lattice_conv=False,
#     mix_weight=0.1,
#     use_radial_embedding=False,
#     use_positional_embedding=False,
#     trainable_gaussian_params=False,
#     num_gaussians=16,
#     pos_embed_dim=16,
#     hidden_dim=64,
# ):
#     return nn.Sequential(
#         LatticeConv3d(
#             cin,
#             cout,
#             3,
#             stride=2,
#             padding=1,
#             padding_mode="circular",
#             use_lattice_conv=use_lattice_conv,
#             mix_weight=mix_weight,
#             use_radial_embedding=use_radial_embedding,
#             use_positional_embedding=use_positional_embedding,
#             trainable_gaussian_params=trainable_gaussian_params,
#             num_gaussians=num_gaussians,
#             pos_embed_dim=pos_embed_dim,
#             hidden_dim=hidden_dim,
#         ),
#         # nn.Conv3d(cin, cout, 3, stride=2, padding=1, padding_mode="circular"),
#         nn.InstanceNorm3d(cout),
#         nn.PReLU(),
#     )
