from __future__ import annotations

import torch
from lightning.pytorch import LightningModule

from src.electrai.model.loss.charge import NormMAE
from src.electrai.model.srgan_layernorm_pbc import GeneratorResNet
from src.electrai.model.unet_3d_pbc import GeneratorUNet


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Model selection based on config
        model_type = getattr(cfg, "model_type", "resnet").lower()

        if model_type == "resnet":
            self.model = GeneratorResNet(
                n_residual_blocks=int(cfg.n_residual_blocks),
                n_upscale_layers=int(cfg.n_upscale_layers),
                C=int(cfg.n_channels),
                K1=int(cfg.kernel_size1),
                K2=int(cfg.kernel_size2),
                normalize=cfg.normalize,
                use_checkpoint=getattr(cfg, "use_checkpoint", True),
            )
        elif model_type == "unet":
            self.model = GeneratorUNet(
                n_upscale_layers=int(cfg.n_upscale_layers),
                C=int(cfg.n_channels),
                depth=getattr(cfg, "unet_depth", 4),
                K1=int(cfg.kernel_size1),
                K2=int(cfg.kernel_size2),
                channel_multiplier=tuple(
                    getattr(cfg, "channel_multiplier", [1, 2, 4, 8])
                ),
                normalize=cfg.normalize,
                use_checkpoint=getattr(cfg, "use_checkpoint", True),
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Use 'resnet' or 'unet'."
            )

        self.loss_fn = NormMAE()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        loss = self._loss_calculation(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch):
        loss = self._loss_calculation(batch)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def _loss_calculation(self, batch):
        x, y = batch
        if isinstance(x, list):
            losses = []
            for x_i, y_i in zip(x, y, strict=True):
                pred = self(x_i.unsqueeze(0))
                loss = self.loss_fn(pred, y_i.unsqueeze(0))
                losses.append(loss)
            loss = torch.stack(losses).mean()
        else:
            pred = self(x)
            loss = self.loss_fn(pred, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        linsch = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            end_factor=1,
            total_iters=self.cfg.warmup_length,
        )
        cossch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.cfg.epochs) - self.cfg.warmup_length
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linsch, cossch], milestones=[self.cfg.warmup_length]
        )
        return [optimizer], [scheduler]
