from __future__ import annotations

import torch
from lightning.pytorch import LightningModule
from src.electrai.model.loss.charge import NormMAE
from src.electrai.model.srgan_layernorm_pbc import GeneratorResNet


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = GeneratorResNet(
            n_residual_blocks=int(cfg.n_residual_blocks),
            n_upscale_layers=int(cfg.n_upscale_layers),
            C=int(cfg.n_channels),
            K1=int(cfg.kernel_size1),
            K2=int(cfg.kernel_size2),
            normalize=cfg.normalize,
            use_checkpoint=getattr(cfg, "use_checkpoint", True),
        )
        self.loss_fn = NormMAE()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
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
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
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
