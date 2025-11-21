from __future__ import annotations

import pytorch_lightning as pl
import torch

from src.electrai.model.loss.charge import NormMAE
from src.electrai.model.srgan_layernorm_pbc import GeneratorResNet


class LightningGenerator(pl.LightningModule):
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
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # release GPU cache for next batch
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # release GPU cache for next batch
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        linsch = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-5, end_factor=1, total_iters=1
        )
        cossch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.cfg.epochs) - 1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linsch, cossch], milestones=[1]
        )
        return [optimizer], [scheduler]
