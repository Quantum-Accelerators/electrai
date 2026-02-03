from __future__ import annotations

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from src.electrai.model.loss.charge import NormMAE, RegE


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.loss_fn = NormMAE()
        self.loss_fn_reg = RegE()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        total, base, reg = self._loss_calculation(batch)
        self.log_dict(
            {"train_loss": total, "train_loss_base": base, "train_loss_reg": reg},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        return total

    def validation_step(self, batch):
        total, base, reg = self._loss_calculation(batch)
        self.log_dict(
            {"val_loss": total, "val_loss_base": base, "val_loss_reg": reg},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return total

    def compute_total_loss(self, pred, y):
        base = self.loss_fn(pred, y)

        reg = torch.zeros((), device=pred.device, dtype=pred.dtype)

        if self.cfg.regularization:
            if self.cfg.reg_schedule:
                lam = lambda_schedule(
                    self.current_epoch,
                    start=2,
                    end=15,
                    lam_max=float(getattr(self.cfg, "lambda_nelec", 1)),
                )
                reg = lam * self.loss_fn_reg(pred, y)
            else:
                lam = float(getattr(self.cfg, "lambda_nelec", 1))
                reg = reg + lam * self.loss_fn_reg(pred, y)

        total = base + reg
        return total, base, reg

    def _loss_calculation(self, batch):
        x = batch["data"]
        y = batch["label"]

        if isinstance(x, list):
            totals, bases, regs = [], [], []

            for x_i, y_i in zip(x, y, strict=True):
                pred_i = self(x_i.unsqueeze(0))
                total_i, base_i, reg_i = self.compute_total_loss(
                    pred_i, y_i.unsqueeze(0)
                )
                totals.append(total_i)
                bases.append(base_i)
                regs.append(reg_i)

            total = torch.stack(totals).mean()
            base = torch.stack(bases).mean()
            reg = torch.stack(regs).mean()

        else:
            pred = self(x)
            total, base, reg = self.compute_total_loss(pred, y)

        return total, base, reg

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


def lambda_schedule(epoch, start=0, end=50, lam_max=10.0):
    if epoch < start:
        return 0.0
    if epoch >= end:
        return lam_max
    t = (epoch - start) / (end - start)
    return lam_max * t
