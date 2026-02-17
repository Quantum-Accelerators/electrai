from __future__ import annotations

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from src.electrai.model.loss.charge import NormMAE


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.loss_fn = NormMAE()

    def forward(self, x, lattice_vectors=None):
        return self.model(x, lattice_vectors)

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
        if hasattr(self.model, "conv1") and hasattr(
            self.model.conv1, "last_debug_stats"
        ):
            stats = self.model.conv1.last_debug_stats
            for key, values in stats.items():
                for metric, val in values.items():
                    self.log(f"debug/{key}/{metric}", val, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch):
        loss = self._loss_calculation(batch)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    # def _log_gaussian_params(self, prefix="train_"):
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, torch.nn.Module) and hasattr(
    #             module, "gaussian_smear"
    #         ):
    #             gaussian_smear = module.gaussian_smear

    #             if hasattr(gaussian_smear, "centers"):
    #                 centers = gaussian_smear.centers
    #                 self.log(
    #                     f"{prefix}gaussian/centers_mean",
    #                     centers.mean(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #                 self.log(
    #                     f"{prefix}gaussian/centers_std",
    #                     centers.std(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #                 self.log(
    #                     f"{prefix}gaussian/centers_min",
    #                     centers.min(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #                 self.log(
    #                     f"{prefix}gaussian/centers_max",
    #                     centers.max(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )

    #             if hasattr(gaussian_smear, "widths"):
    #                 widths = gaussian_smear.widths
    #                 self.log(
    #                     f"{prefix}gaussian/widths_mean",
    #                     widths.mean(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #                 self.log(
    #                     f"{prefix}gaussian/widths_std",
    #                     widths.std(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #                 self.log(
    #                     f"{prefix}gaussian/widths_min",
    #                     widths.min(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #                 self.log(
    #                     f"{prefix}gaussian/widths_max",
    #                     widths.max(),
    #                     on_step=True,
    #                     on_epoch=True,
    #                 )
    #             break

    def _loss_calculation(self, batch):
        x = batch["data"]
        y = batch["label"]
        A = batch["lattice"]
        if isinstance(x, list):
            losses = []
            for x_i, y_i, A_i in zip(x, y, strict=True):
                pred = self(x_i.unsqueeze(0), A_i.unsqueeze(0))
                loss = self.loss_fn(pred, y_i.unsqueeze(0))
                losses.append(loss)
            loss = torch.stack(losses).mean()
        else:
            pred = self(x, A)
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
