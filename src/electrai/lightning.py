from __future__ import annotations

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from src.electrai.model.LCN import LatticeConv3d
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

    @rank_zero_only
    def _collect_kernel_stats(self, target_layer: str | None = None) -> None:
        # no need for trainer.is_global_zero now; rank_zero_only handles it
        for name, module in self.model.named_modules():
            if not isinstance(module, LatticeConv3d) or not hasattr(
                module, "kernel_stats"
            ):
                continue
            if target_layer is not None and name != target_layer:
                continue

            s = module.kernel_stats
            if self.cfg.model["use_lattice_conv"]:
                log_dict = {
                    f"kernels/{name}/alpha": s["alpha"],
                    f"kernels/{name}/ratio": s["ratio"],
                    f"kernels/{name}/geo_rms": s["geo_rms"],
                    f"kernels/{name}/base_rms": s["base_rms"],
                }
            else:
                log_dict = {f"kernels/{name}/base_rms": s["base_rms"]}

            # IMPORTANT: let Lightning manage step + syncing
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,  # keep False if you're only logging on rank 0
            )

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

    def on_train_batch_end(self, outputs, batch, batch_idx):  # noqa: ARG002
        self._collect_kernel_stats(target_layer="mid.0.conv1")

    def validation_step(self, batch):
        loss = self._loss_calculation(batch)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

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
            betas=(getattr(self.cfg, "beta1", 0.9), getattr(self.cfg, "beta2", 0.999)),
        )

        # flat = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        # cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.cfg.epochs - 1, eta_min=1e-5
        # )

        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, [flat, cos], milestones=[1]
        # )
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
