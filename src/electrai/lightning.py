from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningModule
from src.electrai.model.loss.charge import MAE
from src.electrai.model.srgan_layernorm_pbc import GeneratorResNet


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        elf = cfg.data["elf"]
        self.model = GeneratorResNet(
            n_residual_blocks=int(cfg.n_residual_blocks),
            n_upscale_layers=int(cfg.n_upscale_layers),
            C=int(cfg.n_channels),
            K1=int(cfg.kernel_size1),
            K2=int(cfg.kernel_size2),
            use_checkpoint=getattr(cfg, "use_checkpoint", True),
            elf=elf,
        )
        self.cfg = cfg
        self.loss_fn = MAE(elf)

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
        x = batch["data"]
        y = batch["label"]
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

    def on_test_start(self):
        self.log_dir = self.test_cfg.log_dir
        self.out_dir = self.test_cfg.out_dir
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(exist_ok=True, parents=True)
        if self.log_dir is not None:
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)
            self.tmp_dir = Path(self.out_dir) / "tmp"
            self.tmp_dir.mkdir(exist_ok=True, parents=True)
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        start_time = time.time()
        x = batch["data"]
        y = batch["label"]
        indices = batch["index"]

        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        return {
            "pred": preds.detach().cpu(),
            "target": y.detach().cpu(),
            "index": indices,
            "nmae": loss.detach().cpu(),
            "time": time.time() - start_time,  # + batch["load_time"][0], ???
        }

    def on_test_batch_end(self, outputs, batch, batch_idx):
        if self.out_dir is not None:
            preds = outputs["pred"]
            indices = outputs["index"]
            nmae = outputs["nmae"]

            # Save prediction files
            for i in range(len(indices)):
                idx = indices[i]
                np.save(self.out_dir / f"{idx}.npy", preds[i].squeeze(0).cpu().numpy())

        if self.log_dir is not None:
            # Save batch-level CSV
            if isinstance(nmae, torch.Tensor) and nmae.ndim == 0:
                nmae = nmae.unsqueeze(0)
            tmp_csv = self.tmp_dir / f"metrics_batch_{self.global_rank}_{batch_idx}.csv"
            with open(tmp_csv, "w") as f:
                for i, n in zip(indices, nmae, strict=False):
                    idx = i
                    f.write(f"{idx},{n.item()}\n")

    def on_test_epoch_end(self):
        if self.log_dir is None:
            return

        final_csv = self.log_dir / "metrics.csv"

        # gather all batch CSVs
        all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_batch_*.csv"))

        # write final CSV with header
        with open(final_csv, "w") as f_out:
            f_out.write("index,nmae\n")
            for tmp_csv in all_tmp_csvs:
                with open(tmp_csv) as f_in:
                    for line in f_in:
                        f_out.write(line)
