from __future__ import annotations

import time
from pathlib import Path

import numpy as np
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
            out_dir = Path(self.out_dir)
            out_dir.mkdir(exist_ok=True, parents=True)

            preds = outputs["pred"]
            indices = outputs["index"]

            for i in range(len(indices)):
                idx = indices[i]
                pred_i = preds[i].numpy()
                np.save(out_dir / f"{idx}.npy", pred_i)

        self.test_outputs.append(outputs)

    def on_test_epoch_end(self):
        index = []
        nmae_all = []

        for o in self.test_outputs:
            index.extend(list(o["index"]))

            n = o["nmae"]
            if n.ndim == 0:
                nmae_all.append(n.unsqueeze(0))
            else:
                nmae_all.append(n)

        nmae = torch.cat(nmae_all, dim=0)

        if self.log_dir is not None:
            log_dir = Path(self.log_dir)
            log_dir.mkdir(exist_ok=True, parents=True)
            csv_path = Path(self.log_dir) / "metrics.csv"

            with open(csv_path, "w") as f:
                f.write("index,nmae\n")
                for ind, err in zip(index, nmae.tolist(), strict=False):
                    f.write(f"{ind},{err}\n")
