from __future__ import annotations

import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
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
        self.tmp_dir = self.test_cfg.tmp_dir
        self.save_pred = self.test_cfg.save_pred
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        start_time = time.time()
        x = batch["data"]
        y = batch["label"]
        indices = batch["index"]

        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        out = {
            "target": y.detach().cpu(),
            "index": indices,
            "nmae": loss.detach().cpu(),
            "time": time.time() - start_time,
        }
        if self.save_pred:
            out["pred"] = preds.detach().cpu()
        return out

    def on_test_batch_end(self, outputs, batch, batch_idx):
        indices = outputs["index"]
        nmae = outputs["nmae"]

        if self.save_pred:
            preds = outputs["pred"]
            for i in range(len(indices)):
                idx = indices[i]
                np.save(
                    self.out_dir / f"rank_{self.global_rank}_{idx}.npy",
                    preds[i].squeeze(0).cpu().numpy(),
                )

        if isinstance(nmae, torch.Tensor) and nmae.ndim == 0:
            nmae = nmae.unsqueeze(0)
        tmp_csv = self.tmp_dir / f"metrics_batch_{self.global_rank}_{batch_idx}.csv"
        with open(tmp_csv, "w") as f:
            for idx, n in zip(indices, nmae, strict=True):
                f.write(f"{idx},{n.item()}\n")

    def on_test_epoch_end(self):
        is_dist = dist.is_available() and dist.is_initialized()

        # Each rank counts how many tmp CSVs it wrote
        local_count = len(
            list(self.tmp_dir.glob(f"metrics_batch_{self.global_rank}_*.csv"))
        )

        if is_dist:
            # Sum file counts across all ranks so rank 0 knows the expected total
            count_tensor = torch.tensor(
                [local_count], dtype=torch.long, device=self.device
            )
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            expected_total = count_tensor.item()
            dist.barrier()
        else:
            expected_total = local_count

        if self.global_rank == 0:
            final_csv = self.log_dir / "metrics.csv"

            # Retry glob until all files are visible (handles NFS caching)
            all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_batch_*.csv"))
            retries = 0
            while len(all_tmp_csvs) < expected_total and retries < 30:
                time.sleep(1)
                all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_batch_*.csv"))
                retries += 1

            if len(all_tmp_csvs) < expected_total:
                raise RuntimeError(
                    f"Expected {expected_total} CSV files but found {len(all_tmp_csvs)}. Possible NFS caching issue."
                )

            with open(final_csv, "w") as f_out:
                f_out.write("index,nmae\n")
                for tmp_csv in all_tmp_csvs:
                    with open(tmp_csv) as f_in:
                        for line in f_in:
                            f_out.write(line)

            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        if is_dist:
            dist.barrier()
