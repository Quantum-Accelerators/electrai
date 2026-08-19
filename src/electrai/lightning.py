from __future__ import annotations

import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from lightning.pytorch import LightningModule

from electrai.model.loss.charge import MeanMAE, NormMAE

_LOSS_BY_MODE = {"rho": NormMAE, "elf": MeanMAE}


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        property_mode = getattr(cfg, "property_mode", "rho")
        self.property_mode = property_mode
        model_cfg = dict(cfg.model)
        model_cfg["property_mode"] = property_mode
        self.model = instantiate(model_cfg)
        loss_cls = _LOSS_BY_MODE.get(property_mode)
        if loss_cls is None:
            raise ValueError(
                f"property_mode must be 'rho' or 'elf', got '{property_mode}'"
            )
        self.loss_fn = loss_cls()

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
        # batch["Dataset_ID"] is available for future multi-head model extensions
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
            betas=(getattr(self.cfg, "beta1", 0.9), getattr(self.cfg, "beta2", 0.999)),
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

    def test_step(self, batch):
        x = batch["data"]
        y = batch["label"]
        indices = batch["index"]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if isinstance(x, list):
            preds_list, losses = [], []
            for x_i, y_i in zip(x, y, strict=True):
                p = self(x_i.unsqueeze(0))
                preds_list.append(p)
                losses.append(self.loss_fn(p, y_i.unsqueeze(0)))
            preds = torch.cat(preds_list)
            loss = torch.stack(losses).mean()
        else:
            preds = self(x)
            loss = self.loss_fn(preds, y)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        y_cpu = (
            torch.cat([t.unsqueeze(0) for t in y]).detach().cpu()
            if isinstance(y, list)
            else y.detach().cpu()
        )
        metric_key = "mae" if getattr(self, "property_mode", "rho") == "elf" else "nmae"
        out = {
            "target": y_cpu,
            "index": indices,
            metric_key: loss.detach().cpu(),
            "duration": elapsed,
        }
        if self.save_pred:
            out["pred"] = preds.detach().cpu()
        return out

    def on_test_batch_end(self, outputs, _batch, batch_idx):
        indices = outputs["index"]
        metric_key = "mae" if getattr(self, "property_mode", "rho") == "elf" else "nmae"
        nmae = outputs[metric_key]

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
        tmp_csv = (
            self.tmp_dir / f"metrics_rank_{self.global_rank}_batch_{batch_idx}.csv"
        )
        with tmp_csv.open("w") as f:
            for idx, n in zip(indices, nmae, strict=True):
                f.write(f"rank_{self.global_rank},{idx},{n.item()}\n")

    def on_test_epoch_end(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        # Count only files written by THIS rank
        local_count = len(list(self.tmp_dir.glob(f"metrics_rank_{rank}_batch_*.csv")))

        if is_dist:
            count_tensor = torch.tensor(
                [local_count], dtype=torch.long, device=self.device
            )
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            expected_total = int(count_tensor.item())
            dist.barrier()
        else:
            expected_total = local_count

        final_csv = self.log_dir / "metrics.csv"

        if self.global_rank == 0:
            retries = 0
            all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_rank_*_batch_*.csv"))
            while len(all_tmp_csvs) < expected_total and retries < 60:
                time.sleep(1)
                all_tmp_csvs = sorted(self.tmp_dir.glob("metrics_rank_*_batch_*.csv"))
                retries += 1

            if len(all_tmp_csvs) < expected_total:
                raise RuntimeError(
                    f"Expected {expected_total} CSV files but found {len(all_tmp_csvs)}."
                )

            metric_col = "mae" if self.property_mode == "elf" else "nmae"
            with final_csv.open("w") as f_out:
                f_out.write(f"rank,index,{metric_col}\n")
                for tmp_csv in all_tmp_csvs:
                    with tmp_csv.open() as f_in:
                        for line in f_in:
                            f_out.write(line)

            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        if is_dist:
            dist.barrier()
