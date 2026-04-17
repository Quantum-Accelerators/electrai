from __future__ import annotations

import contextlib
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from lightning.pytorch import LightningModule

from electrai.dataloader.patchify import unpatchify
from electrai.model.loss.charge import NormMAE


class LightningGenerator(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.loss_fn = NormMAE()
        # Manual optimization lets us call backward() per patch so each patch's
        # computation graph is freed immediately rather than keeping all graphs
        # alive until a single backward at the end (which OOMs on large grids).
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        opt = self.optimizers()
        opt.zero_grad()

        x = batch["data"]
        y = batch["label"]

        if isinstance(x, list):
            # Patchified or variable-shape: process one patch at a time so each
            # graph is freed after its backward() before the next forward pass.
            # Use no_sync() for all but the last patch so DDP all_reduce fires
            # exactly once per step regardless of how many patches each rank has.
            n = len(x)
            total_loss = torch.tensor(0.0, device=self.device)
            ddp_model = self.trainer.model
            no_sync = getattr(ddp_model, "no_sync", None)
            for i, (x_i, y_i) in enumerate(zip(x, y, strict=True)):
                pred = self(x_i.unsqueeze(0))
                patch_loss = self.loss_fn(pred, y_i.unsqueeze(0)) / n
                ctx = (
                    no_sync()
                    if (no_sync is not None and i < n - 1)
                    else contextlib.nullcontext()
                )
                with ctx:
                    self.manual_backward(patch_loss)
                total_loss = total_loss + patch_loss.detach()
        else:
            pred = self(x)
            total_loss = self.loss_fn(pred, y)
            self.manual_backward(total_loss)
            total_loss = total_loss.detach()

        clip_val = getattr(self.cfg, "gradient_clip_value", 1.0)
        self.clip_gradients(opt, gradient_clip_val=clip_val)
        opt.step()

        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        return total_loss

    def on_train_epoch_end(self):
        self._scheduler.step()

    def validation_step(self, batch):
        loss = self._eval_loss(batch)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def _eval_loss(self, batch):
        """Loss for validation/test — no backward needed, so graph can be freed normally."""
        x = batch["data"]
        y = batch["label"]
        if isinstance(x, list):
            losses = []
            for x_i, y_i in zip(x, y, strict=True):
                pred = self(x_i.unsqueeze(0))
                losses.append(self.loss_fn(pred, y_i.unsqueeze(0)))
            return torch.stack(losses).mean()
        pred = self(x)
        return self.loss_fn(pred, y)

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
        self._scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [linsch, cossch], milestones=[self.cfg.warmup_length]
        )
        # With manual optimization, schedulers must be stepped manually.
        # Return only the optimizer; _scheduler is stepped in on_train_epoch_end.
        return optimizer

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
            # Patchified: run model on each patch then reassemble
            patch_preds = [self(x_i.unsqueeze(0)).squeeze(0) for x_i in x]

            if "original_shape" in batch and "patch_positions" in batch:
                patch_size = x[0].shape[-1]  # patches are cubic (1, P, P, P)
                preds = unpatchify(
                    patch_preds,
                    batch["patch_positions"],
                    patch_size,
                    batch["original_shape"],
                )
                # Reconstruct label the same way for a fair loss comparison
                label_patches = list(y)
                label_full = unpatchify(
                    label_patches,
                    batch["patch_positions"],
                    patch_size,
                    batch["original_shape"],
                )
                loss = self.loss_fn(preds.unsqueeze(0), label_full.unsqueeze(0))
            else:
                # No position metadata; fall back to patch-level loss
                losses = []
                for x_i, y_i in zip(x, y, strict=True):
                    pred_i = self(x_i.unsqueeze(0))
                    losses.append(self.loss_fn(pred_i, y_i.unsqueeze(0)))
                loss = torch.stack(losses).mean()
                preds = patch_preds  # list; save_pred will be skipped below
        else:
            preds = self(x)
            loss = self.loss_fn(preds, y)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        out = {
            "target": y.detach().cpu() if isinstance(y, torch.Tensor) else None,
            "index": indices,
            "nmae": loss.detach().cpu(),
            "duration": elapsed,
        }
        if self.save_pred and isinstance(preds, torch.Tensor):
            out["pred"] = preds.detach().cpu()
        return out

    def on_test_batch_end(self, outputs, _batch, batch_idx):
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

            with final_csv.open("w") as f_out:
                f_out.write("rank,index,nmae\n")
                for tmp_csv in all_tmp_csvs:
                    with tmp_csv.open() as f_in:
                        for line in f_in:
                            f_out.write(line)

            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        if is_dist:
            dist.barrier()
