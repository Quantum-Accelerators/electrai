from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from src.electrai.dataloader.registry import get_data
from src.electrai.loggers.training import get_logger
from src.electrai.model.loss import charge
from src.electrai.model.srgan_layernorm_pbc import GeneratorResNet
from torch.utils.data import DataLoader

logger = get_logger("train", log_file="logs/train.log")


def loss_fn_sum(loss_fn, output, target, t):
    total_loss = 0
    for loss, weight in zip(loss_fn["loss"], loss_fn["weight"], strict=False):
        w = weight(t) if isinstance(weight, Callable) else weight
        total_loss += w * loss(output, target)
    return total_loss


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    t: int,
    accum_iter: int = 1,
) -> None:
    size = len(dataloader.dataset)
    model.train()
    # if isinstance(loss_fn, dict):

    optimizer.zero_grad()
    for batch, (X_in, y_in) in enumerate(dataloader):
        logger.info("batch: %d", batch)
        X, y = X_in.to(device), y_in.to(device)
        pred = model(X)

        if isinstance(loss_fn, dict):
            loss = loss_fn_sum(loss_fn, pred, y, t) / accum_iter
        else:
            loss = loss_fn(pred, y) / accum_iter

        loss.backward()
        if ((batch + 1) % accum_iter == 0) or (batch + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        if batch % 50 == 0:
            current = batch * len(X)
            logger.info("loss: %.6e  [%d/%d]", loss.item(), current, size)


def test_epoch(
    dataloader: DataLoader,
    model: torch.nn.module,
    loss_fn: torch.nn.module,
    device: str,
    t: int,
) -> float:
    num_batches = len(dataloader)
    model.eval()
    test_loss = np.zeros(len(loss_fn["loss"])) if isinstance(loss_fn, dict) else 0
    with torch.no_grad():
        for X_in, y_in in dataloader:
            X, y = X_in.to(device), y_in.to(device)
            pred = model(X)
            if isinstance(loss_fn, dict):
                for i in range(len(loss_fn["loss"])):
                    test_loss[i] += loss_fn["loss"][i](pred, y).item()
            else:
                test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    if isinstance(loss_fn, dict):
        components = test_loss.copy()
        weights = [w(t) if isinstance(w, Callable) else w for w in loss_fn["weight"]]
        test_loss = np.dot(components, weights)
        logger.info("Test Error: Avg loss: %.7e", test_loss)
        logger.info("    Individual loss: %s", components)
    else:
        logger.info("Test Error: Avg loss: %.7e", test_loss)
    return test_loss


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def train(args):
    config_path = Path(args.config)
    with Path.open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = SimpleNamespace(**cfg_dict)

    assert 0 < cfg.train_fraction < 1, "train_fraction must be between 0 and 1."
    # assert 2**cfg.n_upscale_layers == cfg.downsample_data / cfg.downsample_label

    train_data, test_data = get_data(cfg)
    train_loader = DataLoader(train_data, batch_size=int(cfg.nbatch), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=int(cfg.nbatch), shuffle=False)

    model = GeneratorResNet(
        n_residual_blocks=int(cfg.n_residual_blocks),
        n_upscale_layers=int(cfg.n_upscale_layers),
        C=int(cfg.n_channels),
        K1=int(cfg.kernel_size1),
        K2=int(cfg.kernel_size2),
        normalize=not cfg.normalize_label,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )

    # Linear + Cosine scheduler
    linsch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5, end_factor=1, total_iters=1
    )
    cossch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.epochs) - 1
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [linsch, cossch], milestones=[1]
    )

    loss_fn = charge.NormMAE()

    prev_loss = float("inf")
    for t in range(int(cfg.epochs)):
        logger.info("Epoch %d\n%s", t + 1, "-" * 30)
        train_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            cfg.device,
            t,
            accum_iter=int(cfg.nbatch),
        )
        test_loss = test_epoch(test_loader, model, loss_fn, cfg.device, t)

        if test_loss < prev_loss:
            save_checkpoint(t, model, optimizer, scheduler, f"{cfg.model_prefix}.pth")
            prev_loss = test_loss

        if t % int(cfg.save_every_epochs) == 1:
            save_checkpoint(
                t, model, optimizer, scheduler, f"{cfg.model_prefix}_{t}.pth"
            )

        scheduler.step()
        logger.info("Learning Rate: %s", scheduler.get_last_lr())

    logger.info("Training complete!")
