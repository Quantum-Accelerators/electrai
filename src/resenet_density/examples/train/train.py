from __future__ import annotations

import argparse
from typing import Callable

from resnet.rho_data import *
from resnet.srgan_layernorm_pbc import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--device", default="cpu")
parser.add_argument("--n_residual_blocks")
parser.add_argument("--n_upscale_layers")
parser.add_argument("--n_channels")
parser.add_argument("--kernel_size1")
parser.add_argument("--kernel_size2")
parser.add_argument(
    "--no_normalize", action="store_true", help="not normalize to correct Nelec"
)
parser.add_argument("--downsample_data")
parser.add_argument("--downsample_label")
parser.add_argument("--model_prefix", default="chk")
parser.add_argument(
    "--save_every_epochs", default=2, help="save checkpoint every this epochs"
)
parser.add_argument("--epochs", default=50)
parser.add_argument("--nbatch", default=1)
parser.add_argument("--lr", default=0.1)
parser.add_argument("--weight_decay", default=0.0)
args = parser.parse_args()

device = args.device
n_residual_blocks = int(args.n_residual_blocks)
n_upscale_layers = int(args.n_upscale_layers)
C = int(args.n_channels)
K1 = int(args.kernel_size1)
K2 = int(args.kernel_size2)
normalize = not args.no_normalize
downsample_data = int(args.downsample_data)
downsample_label = int(args.downsample_label)
assert 2**n_upscale_layers == downsample_data / downsample_label
model_prefix = args.model_prefix
save_every_epochs = int(args.save_every_epochs)
epochs = int(args.epochs)
nbatch = int(args.nbatch)
lr = float(args.lr)
weight_decay = float(args.weight_decay)


def train(dataloader, model, loss_fn, optimizer, t, accum_iter=1):
    size = len(dataloader.dataset)
    model.train()
    if type(loss_fn) is dict:

        def loss_fn_sum(output, target):
            loss = 0
            for l, w in zip(loss_fn["loss"], loss_fn["weight"]):
                if isinstance(w, Callable):
                    w = w(t)
                loss += w * l(output, target)
            return loss

    optimizer.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        if type(loss_fn) is dict:
            loss = loss_fn_sum(pred, y) / accum_iter
        else:
            loss = loss_fn(pred, y) / accum_iter

        # Backpropagation
        loss.backward()
        if ((batch + 1) % accum_iter == 0) or (batch + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7e}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, t):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = np.zeros(len(loss_fn["loss"])) if type(loss_fn) is dict else 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if type(loss_fn) is dict:
                for i in range(len(loss_fn["loss"])):
                    test_loss[i] += loss_fn["loss"][i](pred, y).item()
            else:
                test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    if type(loss_fn) is dict:
        components = test_loss.copy()
        weights = []
        for w in loss_fn["weight"]:
            if isinstance(w, Callable):
                weights.append(w(t))
            else:
                weights.append(w)
        test_loss = np.dot(components, weights)
        print(f"Test Error:  Avg loss: {test_loss:>8f}")
        print("    Individual loss: ", *components)
    else:
        print(f"Test Error:  Avg loss: {test_loss:>7e} \n")
    return test_loss


class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self, output, target):
        mae = self.mae(output, target)
        nelec = torch.sum(target, axis=(-3, -2, -1))
        mae = mae / nelec[..., None, None, None]
        return torch.sum(mae)


train_data = RhoData(
    "lists/list_d",
    "lists/list_l",
    "lists/list_dgs",
    "lists/list_lgs",
    downsample_data=downsample_data,
    downsample_label=downsample_label,
)
test_data = RhoData(
    "lists/list_d",
    "lists/list_l",
    "lists/list_dgs",
    "lists/list_lgs",
    downsample_data=downsample_data,
    downsample_label=downsample_label,
    data_augmentation=False,
)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = GeneratorResNet(
    n_residual_blocks=n_residual_blocks,
    n_upscale_layers=n_upscale_layers,
    C=C,
    K1=K1,
    K2=K2,
    normalize=normalize,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss = NormMAE()
linsch = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-5, end_factor=1, total_iters=1
)
cossch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 1)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [linsch, cossch], milestones=[1]
)


def save(epoch, model, optimizer, scheduler, PATH):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        PATH,
    )


prev_loss = 1e10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model, loss, optimizer, t, accum_iter=nbatch)
    test_loss = test(test_loader, model, loss, t)
    if test_loss < prev_loss:
        save(t, model, optimizer, scheduler, f"{model_prefix}.pth")
        prev_loss = test_loss
    if t % save_every_epochs == 1:
        save(t, model, optimizer, scheduler, f"{model_prefix}_{t}.pth")
    scheduler.step()
    print("Learning Rate: ", *scheduler.get_last_lr())
print("Done!")
