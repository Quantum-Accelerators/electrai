from __future__ import annotations

import argparse
from collections.abc import Callable

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
parser.add_argument("--chk")
parser.add_argument("--downsample_data")
parser.add_argument("--downsample_label")
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
chk = args.chk


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
                loss_value = 0.0
                for i in range(len(loss_fn["loss"])):
                    loss_value += loss_fn["loss"][i](pred, y).item()
            else:
                loss_value = loss_fn(pred, y).item()
            print(loss_value)
            test_loss += loss_value
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


test_data = RhoData(
    "lists/list_d",
    "lists/list_l",
    "lists/list_dgs",
    "lists/list_lgs",
    downsample_data=downsample_data,
    downsample_label=downsample_label,
    data_augmentation=False,
)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

model = GeneratorResNet(
    n_residual_blocks=n_residual_blocks,
    n_upscale_layers=n_upscale_layers,
    C=C,
    K1=K1,
    K2=K2,
    normalize=normalize,
).to(device)
loss = NormMAE()
chk = torch.load(chk, map_location=torch.device(device))
try:
    model.load_state_dict(chk)
except:
    model.load_state_dict(chk["model_state_dict"])

test_loss = test(test_loader, model, loss, 0)
