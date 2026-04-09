from __future__ import annotations

import argparse

import torch

from electrai.entrypoints.test import test
from electrai.entrypoints.train import train

torch.backends.cudnn.conv.fp32_precision = "tf32"


def main() -> None:
    """Entry point.

    Parameters
    ----------
    args : list[str]
        list of command line arguments

    Raises
    ------
    RuntimeError
        if no command was input
    """
    parser = argparse.ArgumentParser(description="Electrai Entry Point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, required=True)

    test_parser = subparsers.add_parser("test", help="Evaluate the model")
    test_parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
