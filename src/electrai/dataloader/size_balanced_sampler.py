"""Size-balanced distributed sampling for full-grid training.

With DDP every optimizer step waits for the slowest rank, and grids in one dataset
can differ in voxel count by more than 30x. A plain shuffled ``DistributedSampler``
therefore makes most ranks idle while one processes a large grid.
``SizeBalancedDistributedSampler`` builds each step from samples of similar size
instead: within shuffled windows the samples are sorted by voxel count and cut into
steps of ``num_replicas * batch_size`` samples, then the step order is shuffled.
Samples, batch size and steps per epoch are the same as with ``DistributedSampler``;
only which samples share a step changes.
"""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset

from electrai.dataloader import utils

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class SizeBalancedDistributedSampler(Sampler[int]):
    """Distributed sampler whose steps group samples of similar size.

    Parameters
    ----------
    sizes
        Cost proxy (voxel count) for every index of the dataset.
    num_replicas, rank
        DDP world size and this process's rank.
    batch_size
        Per-rank batch size of the DataLoader that consumes this sampler.
    window_steps
        Steps per sorting window. The epoch is shuffled, split into windows of
        ``window_steps`` steps, and sorted by size only within each window, so step
        composition changes between epochs. Larger windows balance better; a window
        covering the whole epoch is a full sort.
    seed
        Base seed; the permutation of epoch ``e`` uses ``seed + e``.
    """

    def __init__(
        self,
        sizes: Sequence[float],
        num_replicas: int,
        rank: int,
        batch_size: int = 1,
        window_steps: int = 64,
        seed: int = 0,
    ):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        self.sizes = list(sizes)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.window_steps = max(1, int(window_steps))
        self.seed = int(seed)
        self.epoch = 0
        self.step_size = self.num_replicas * self.batch_size
        self.num_steps = math.ceil(len(self.sizes) / self.step_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def steps(self) -> list[list[int]]:
        """All steps of the current epoch, each a list of ``step_size`` indices."""
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.sizes)))
        rng.shuffle(order)
        # pad by repeating samples, as DistributedSampler does, so every rank gets
        # the same number of samples
        order += order[: self.num_steps * self.step_size - len(order)]
        window = self.window_steps * self.step_size
        steps = []
        for start in range(0, len(order), window):
            chunk = sorted(order[start : start + window], key=self.sizes.__getitem__)
            steps += [
                chunk[i : i + self.step_size]
                for i in range(0, len(chunk), self.step_size)
            ]
        rng.shuffle(steps)
        for step in steps:
            rng.shuffle(step)
        return steps

    def __iter__(self) -> Iterator[int]:
        lo = self.rank * self.batch_size
        for step in self.steps():
            yield from step[lo : lo + self.batch_size]

    def __len__(self) -> int:
        return self.num_steps * self.batch_size


def _grid_voxels(path: Path) -> int:
    """Voxel count of the grid at `path`: zarr array metadata, or file size for text grids."""
    if path.suffix == ".zarr" or utils._is_zarr_store(path):
        import zarr

        z = zarr.open_group(str(path), mode="r")
        name = next(
            n
            for n in ("charge_density_total", "charge_density_up")
            if n in z.array_keys()
        )
        return math.prod(z[name].shape)
    return path.stat().st_size


def _input_paths(dataset: Dataset, indices: Sequence[int] | None = None) -> list[Path]:
    """Input file path of every index (or of `indices`), unwrapping AddDatasetID /
    Subset / ConcatDataset. Only the requested members are resolved: filelists can
    name materials outside the split that have no input file."""
    if indices is None:
        indices = range(len(dataset))
    if isinstance(dataset, ConcatDataset):
        bounds = [0, *dataset.cumulative_sizes]
        return [
            p
            for k, part in enumerate(dataset.datasets)
            for p in _input_paths(
                part, [i - bounds[k] for i in indices if bounds[k] <= i < bounds[k + 1]]
            )
        ]
    if isinstance(dataset, Subset):
        return _input_paths(dataset.dataset, [dataset.indices[i] for i in indices])
    if hasattr(dataset, "base"):  # AddDatasetID
        return _input_paths(dataset.base, indices)
    if dataset.fmt == "explicit":
        directory = dataset.input_dir
    else:
        directory = dataset.root / "data"
    return [utils._resolve_path(directory, dataset.member_list[i]) for i in indices]


def grid_sizes(dataset: Dataset, cache: str | None = None) -> list[int]:
    """Voxel count of every sample in `dataset`, read from metadata only.

    With `cache`, sizes are stored as ``{input path: voxels}`` JSON. Under DDP rank 0
    fills the cache and the other ranks read it after a barrier.
    """
    paths = [str(p) for p in _input_paths(dataset)]
    known: dict[str, int] = {}
    cache_path = Path(cache) if cache else None
    if cache_path is not None and cache_path.exists():
        known = json.loads(cache_path.read_text())
    missing = sorted(set(paths) - known.keys())
    if missing:
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed or dist.get_rank() == 0 or cache_path is None:
            with ThreadPoolExecutor(max_workers=16) as pool:
                known.update(
                    zip(
                        missing,
                        pool.map(lambda p: _grid_voxels(Path(p)), missing),
                        strict=False,
                    )
                )
            if cache_path is not None and (not distributed or dist.get_rank() == 0):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(known))
        if distributed and cache_path is not None:
            dist.barrier()
            if dist.get_rank() != 0:
                known = json.loads(cache_path.read_text())
    return [known[p] for p in paths]
