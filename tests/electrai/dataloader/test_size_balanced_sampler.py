from __future__ import annotations

import random

import pytest

from electrai.dataloader.size_balanced_sampler import SizeBalancedDistributedSampler


def _samplers(sizes, world, **kw):
    return [
        SizeBalancedDistributedSampler(sizes, num_replicas=world, rank=r, **kw)
        for r in range(world)
    ]


@pytest.mark.parametrize("batch_size", [1, 3])
def test_covers_every_sample_once_per_epoch_with_padding(batch_size):
    sizes = [random.Random(i).randint(1, 1000) for i in range(103)]
    samplers = _samplers(sizes, 8, batch_size=batch_size, window_steps=4, seed=1)
    per_rank = [list(s) for s in samplers]
    assert {len(x) for x in per_rank} == {len(samplers[0])}
    seen = [i for x in per_rank for i in x]
    assert set(seen) == set(range(len(sizes)))
    assert len(seen) == len(samplers[0]) * 8 >= len(sizes)


def test_steps_group_similar_sizes():
    rng = random.Random(0)
    sizes = [rng.choice([10, 100, 1000, 10000]) for _ in range(4096)]
    world = 32
    full_sort = SizeBalancedDistributedSampler(
        sizes, num_replicas=world, rank=0, window_steps=10**6
    )
    balanced_cost = sum(max(sizes[i] for i in step) for step in full_sort.steps())
    order = list(range(len(sizes)))
    rng.shuffle(order)
    random_cost = sum(
        max(sizes[i] for i in order[k : k + world]) for k in range(0, len(order), world)
    )
    assert balanced_cost < 0.5 * random_cost


def test_ranks_agree_and_epochs_differ():
    sizes = list(range(256))
    a, b = _samplers(sizes, 2, window_steps=8, seed=3)
    for s in (a, b):
        s.set_epoch(0)
    assert a.steps() == b.steps()
    first = a.steps()
    a.set_epoch(1)
    assert a.steps() != first
