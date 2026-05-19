# Upstream code snapshots

Frozen reference copies of the four upstream files the ChargE3Net
benchmark depends on. Included so a reviewer can verify the three
correctness claims the benchmark makes about its environment:

| File | Source (commit-pinned on the matching branch) | Verifies |
|---|---|---|
| `dataset.py` | [`hanaol/charge3net@benchmark/inference-throughput :: src/charge3net/data/dataset.py`](https://github.com/hanaol/charge3net/blob/benchmark/inference-throughput/src/charge3net/data/dataset.py) | `DensityGraphDataset.__getitem__` records `load_time` covering the full `__getitem__` body. The benchmark's arithmetic `file_load_s = load_time - graph_build_s` depends on this being end-to-end. |
| `graph_construction.py` | [`hanaol/charge3net@benchmark/inference-throughput :: src/charge3net/data/graph_construction.py`](https://github.com/hanaol/charge3net/blob/benchmark/inference-throughput/src/charge3net/data/graph_construction.py) | `KdTreeGraphConstructor.__call__` is what `TimedKdTreeGraphConstructor` subclasses; `graph_build_time` is recorded around exactly this call, so it isolates graph construction with no file I/O leakage. Also documents what `num_probes` actually counts (set in `GraphConstructor.__call__` from `probe_target.shape[0]`). |
| `predictions.py` | [`hanaol/charge3net@benchmark/inference-throughput :: src/utils/predictions.py`](https://github.com/hanaol/charge3net/blob/benchmark/inference-throughput/src/utils/predictions.py) | `split_batch` mutates `num_probes`/`num_probe_edges` in place on each yield. The benchmark's comment "don't materialize the generator" depends on this. The mutation pattern is visible in lines 53–56. |
| `e3.py` | [`hanaol/charge3net@benchmark/inference-throughput :: src/charge3net/models/e3.py`](https://github.com/hanaol/charge3net/blob/benchmark/inference-throughput/src/charge3net/models/e3.py) | `E3DensityModel.atom_model` is purely a function of `input_dict` atom keys; the `atom_representation` returned is the same object passed back to each `probe_model` call. So reusing `atom_repr` across chunked probe sub-batches is equivalent to computing it fresh on each, modulo the `pred_utils.split_batch` invariant above. See lines 67–77 for the canonical forward; the benchmark's chunked path matches what `trainer._test_step` does. |

## Drift policy

Same as elsewhere in this directory: do not edit in place. If upstream
changes, regenerate from a fresh `hanaol/charge3net` checkout at the
matching commit. The directory is lint-exempt via `pyproject.toml`.

## What's NOT here

These files exist standalone for review-by-reading; they are **not**
imported. The benchmark on della runs against an actual clone of the
charge3net fork at the matching branch — `inference_benchmark/`
sits inside that clone, and its scripts import `src.charge3net.*` from
the clone's source tree.
