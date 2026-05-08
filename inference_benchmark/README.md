# Inference throughput benchmark: ResUNet vs ChargE3Net

A production-like single-GPU throughput comparison between electrai's
ResUNet3D and ChargE3Net (Hofgard et al.), on the same set of MP materials.

## What we're measuring

Per-material wall-clock time to predict the full charge density grid for one
structure on a single A100, plus a `cuda.Event` forward-only time. The
headline number is **median time per material on the full grid** — that's
the question someone deploying either model is asking.

Both benchmarks share a CSV schema (`filename, num_atoms, grid_voxels,
forward_ms, load_s, e2e_s, voxels_per_sec_*, warmup`) so the two output
files can be joined on `filename` for a per-material comparison.

## Methodology

| | ResUNet (electrai) | ChargE3Net |
|---|---|---|
| Input | Low-res density (zarr) | Atom positions + cell |
| Output | High-res density grid (single forward) | Density per probe (chunked, 2500 probes/chunk) |
| Forward call | One `model(data)` | `model.atom_model(...)` once + `model.probe_model(...)` per chunk |
| Forward timing | `cuda.Event` around `model(x)` | **Outer** `cuda.Event` around atom + entire probe loop, single sync at the end (matches ResUNet's wall-clock-equivalent semantics; per-stage events kept as diagnostic only) |
| Loader cost (per inference) | `zarr.open` → `file_load_s` | `np.load + pickle.load` → `file_load_s`; KdTree graph construction → `graph_build_s` (per-partial) |
| Materials | Sampled from `rho_gga/split_limit_22M.json["test"]` (held out from ckpt_1's training) — same set on both sides | |
| Warmup | First 3 materials excluded | First 3 materials excluded[^1] |

[^1]: Implemented as "drop entire materials whose any partial fell inside
    the warmup window of the first 3 dataloader iterations". For grids
    < `MAX_GRID_SIZE` (1e7 voxels) each material is one partial, so this
    collapses to "first 3 materials". The partial-aware form matters
    only when a single material spans multiple partials.

### Cost accounting on ChargE3Net

The dataloader's `__getitem__` records a single `load_time` covering both
the npy/pkl read and the graph construction. We split these into:

- **`file_load_s`** — npy + atoms.pkl read. The dataloader re-reads on
  every partial (no amortization), so per-material aggregation uses the
  first partial's value, not a sum.
- **`graph_build_s`** — KdTree + supercell unrolling for the probe set.
  This *is* per-partial work (the graph differs per probe slice) and we
  sum it. Graph build is a real per-inference cost — production
  inference pays it on every structure, not just the first one.

Per-material `e2e_s` is reconstructed as `file_load_s + graph_build_s +
forward_ms / 1000`, NOT summed across partials (which would multiply
`file_load_s` by the partial count).

Both run single-GPU, batch_size=1, fp32. ChargE3Net's distributed scaffold
remains in place but is a no-op when `world_size == 1` — per-rank
throughput is what we report.

`torch.backends.cudnn.benchmark` is set to `False` on both sides so
first-of-shape kernels don't pay an autotune cost mid-benchmark; without
this, ChargE3Net's `atom_ms` showed 6–7 s outliers on a few materials
where typical was ~30 ms.

### Production cost framing — what each side's number includes

Both benchmarks start from files already on disk; neither timing
includes the upstream DFT cost. After that, the two sides differ in
what their reported `e2e_s` actually represents:

- **ChargE3Net `e2e_s` ≈ true per-inference production cost.** Of the
  per-material e2e, ~85% is forward, ~14% is graph construction
  (`graph_build_s` — building the KdTree + supercell + probe edges),
  and <1% is file I/O. Graph construction is **not** preprocessing —
  it has to happen on every input structure. So ChargE3Net's e2e is
  honest as a production number even if you hand it a Structure
  object directly from a Python API.
- **ResUNet `e2e_s` = forward + small file read, MINUS the low-res
  DFT.** ResUNet's input is the `data/` half of rho_gga: a coarse
  density that comes from a cheap pre-pass. We don't measure that
  pre-pass here. Real production cost is what's reported **plus** the
  low-res DFT.

For a like-for-like model-vs-model comparison on the same end-state
grid, the numbers below are the right ones. They should not be quoted
as production parity without naming both sides' implicit assumptions
(ResUNet excludes low-res DFT; ChargE3Net's graph build is real).

## Same materials, different repos

The two scripts live in different codebases (electrai vs the charge3net
fork) because each imports its own model. The matched materials are 1000
MP entries from `chg_datasets/dataset_4` — preprocessed once into
ChargE3Net's `.npy` + `_atoms.pkl` format (see
`charge3net/preprocess_chgcars.py`); ResUNet reads the same materials'
CHGCARs directly.

```
electrai (this repo)
├── scripts/benchmark_resunet_throughput.py     # ResUNet benchmark
├── job_resunet_throughput.slurm                # ResUNet slurm wrapper
└── inference_benchmark/
    ├── README.md                               # this file
    └── charge3net/                             # reference copy of the
        ├── benchmark_inference.py              #   charge3net side; lives
        ├── launch_benchmark.py                 #   on the
        ├── run_benchmark.slurm                 #   benchmark/inference-throughput
        ├── preprocess_chgcars.py               #   branch of the charge3net
        └── run_preprocess.slurm                #   fork
```

The `charge3net/` subdirectory is included **as a reference for review** —
the actual scripts run from a checkout of the
[hanaol/charge3net](https://github.com/hanaol/charge3net) fork on branch
`benchmark/inference-throughput`, which has the upstream model and data
loaders in place.

### Drift policy

The `charge3net/` mirror is a frozen copy of the upstream branch and is
NOT subject to this repo's lint policy (pyproject.toml exempts the
directory entirely). Treat it as documentation, not as a buildable
target. If the upstream branch changes, regenerate the mirror by copying
the same files from a fresh checkout — never edit them in place here.

```bash
# To refresh the mirror after upstream changes:
rsync -av --delete \
  /path/to/charge3net-checkout/inference_benchmark/ \
  inference_benchmark/charge3net/
```

## Smoke-test result — order-of-magnitude only

> ⚠ **Smoke test, n=17 same materials (sampled from rho_gga test
> partition), single A100, post-fix code.** Treat as order-of-magnitude.
> The headline run is the full 1000-material sweep.

| Metric | ResUNet | ChargE3Net | Ratio |
|---|---|---|---|
| forward_ms median | 67.7 | 41,416 | ~610× |
| forward_ms p95 | 230 | 76,815 | ~330× |
| e2e_s median | 0.084 | 47.8 | **~570×** |
| e2e_s p95 | 0.273 | 89.7 | ~330× |
| voxels/sec (forward) median | ~14M | ~25k | ~570× |

ChargE3Net's per-material e2e breaks down roughly as **~85% forward,
~14% graph construction, <1% file I/O**. Graph construction is a real
per-inference cost — it gets paid every time the model is called on a
new structure, not just on the first.

**First-prediction sanity stats** (verifies neither model is silently
emitting zeros or NaN):

| Model | min | max | mean | std |
|---|---|---|---|---|
| ResUNet | 9.76e-6 | 1.46e-3 | 4.30e-5 | 8.81e-5 |
| ChargE3Net | 0.0145 | 7.84 | 0.493 | 1.263 |

ResUNet outputs are small in magnitude because of the `density /
np.prod(gridsize)` normalization — total electrons per voxel for a
typical cell is ~10⁻⁵. ChargE3Net predicts raw density values directly
(electrons/Å³ scale).

## Reproducing

### 1. Preprocess data (one-time, ~30 s for 1000 materials)

The ChargE3Net dataloader needs `<mpid>.npy` + `<mpid>_atoms.pkl` files;
electrai reads zarr directly. The preprocessor samples from
`rho_gga/split_limit_22M.json["test"]` (4303 candidates, default sample
size 1000) and converts each material's `<mpid>.zarr` into the
ChargE3Net format:

```bash
# On della, from the charge3net fork checkout
sbatch inference_benchmark/run_preprocess.slurm  # LIMIT=1000, samples test partition
```

Outputs `data_preprocessed/{filelist.txt, probe_counts.csv, split.json,
mp-*.npy, mp-*_atoms.pkl}`. The sampled mpids are the *same set* the
ResUNet benchmark reads from rho_gga (matched by `filelist.txt`).

### 2. ChargE3Net benchmark

```bash
# From the charge3net fork checkout, branch benchmark/inference-throughput
sbatch --export=ALL,SHAKEDOWN_LIMIT=20 inference_benchmark/run_benchmark.slurm
```

### 3. ResUNet benchmark

```bash
# From this repo
sbatch --export=ALL,SHAKEDOWN=20 job_resunet_throughput.slurm
```

Both write to `/scratch/.../charge3net-benchmark/{resunet_results,inference_benchmark/results}/throughput_by_material.csv`.

### 4. Compare

The CSVs share `filename` — join them in pandas/notebook to produce
per-material comparison plots (forward time vs grid_voxels, ratio
distributions, etc.).

## Open questions for review

1. **fp16/bf16 sweep for both models** — ChargE3Net's `E3DensityModel`
   is the bigger model and has more to gain from reduced precision.
   Doing the sweep on only ResUNet would artificially widen the gap, so
   either both or neither.
2. **Going to the full 1000 materials** — ChargE3Net would need ~15 h
   single-A100, or shard across 4 GPUs (`NPROCS=4`) for ~4 h. ResUNet
   finishes 1000 in ~20 min. Decide if the smoke-test shape is enough or
   we want the headline run (after the post-fix re-run).
3. **ChargE3Net hyperparameters** — the script bakes in
   `cutoff=4.0, num_interactions=3, num_neighbors=20, mul=500, lmax=4,
   basis="gaussian", num_basis=20` from `train_mp_e3_final.yaml`. Worth
   confirming with Hananeh that those match the published `charge3net_mp.pt`
   checkpoint before publishing the headline number.
4. **`atom_ms` outliers** — `cudnn.benchmark=False` is now set on both
   sides, which should remove the 6–7 s autotune outliers we saw in the
   smoke test. Worth verifying once the post-fix run lands.
5. **`cudnn.benchmark=False` on ResUNet is likely costing real time.**
   Post-fix smoke showed ResUNet's forward p95 climbed 298 → 518 ms (the
   median moved -6%). This is probably not n=17 noise — ResUNet conv
   shapes vary across materials (different unit cells → different grid
   sizes), so with autotune off, every material pays the slow-default-
   kernel cost. Two cleaner options before publishing a headline number:
   - **Pre-warm per shape**: walk the dataset once and run a dummy
     forward at each unique grid shape with `cudnn.benchmark=True`, then
     start measurement. ResUNet has a small number of distinct shapes;
     ChargE3Net's per-chunk variance defeats this anyway, so it's an
     honest win for ResUNet only.
   - **Per-side asymmetry**: leave `cudnn.benchmark=True` on ResUNet,
     off on ChargE3Net, document why. Defensible but easy to get wrong
     in re-runs.

   Either is preferable to the current "off everywhere" once we go to
   the full run. The current setting is acceptable to land this PR
   because the smoke result is explicitly labelled order-of-magnitude.
