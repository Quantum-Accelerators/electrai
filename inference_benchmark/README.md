# Inference throughput benchmark: ResUNet vs ChargE3Net

A production-like single-GPU throughput comparison between electrai's
ResUNet3D and ChargE3Net (Hofgard et al.), on the same set of MP materials.

## What we're measuring

Per-material wall-clock time to predict the full charge density grid for one
structure on a single A100, plus a `cuda.Event` forward-only time. The
headline number is **median time per material on the full grid** — that's
the question someone deploying either model is asking.

Both benchmarks share a CSV schema (`filename, num_atoms, grid_voxels,
forward_ms, load_s, e2e_s, peak_memory_mb, voxels_per_sec_*, warmup`)
so the two output files can be joined on `filename` for a per-material
comparison. `peak_memory_mb` is the marginal GPU peak for processing
one material (counter is reset before each material on the ResUNet
side and before each partial on the ChargE3Net side, then aggregated
as `max` per material) — useful for catching memory-driven scaling
asymmetries the throughput numbers alone miss.

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
    ├── configs/                                # frozen training-config snapshots
    │   ├── SOURCES.md                          #   for the two checkpoints we benchmark
    │   ├── resunet/ckpt_1_training.yaml        #   — verifies inference knobs
    │   └── charge3net/{train_mp_e3_final,      #     match training
    │                   e3_density,
    │                   mp_data,
    │                   kdtree}.yaml
    └── charge3net/                             # reference copy of the
        ├── benchmark_inference.py              #   charge3net side; lives
        ├── launch_benchmark.py                 #   on the
        ├── run_benchmark.slurm                 #   benchmark/inference-throughput
        ├── preprocess_density.py               #   branch of the charge3net
        ├── run_preprocess.slurm                #   fork
        └── upstream/                           # frozen copies of the four
            ├── README.md                       #   upstream files the
            ├── dataset.py                      #   ChargE3Net benchmark
            ├── graph_construction.py           #   depends on, so a
            ├── predictions.py                  #   reviewer can verify
            └── e3.py                           #   its environment claims
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

## Headline result — full rho_gga test sample, n=994 same materials

Both runs on a single A100, fp32, batch_size=1. Same 1000 mpids
sampled (seed=42) from `rho_gga/split_limit_22M.json["test"]`; 994
land in both per-material CSVs after warmup is excluded on each side.

| Metric | ResUNet (n=997) | ChargE3Net (n=994) | **Ratio** |
|---|---|---|---|
| forward_ms median | 85 | 47,352 | **557×** |
| forward_ms p95 | 285 | 186,381 | 655× |
| e2e_s median | 0.110 | 55.4 | **503×** |
| e2e_s p95 | 0.331 | 218.9 | 661× |
| voxels/sec (forward) median | ~13M | ~26k | ~500× |

Wall time on a single A100: **ResUNet 2:46 for 1000 materials**;
**ChargE3Net ~17 h for 749 materials** (the per-material join below
uses 994 materials combined from a salvaged partial run plus this
remaining-materials run).

### Per-material ratio distribution (ChargE3Net / ResUNet)

The aggregate medians don't capture how much the ratio varies by
material. Joined on `filename` (994 materials in both):

| Metric | p5 | **median** | p95 | min | max |
|---|---|---|---|---|---|
| forward_ratio | 298 | **572** | 877 | 49 | 1,288 |
| e2e_ratio | 268 | **514** | 818 | 52 | 1,255 |

**0/994 materials where ChargE3Net is faster than ResUNet on forward.**
The minimum 49× gap is the closest race, not a flip.

The worst-case ChargE3Net forward in the sample (mp-1936073, 44 atoms,
11.2M voxels) is **537 seconds**, vs. 0.71 s for ResUNet on the same
material — a 761× single-material gap. ChargE3Net's forward grows
multiplicatively with cell size *and* grid size because each probe's
graph has more atom edges to evaluate.

### Cost decomposition on ChargE3Net (timing)

Per-material e2e breaks down roughly as **~85% forward, ~14% graph
construction, <1% file I/O**. Graph construction is a real
per-inference cost — every new structure pays it.

### Memory (secondary)

ResUNet: ~1.5 GB median, 11.8 GB worst-case. ChargE3Net: ~7.3 GB
median, 18.5 GB worst-case (~46% of an A100-40GB). The memory ratio
shrinks for big grids: 35/994 materials (3.5%, all small cells with
sparse but large grids) actually use less GPU memory on ChargE3Net
than on ResUNet, because ChargE3Net's memory is bounded by probe-chunk
size while ResUNet's scales with whole-grid 3D-conv activations. See
`joined_per_material.csv` for the full distribution.

### Data products (on della, group-readable)

`/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/inference_benchmark/`

| File | Rows | Contents |
|---|---|---|
| `joined_per_material.csv` | 994 | Per-material join. Both sides' `forward_ms / e2e_s / peak_memory_mb` plus computed `forward_ratio / e2e_ratio / peak_mem_ratio`. The right starting point for plots. |
| `resunet_per_material.csv` | 1000 | ResUNet headline CSV (3 warmup rows tagged in column). |
| `charge3net_per_material.csv` | 994 | ChargE3Net headline CSV. |
| `charge3net_partials_rank0.csv` | 251 | One row per (material, probe_offset) partial chunk from a partial-run salvage. Used to derive ChargE3Net's atom_ms / probe_ms / graph_build_s diagnostics. |
| `charge3net_partials_remaining.csv` | ~750 | Same schema, from the single-A100 remaining-materials run. |

### First-prediction sanity stats

| Model | min | max | mean | std |
|---|---|---|---|---|
| ResUNet | 9.76e-6 | 1.46e-3 | 4.30e-5 | 8.81e-5 |
| ChargE3Net | 0.0145 | 7.84 | 0.493 | 1.263 |

ResUNet outputs are small because of the `density / np.prod(gridsize)`
normalization (total electrons per voxel ~10⁻⁵). ChargE3Net predicts
raw density values (electrons/Å³ scale).

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
2. ~~**Going to the full 1000 materials**~~ — done. Per-material join at
   n=994 is in `joined_per_material.csv` on della. ResUNet completed in
   2:46; ChargE3Net in ~17 h on a single A100 across the sample.
3. ~~**ChargE3Net hyperparameters confirmation**~~ — resolved: the
   training-config snapshots in `configs/charge3net/` confirm the
   `cutoff=4.0, num_interactions=3, num_neighbors=20, mul=500, lmax=4,
   basis="gaussian", num_basis=20, max_predict_batch_probes=2500`
   constants baked into the benchmark match the published checkpoint's
   training config.
4. **`atom_ms` outliers** — `cudnn.benchmark=False` is now set on both
   sides, which should remove the 6–7 s autotune outliers we saw in the
   smoke test. Worth verifying once the post-fix run lands.
5. **`cudnn.benchmark=False` on ResUNet — open whether to flip back.**
   Smoke had flagged ResUNet's forward p95 climbing 298 → 518 ms with
   autotune disabled. The full run shows median 85 ms, p95 285 ms —
   still over the autotune-on smoke baseline, but on a different data
   sample so not directly comparable. Worth a controlled A/B (one full
   run with `cudnn.benchmark=True` on ResUNet only) before quoting the
   forward number externally. Even at 285 ms p95, the ratio vs
   ChargE3Net stays in the same order of magnitude.
