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
| Input | Low-res CHGCAR | Atom positions + cell |
| Output | High-res density grid (single forward) | Density per probe (chunked, 2500 probes/chunk) |
| Forward call | One `model(data)` | `model.atom_model(...)` once + `model.probe_model(...)` per chunk |
| Forward timing | `cuda.Event` around `model(x)` | `cuda.Event` around atom + probe (summed) |
| Loader | `pymatgen.io.vasp.Chgcar` | `np.load(.npy) + pickle.load(_atoms.pkl)` |
| Warmup | First 3 materials excluded | First 3 partials excluded |

Both run single-GPU, batch_size=1, fp32. ChargE3Net's distributed scaffold
remains in place but is a no-op when `world_size == 1` — per-rank
throughput is what we report.

`torch.backends.cudnn.benchmark` is set to `False` on both sides so
first-of-shape kernels don't pay an autotune cost mid-benchmark; without
this, ChargE3Net's `atom_ms` showed 6–7 s outliers on a few materials
where typical was ~30 ms.

### Out of scope: input-generation cost

The ResUNet input is a **low-resolution CHGCAR** that itself comes from a
fast/cheap DFT run (the `data/` half of `chg_datasets/dataset_4`). That
upstream DFT cost is not measured here — both benchmarks start from
files already on disk. ChargE3Net needs only the structure (atoms +
cell), so its real production cost is roughly what's reported. ResUNet's
real production cost is what's reported **plus** whatever it takes to
generate the low-res input. Whether that delta matters depends on the
deployment scenario; for a like-for-like model-vs-model comparison on
the same end-state grid, the numbers below are the right ones.

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

> ⚠ **Smoke test, n=17, single A100.** Treat these numbers as
> order-of-magnitude. They were collected before the bugfixes in this
> PR (label-CHGCAR load on the ResUNet side; ChargE3Net per-material
> warmup aggregation) and before `cudnn.benchmark=False`. The headline
> run is the full 1000-material sweep on the post-fix code.

| Metric | ResUNet | ChargE3Net | Ratio (smoke, indicative) |
|---|---|---|---|
| forward_ms median | 70 | 47,534 | ~680× |
| forward_ms p95 | 298 | 101,406 | ~340× |
| e2e_s median | 1.43 | 55.3 | ~39× |
| e2e_s p95 | 3.00 | 118.0 | ~39× |
| voxels/sec (forward) median | ~14M | ~21k | ~670× |

The e2e ratio is much smaller than forward-only because both share CHGCAR
load overhead (~1–3 s/material). In a true production setting starting
from a structure rather than a CHGCAR-on-disk, the e2e gap would widen
toward the forward-only ratio (load cost matters more for the fast model).

## Reproducing

### 1. Preprocess data (one-time, ~30 s for 1000 materials)

The ChargE3Net dataloader needs `<mpid>.npy` + `<mpid>_atoms.pkl` files;
electrai reads CHGCARs directly. The preprocessing converts CHGCARs once:

```bash
# On della, from the charge3net fork checkout
sbatch inference_benchmark/run_preprocess.slurm  # LIMIT=1000 by default
```

Outputs `data_preprocessed/{filelist.txt, probe_counts.csv, split.json,
mp-*.npy, mp-*_atoms.pkl}`.

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
