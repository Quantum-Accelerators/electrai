# Modal vs EC2 GPU Benchmark Analysis

## Setup

Both platforms run the same code (`scripts/e2e_train.py`) with identical config:
- **Model**: ResUNet3D, 32 channels, 16 residual blocks, gradient checkpointing
- **Data**: 50 samples from `s3://openathena/electrai/input/` (≤25MB each)
- **Training**: 5 epochs, seed=42, batch_size=1
- **GPU**: NVIDIA L4 (22 GB)
- **WandB project**: `elf-net-ci`

| Platform | Instance | GPU Access | Container Isolation |
|----------|----------|------------|---------------------|
| EC2 | `g6.xlarge` | Bare-metal NVIDIA driver | Linux namespaces/cgroups |
| Modal | L4 | nvproxy (gVisor ioctl proxy) | gVisor (user-space kernel) |

## Per-Epoch Timing (2026-04-07)

| Epoch | EC2 (s) | Modal (s) | Ratio |
|-------|---------|-----------|-------|
| 0 | 127.6 | **197.2** | **1.55x** |
| 1 | 128.1 | 141.3 | 1.10x |
| 2 | 128.5 | 141.8 | 1.10x |
| 3 | 129.3 | 142.3 | 1.10x |
| 4 | 128.8 | 142.2 | 1.10x |
| **Mean** | **128.5** | **152.9** | **1.19x** |

| Metric | EC2 | Modal |
|--------|-----|-------|
| Wallclock (`trainer.fit`) | 646s | 774s |
| Non-epoch overhead | 3.5s | 9.3s |
| val_loss | 0.246 | 0.241 |

## Analysis

### Epoch 0 warmup penalty (55% slower)

The first epoch on Modal is 70s slower than EC2. This is likely due to **CUDA context initialization and JIT kernel compilation** going through gVisor's nvproxy layer. These operations are ioctl-heavy (many driver round-trips for memory allocation, kernel loading, stream creation), and each ioctl incurs nvproxy's parameter-copy overhead.

On EC2, ioctls go directly to the kernel driver. On Modal, the path is:
```
User code → CUDA runtime → ioctl() → gVisor sentry → nvproxy → host NVIDIA driver → GPU
```

The nvproxy copies ioctl parameter structs between address spaces and translates pointers. For the data path (GPU memory operations), there's zero overhead (shared memory). But the control path (kernel launches, memory allocs, stream syncs) adds microseconds per call, which compounds during initialization.

### Steady-state overhead (10% slower)

After warmup, each epoch is consistently ~13s slower on Modal. At ~30 training steps/epoch, that's ~0.4s per step. This likely reflects:
- nvproxy ioctl overhead per CUDA kernel launch
- Possible differences in CPU-side data loading (gVisor syscall overhead for file I/O)
- Memory allocation patterns (PyTorch's caching allocator going through nvproxy)

### Non-epoch overhead (6s difference)

The 6s extra setup/teardown on Modal includes:
- gVisor container startup → GPU allocation
- FUSE filesystem mount for Volume
- WandB initialization through gVisor's network stack

### Scaling: 50 vs 100 samples (L4, nvproxy overhead)

| | 50 samples | 100 samples |
|---|---|---|
| EC2 L4 epoch (steady) | 128.5s | 300.7s |
| Modal L4 epoch (steady) | 141.9s | 333.5s |
| **Steady-state ratio** | **1.10x** | **1.11x** |
| EC2 L4 epoch 0 | 127.6s | 298.2s |
| Modal L4 epoch 0 | 197.2s | 351.1s |
| **Epoch-0 ratio** | **1.55x** | **1.18x** |
| Total wallclock ratio | 1.20x | 1.12x |

The epoch-0 warmup penalty is roughly fixed (~50-70s extra) and amortizes with more steps per epoch. Steady-state overhead is consistent at ~10-11%.

### GPU comparison: L4 vs A100 (100 samples, S3 ≤25MB)

| Epoch | EC2 L4 | Modal L4 | Modal A100 |
|-------|--------|----------|------------|
| 0 | 298.2s | 351.1s | 195.9s |
| 1-4 avg | 300.7s | 333.5s | 166.6s |
| **Wallclock** | **1505s** | **1693s** | **870s** |
| val_loss | 0.235 | 0.243 | 0.243 |

- **A100 is 1.74x faster than L4** on the same data/config (Modal platform)
- **Modal A100 is 1.73x faster than EC2 L4** (different GPU + nvproxy overhead cancel out)
- Inferred nvproxy overhead on A100: ~11% (same as L4, assuming bare-metal A100 ≈ 150s/epoch)

### Dataset_4 on A100 (large grids, no file size filter)

50 samples from dataset_4 (files up to 73MB, much larger grids than S3 set):

| Epoch | Modal A100 |
|-------|------------|
| 0 | 579.0s |
| 1-4 avg | 479.7s |
| **Wallclock** | **2520s (42 min)** |
| val_loss | 0.163 |

Lower val_loss (0.163 vs 0.243) reflects the different/larger grid data.

### Cost projections

| Scenario | GPU | Samples | Epochs | Est. time | Est. cost |
|----------|-----|---------|--------|-----------|-----------|
| CI benchmark | L4 | 50 (S3) | 5 | ~13 min | ~$0.17 |
| Mid benchmark | A100 | 100 (S3) | 5 | ~15 min | ~$0.45 |
| Dataset_4 benchmark | A100 | 50 | 5 | ~42 min | ~$1.26 |
| Full dataset_2 training | 1×A100 | 5,867 | 50 | ~19 hr | ~$34 |
| Full dataset_2 training | 4×A100 DDP | 5,867 | 50 | ~5 hr (est) | ~$36 |
| Full dataset_2 training | 8×A100 DDP | 5,867 | 50 | ~3 hr (est) | ~$42 |

Modal A100 pricing: ~$1.80/hr. DDP estimates assume ~80% scaling efficiency.

### Amortization for production runs

For Betsy's `dataset_2` runs (~5,867 samples, 50 epochs, A100):
```
Della runtime: ~2 hours (4×A100 DDP)
Modal 1×A100 estimate: ~19 hours
Modal 4×A100 DDP estimate: ~5 hours ($36)
```

The single-GPU estimate is much longer because Betsy uses 4-GPU DDP on Della.
Multi-GPU on Modal requires the `@clustered` decorator (beta) — needs separate benchmarking.

## Volume I/O

Tested separately: Modal Volume reads at ~49 MB/s vs local SSD at ~762 MB/s (15x slower). However, copying data to local `/tmp/` before training showed negligible improvement (~10s), confirming that training is GPU-bound, not I/O-bound, for this dataset size.

## Recommendations

1. **For CI benchmarks**: Accept 10-20% overhead; the faster cold start (~30s vs ~3-5 min EC2 boot) partially compensates
2. **For production training**: The 10% steady-state overhead is likely acceptable given Modal's GPU availability advantage over Lambda Labs
3. **For Modal team**: The epoch-0 warmup penalty is the most impactful issue; CUDA context initialization through nvproxy could potentially be optimized or cached
4. **For multi-GPU (future)**: DDP adds inter-GPU communication overhead that may interact differently with gVisor's network stack; needs separate benchmarking

## Raw Data

### EC2 (GPU Benchmark, `g6.xlarge`, 50 samples)
```
Epoch times: ['127.6s', '128.1s', '128.5s', '129.3s', '128.8s']
Mean epoch: 128.5s
Overhead (wallclock - sum epochs): 3.5s
```

### Modal (L4, dataset=s3, 50 samples)
```
Epoch 0: 197.2s
Epoch 1: 141.3s
Epoch 2: 141.8s
Epoch 3: 142.3s
Epoch 4: 142.2s
Mean epoch: 152.9s
Overhead (wallclock - sum epochs): 9.3s
```

### EC2 (GPU Benchmark, `g6.xlarge`, 100 samples)
```
Epoch times: ['298.2s', '300.2s', '300.8s', '300.5s', '301.2s']
Mean epoch: 300.2s
Overhead (wallclock - sum epochs): 4.1s
val_loss: 0.235187
```

### Modal (L4, dataset=s3, 100 samples)
```
Epoch 0: 351.1s
Epoch 1: 334.1s
Epoch 2: 332.7s
Epoch 3: 332.3s
Epoch 4: 334.8s
Mean epoch: 337.0s
Overhead (wallclock - sum epochs): 8.1s
val_loss: 0.242915
```

### Modal (A100, dataset=s3, 100 samples)
```
Epoch 0: 195.9s
Epoch 1: 166.2s
Epoch 2: 167.5s
Epoch 3: 166.2s
Epoch 4: 166.6s
Mean epoch: 172.5s
Overhead (wallclock - sum epochs): 7.7s
val_loss: 0.243293
```

### Modal (A100, dataset=dataset_4, 50 samples, no file size filter)
```
Epoch 0: 579.0s
Epoch 1: 482.3s
Epoch 2: 476.2s
Epoch 3: 484.0s
Epoch 4: 476.4s
Mean epoch: 499.6s
Overhead (wallclock - sum epochs): 22.6s
val_loss: 0.163158
```
