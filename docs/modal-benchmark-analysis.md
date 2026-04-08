# Modal vs EC2 GPU Benchmark Analysis

## Setup

Both platforms run the same code (`scripts/e2e_train.py`) with identical config:
- **Model**: ResUNet3D, 32 channels, 16 residual blocks, gradient checkpointing
- **Data**: 50 samples from `s3://openathena/electrai/input/` (≤25MB each)
- **Training**: 5 epochs, seed=42, batch_size=1
- **GPU**: NVIDIA L4 (22 GB)
- **WandB project**: [`elf-net-ci`][wb-ci], [`elf-net-ci-test`][wb-test]

| Platform | Instance | GPU Access | Container Isolation |
|----------|----------|------------|---------------------|
| EC2 | `g6.xlarge` | Bare-metal NVIDIA driver | Linux namespaces/cgroups |
| Modal | L4 | nvproxy (gVisor ioctl proxy) | gVisor (user-space kernel) |

## Per-Epoch Timing (2026-04-07)

50 S3 samples, 5 epochs. [EC2 GHA][gha-ec2-50] / [Modal GHA][gha-modal-50].

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

100-sample runs: [EC2 GHA][gha-ec2-100] ([WandB][wb-ec2-100]) / [Modal L4 GHA][gha-modal-l4-100] ([WandB][wb-modal-l4-100]) / [Modal A100 GHA][gha-modal-a100-100] ([WandB][wb-modal-a100-100]).

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

50 samples from dataset_4 (files up to 73MB, much larger grids than S3 set). [GHA][gha-modal-a100-d4-50] / [WandB][wb-modal-a100-d4-50].

| Epoch | Modal A100 |
|-------|------------|
| 0 | 579.0s |
| 1-4 avg | 479.7s |
| **Wallclock** | **2520s (42 min)** |
| val_loss | 0.163 |

Lower val_loss (0.163 vs 0.243) reflects the different/larger grid data.

## Multi-GPU Scaling (DDP)

### DDP strategy comparison

Lightning's standard `ddp` strategy (uses `torch.multiprocessing.spawn`) crashes on Modal with `missing output for previous input` — gVisor incompatibility with process forking. `ddp_notebook` (thread-based) works but is GIL-bound (~50% GPU util). **`torchrun`** (subprocess-based, same as Betsy's Della setup) works well.

### Multi-GPU scaling: 187 S3 samples, 5 epochs

[1×A100 WandB][wb-1xa100-187] / [2×A100 WandB][wb-2xa100-187] / [4×A100 WandB][wb-4xa100-187].

| Config | Wallclock | Speedup |
|--------|-----------|---------|
| 1×A100 (`run_training`, 4 workers) | 927s | 1.0x |
| 2×A100 (`torchrun`, 8 workers) | 719s | 1.29x |
| 4×A100 (`torchrun`, 16 workers) | 525s | 1.77x |

Limited scaling due to small sample count (28-57 train steps/GPU/epoch).

### Multi-GPU scaling: 1,000 dataset_4 samples, 2 epochs

[1×A100 GHA][gha-1xa100-1000] ([WandB][wb-1xa100-1000]) / [2×A100 GHA][gha-2xa100-1000] ([WandB][wb-2xa100-1000]) / [4×A100 GHA][gha-4xa100-1000] ([WandB][wb-4xa100-1000]).

| Config | Steps/epoch/GPU | Wallclock | Speedup | GPU util (when active) | Active % |
|--------|-----------------|-----------|---------|----------------------|----------|
| 1×A100 | 600 | ~7140s (119 min) | 1.0x | 90% | 54% |
| 2×A100 | 300 | 3565s (59 min) | **2.00x** | 94-96% | 82% |
| 4×A100 | 150 | 2256s (38 min) | **3.16x** | 99-100% | 75-83% |

Excellent scaling at this sample count. GPUs reach 90-100% utilization when active; idle periods are epoch transitions, validation, and data loading gaps.

### GPU utilization details (1,000 dataset_4 samples)

All GPUs hit 90-100% when actively computing. The "active %" reflects periodic idle time (validation, epoch boundaries). Multi-GPU runs are actually MORE active than single-GPU (82% vs 54%) because `torchrun` data workers keep the pipeline fed better across multiple processes.

### Data workers impact

Adding `train_workers` was a major improvement:
- 1×A100 without workers (old code): 870s, 46% GPU util ([WandB][wb-modal-a100-100-noworkers])
- 1×A100 with 4 workers: ~496s (estimated), 90% GPU util
- **~1.75x speedup from workers alone**

### Cost projections (updated with measured data)

| Scenario | GPU | Samples | Epochs | Measured time | Est. cost |
|----------|-----|---------|--------|---------------|-----------|
| CI benchmark | L4 | 50 (S3) | 5 | 13 min | $0.17 |
| Mid benchmark | 1×A100 | 100 (S3) | 5 | 15 min | $0.45 |
| Dataset_4 benchmark | 1×A100 | 50 | 5 | 42 min | $1.26 |
| Dataset_4 scaling | 1×A100 | 1,000 | 2 | 119 min | $3.57 |
| Dataset_4 scaling | 2×A100 | 1,000 | 2 | 59 min | $3.54 |
| Dataset_4 scaling | 4×A100 | 1,000 | 2 | 38 min | $4.56 |
| Full dataset_4 | 4×A100 | 2,885 | 50 | ~16 hr (est) | ~$115 |
| Full dataset_2 | 4×A100 | 5,867 | 50 | ~33 hr (est) | ~$238 |

Modal A100 pricing: ~$1.80/GPU/hr.

### Comparison with Della (Betsy's runs)

Betsy's closest comparable run: [`resunet-128-dataset`][wb-betsy-d4] — dataset_4, 2,885 samples, 50 epochs on Della (probably 4×A100 DDP):
- **Runtime: 6.1 hours**
- **Config: 16 channels, depth=3, kernel_size=5, val_frac=0.005**

**Not directly comparable to our benchmarks** (32ch, depth=2, kernel_size=3, val_frac=0.4). Our model has ~4x more parameters and trains on only 60% of data (vs 99.5%). A proper comparison requires running the same config on both platforms.

Betsy's `dataset_2` baseline runs (5,867 samples, 50 epochs): ~2.1 hours on Della. Same caveat about different model config applies.

## Volume I/O

Tested separately: Modal Volume reads at ~49 MB/s vs local SSD at ~762 MB/s (15x slower). However, copying data to local `/tmp/` before training showed negligible improvement for small datasets (~10s). For larger datasets (1,000+ samples), Volume I/O may contribute to the single-GPU "active %" gap (54% for 1×A100 vs 82% for multi-GPU with more workers).

## Recommendations

1. **For CI benchmarks**: Accept 10-20% overhead; the faster cold start (~30s vs ~3-5 min EC2 boot) partially compensates
2. **For production training**: Use `torchrun` (not `ddp` or `ddp_notebook`). Scale `train_workers` with GPU count (4× per GPU).
3. **For Modal team**: (a) Epoch-0 warmup penalty through nvproxy; (b) `torch.multiprocessing.spawn` crashes under gVisor
4. **For cost efficiency**: 2×A100 offers the best $/speedup ratio (2.0x speedup for 2x cost). 4×A100 is 3.16x for 4x cost (79% efficient).

## Raw Data

### EC2 L4, 50 S3 samples, 5 epochs ([GHA][gha-ec2-50])
```
Epoch times: ['127.6s', '128.1s', '128.5s', '129.3s', '128.8s']
Mean epoch: 128.5s
Overhead (wallclock - sum epochs): 3.5s
```

### Modal L4, 50 S3 samples, 5 epochs ([GHA][gha-modal-50] / [WandB][wb-modal-50])
```
Epoch 0: 197.2s
Epoch 1: 141.3s
Epoch 2: 141.8s
Epoch 3: 142.3s
Epoch 4: 142.2s
Mean epoch: 152.9s
Overhead (wallclock - sum epochs): 9.3s
```

### EC2 L4, 100 S3 samples, 5 epochs ([GHA][gha-ec2-100] / [WandB][wb-ec2-100])
```
Epoch times: ['298.2s', '300.2s', '300.8s', '300.5s', '301.2s']
Mean epoch: 300.2s
Overhead (wallclock - sum epochs): 4.1s
val_loss: 0.235187
```

### Modal L4, 100 S3 samples, 5 epochs ([GHA][gha-modal-l4-100] / [WandB][wb-modal-l4-100])
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

### Modal 1×A100, 100 S3 samples, 5 epochs ([GHA][gha-modal-a100-100] / [WandB][wb-modal-a100-100])
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

### Modal 1×A100, 50 dataset_4 samples, 5 epochs ([GHA][gha-modal-a100-d4-50] / [WandB][wb-modal-a100-d4-50])
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

### Modal 1×A100, 187 S3 samples, 5 epochs ([WandB][wb-1xa100-187])
```
Mean epoch: 183.8s
Wallclock: 926.6s
Overhead: 7.6s
GPU 0: 90% util (when active)
```

### Modal 2×A100 torchrun, 187 S3 samples, 5 epochs ([WandB][wb-2xa100-187])
```
Wallclock: 719s (speedup: 1.29x)
GPU 0: 94%, GPU 1: 69%
```

### Modal 4×A100 torchrun, 187 S3 samples, 5 epochs ([WandB][wb-4xa100-187])
```
Wallclock: 525s (speedup: 1.77x)
GPU 0-3: 79-83% avg, 92-97% when active
```

### Modal 1×A100, 1000 dataset_4 samples, 2 epochs ([GHA][gha-1xa100-1000] / [WandB][wb-1xa100-1000])
```
Steps/epoch: 600
Wallclock: ~7140s (119 min)
GPU 0: 90% when active, 54% active time
```

### Modal 2×A100 torchrun, 1000 dataset_4 samples, 2 epochs ([GHA][gha-2xa100-1000] / [WandB][wb-2xa100-1000])
```
Steps/epoch: 300
Wallclock: 3565s (59 min) — speedup: 2.00x
GPU 0: 96% when active, 82% active time
GPU 1: 94% when active, 82% active time
```

### Modal 4×A100 torchrun, 1000 dataset_4 samples, 2 epochs ([GHA][gha-4xa100-1000] / [WandB][wb-4xa100-1000])
```
Steps/epoch: 150
Wallclock: 2256s (38 min) — speedup: 3.16x
GPU 0: 100% when active, 83% active time
GPU 1: 99% when active, 83% active time
GPU 2: 100% when active, 75% active time
GPU 3: 100% when active, 75% active time
```

<!-- GHA run links -->
[gha-ec2-50]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24092095128
[gha-modal-50]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24092066067
[gha-ec2-100]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24094632673
[gha-modal-l4-100]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24094614519
[gha-modal-a100-100]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24103642324
[gha-modal-a100-d4-50]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24100220728
[gha-1xa100-1000]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24112212490
[gha-2xa100-1000]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24112219405
[gha-4xa100-1000]: https://github.com/Quantum-Accelerators/electrai/actions/runs/24112226168

<!-- WandB run links -->
[wb-ci]: https://wandb.ai/PrinceOA/elf-net-ci
[wb-test]: https://wandb.ai/PrinceOA/elf-net-ci-test
[wb-modal-50]: https://wandb.ai/PrinceOA/elf-net-ci/runs/y9cdfids
[wb-ec2-100]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/n37g2x2j
[wb-modal-l4-100]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/h7xfq7w6
[wb-modal-a100-100]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/cg1n5is4
[wb-modal-a100-100-noworkers]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/cg1n5is4
[wb-modal-a100-d4-50]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/7rx1ze41
[wb-1xa100-187]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/evobk3xs
[wb-2xa100-187]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/gaybe1k3
[wb-4xa100-187]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/4r07if0k
[wb-1xa100-1000]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/zcpqunmx
[wb-2xa100-1000]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/ohryawsu
[wb-4xa100-1000]: https://wandb.ai/PrinceOA/elf-net-ci-test/runs/4a7wc4qx
[wb-betsy-d4]: https://wandb.ai/PrinceOA/mp-resunet-ablation/runs/n739i5kd
