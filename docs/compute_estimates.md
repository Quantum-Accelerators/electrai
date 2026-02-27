# ElectrAI Compute Estimates

Auto-generated from benchmark results. See `src/electrai/scripts/analyze_benchmark.py`.

## 1. Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | ResUNet3D |
| Depth | 2 |
| Channels | 32 |
| Residual blocks | 1 |
| Kernel size | 5 |
| Total parameters | 12,088,659 |
| Trainable parameters | 12,088,659 |
| Model size | 46.12 MB |

## 2. Benchmark Environment

| Property | Value |
|----------|-------|
| Date | 2026-02-27T14:15:39-0500 |
| Hostname | della-l04g5 |
| GPU | NVIDIA A100 80GB PCIe |
| GPU Memory | 85.09 GB |
| CUDA Version | 12.8 |
| PyTorch Version | 2.9.1+cu128 |

## 3. Data Loading Performance

| Metric | Value |
|--------|-------|
| Samples measured | 100 |
| Mean load time | 0.2916s |
| Median load time | 0.0005s |
| P95 load time | 1.9471s |
| Throughput | 3.43 samples/sec |
| DataLoader workers | 8 |

### Shape Distribution

| Shape | Count |
|-------|-------|
| 128x128x128 | 18 |
| 80x80x80 | 5 |
| 72x72x72 | 4 |
| 56x56x56 | 3 |
| 96x96x96 | 3 |
| 108x108x128 | 3 |
| 72x72x128 | 3 |
| 120x120x120 | 2 |
| 128x128x80 | 2 |
| 96x96x128 | 2 |
| 96x96x80 | 2 |
| 96x128x128 | 2 |
| 80x80x128 | 2 |
| 48x80x84 | 2 |
| 56x56x128 | 2 |
| ... | (43 more) |

## 4. GPU Memory Profile

| Shape | Peak Allocated (MB) | Peak Reserved (MB) |
|-------|--------------------:|-------------------:|
| 100x100x128 | 4,761 | 9,574 |
| 108x108x108 | 4,712 | 9,574 |
| 108x108x84 | 3,861 | 6,796 |
| 108x96x96 | 3,868 | 6,796 |
| 112x112x112 | 5,173 | 10,912 |
| 112x112x80 | 3,954 | 7,328 |
| 112x128x128 | 6,448 | 12,490 |
| 120x120x120 | 6,123 | 11,664 |
| 120x120x128 | 6,473 | 12,490 |
| 120x120x72 | 4,057 | 7,874 |
| 128x128x108 | 6,283 | 12,490 |
| 128x128x120 | 6,848 | 13,326 |
| 128x128x128 | 7,244 | 14,212 |
| 128x128x64 | 4,095 | 8,426 |
| 128x128x80 | 4,890 | 10,226 |
| 128x128x96 | 5,686 | 11,664 |
| 36x36x24 | 877 | 912 |
| 40x56x80 | 1,426 | 1,982 |
| 48x48x48 | 1,151 | 1,408 |
| 48x48x80 | 1,383 | 1,982 |
| 56x128x128 | 3,669 | 6,310 |
| 56x56x128 | 2,082 | 2,902 |
| 56x56x96 | 1,789 | 2,690 |
| 60x108x128 | 3,394 | 4,984 |
| 60x60x108 | 2,049 | 2,902 |
| 60x64x60 | 1,669 | 2,406 |
| 64x128x128 | 4,062 | 7,874 |
| 64x64x112 | 2,250 | 3,336 |
| 64x64x64 | 1,763 | 2,690 |
| 72x72x72 | 1,983 | 2,902 |
| 80x120x120 | 4,376 | 8,426 |
| 80x120x128 | 4,608 | 8,980 |
| 80x128x128 | 4,857 | 10,226 |
| 80x80x108 | 2,936 | 4,984 |
| 80x80x128 | 3,358 | 4,984 |
| 84x84x112 | 3,239 | 4,984 |
| 84x84x128 | 3,616 | 5,874 |
| 84x84x96 | 2,894 | 4,402 |
| 84x96x108 | 3,486 | 5,446 |
| 96x128x128 | 5,652 | 10,912 |
| 96x96x64 | 2,734 | 4,402 |
| 96x96x84 | 3,189 | 4,984 |
| 96x96x96 | 3,528 | 5,446 |

### GPU Compute Summary

| Metric | Value |
|--------|-------|
| Mean step time | 801.69ms |
| Median step time | 800.95ms |
| P95 step time | 803.78ms |
| Throughput | 1.25 samples/sec |

## 5. Bottleneck Analysis

**Bottleneck: GPU COMPUTE**

GPU compute (1.2 samples/s) is slower than data loading (3.4 samples/s). Training is GPU-bound; more/faster GPUs will help.

| Phase | Throughput (samples/sec) |
|-------|------------------------:|
| Data loading only | 3.43 |
| GPU compute only | 1.25 |
| End-to-end | 2.24 |

## 6. Training Time Projections

Projections assume 50 epochs, batch_size=1, end-to-end step time of 0.4472s on benchmark GPU. DDP efficiency degrades ~1%/node beyond 2 nodes (model is ~46.1 MB, so gradient all-reduce is small; network overhead is dominated by synchronization).

### 3,000 structures

| GPUs | A100 (1 epoch) | A100 (50 epochs) | H100 (1 epoch) | H100 (50 epochs) | H200 (1 epoch) | H200 (50 epochs) |
|-----:|---:|---:|---:|---:|---:|---:|
| 1 | 22.4min | 18.6hr | 13.3min | 11.1hr | 9.3min | 7.8hr |
| 4 | 6.2min | 5.2hr | 3.7min | 3.1hr | 2.6min | 2.2hr |
| 8 | 3.3min | 2.7hr | 2.0min | 1.6hr | 1.4min | 1.1hr |
| 16 | 1.7min | 1.4hr | 1.0min | 50.3min | 42.1s | 35.1min |
| 32 | 53.1s | 44.2min | 31.7s | 26.4min | 22.1s | 18.4min |

### 6,000 structures

| GPUs | A100 (1 epoch) | A100 (50 epochs) | H100 (1 epoch) | H100 (50 epochs) | H200 (1 epoch) | H200 (50 epochs) |
|-----:|---:|---:|---:|---:|---:|---:|
| 1 | 44.7min | 1.6days | 26.7min | 22.2hr | 18.6min | 15.5hr |
| 4 | 12.4min | 10.4hr | 7.4min | 6.2hr | 5.2min | 4.3hr |
| 8 | 6.6min | 5.5hr | 3.9min | 3.3hr | 2.7min | 2.3hr |
| 16 | 3.4min | 2.8hr | 2.0min | 1.7hr | 1.4min | 1.2hr |
| 32 | 1.8min | 1.5hr | 1.1min | 52.8min | 44.2s | 36.9min |

### 100,000 structures

| GPUs | A100 (1 epoch) | A100 (50 epochs) | H100 (1 epoch) | H100 (50 epochs) | H200 (1 epoch) | H200 (50 epochs) |
|-----:|---:|---:|---:|---:|---:|---:|
| 1 | 12.4hr | 25.9days | 7.4hr | 15.5days | 5.2hr | 10.8days |
| 4 | 3.5hr | 7.2days | 2.1hr | 4.3days | 1.4hr | 3.0days |
| 8 | 1.8hr | 3.8days | 1.1hr | 2.3days | 45.7min | 1.6days |
| 16 | 56.1min | 1.9days | 33.5min | 1.2days | 23.4min | 19.5hr |
| 32 | 29.5min | 1.0days | 17.6min | 14.7hr | 12.3min | 10.2hr |

## 7. Storage Requirements

Measured mean CHGCAR file size: 40.9 MB (sampled 400 files, range 1.1-72.8 MB)

Per structure storage (data + label): ~82 MB

| Dataset Size | Storage (data + labels) |
|-------------:|------------------------:|
| 3,000 | 240 GB |
| 6,000 | 480 GB |
| 100,000 | 7,992 GB |

## 8. Recommended Cluster Configurations

### 6,000 structures

Target: ~1 hour per epoch (50 epochs = ~50 hours total)

- **A100**: 1 GPUs (1 nodes x 1 GPUs/node) - 44.7min/epoch, 1.6days total
- **H100**: 1 GPUs (1 nodes x 1 GPUs/node) - 26.7min/epoch, 22.2hr total
- **H200**: 1 GPUs (1 nodes x 1 GPUs/node) - 18.6min/epoch, 15.5hr total

### 100,000 structures

Target: ~1 hour per epoch (50 epochs = ~50 hours total)

- **A100**: 16 GPUs (4 nodes x 4 GPUs/node) - 56.1min/epoch, 1.9days total
- **H100**: 16 GPUs (4 nodes x 4 GPUs/node) - 33.5min/epoch, 1.2days total
- **H200**: 8 GPUs (2 nodes x 4 GPUs/node) - 45.7min/epoch, 1.6days total
