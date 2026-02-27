"""Analyze benchmark results and generate compute estimates report.

Takes the JSON output from benchmark.py and produces a filled-in
docs/compute_estimates.md with training time projections, storage
estimates, and cluster recommendations.

Usage:
    uv run python src/electrai/scripts/analyze_benchmark.py \
        --results benchmark_results.json \
        --output docs/compute_estimates.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# GPU specs: name -> (HBM bandwidth TB/s, HBM capacity GB)
GPU_SPECS = {
    "A100": {"bandwidth_tbs": 2.0, "memory_gb": 80},
    "H100": {"bandwidth_tbs": 3.35, "memory_gb": 80},
    "H200": {"bandwidth_tbs": 4.8, "memory_gb": 141},
}

# DDP scaling efficiency (per-node, 4 GPUs/node assumed)
DDP_EFFICIENCY_INTRA_NODE = 0.90  # NVLink within a single node
DDP_EFFICIENCY_INTER_NODE_BASE = 0.85  # 2-node baseline

DATASET_SIZES = [3000, 6000, 100000]
GPU_COUNTS = [1, 4, 8, 16, 32]
GPUS_PER_NODE = 4


def detect_benchmark_gpu(metadata: dict) -> str | None:
    """Try to match the benchmark GPU to a known type."""
    gpu_name = metadata.get("gpu_name", "").upper()
    for key in GPU_SPECS:
        if key in gpu_name:
            return key
    return None


def compute_scaling_factor(benchmark_gpu: str, target_gpu: str) -> float:
    """Compute speed scaling factor based on HBM bandwidth ratio.

    3D convolutions at this model size are memory-bandwidth-bound,
    so we scale by bandwidth ratio.
    """
    if benchmark_gpu == target_gpu:
        return 1.0
    bench_bw = GPU_SPECS[benchmark_gpu]["bandwidth_tbs"]
    target_bw = GPU_SPECS[target_gpu]["bandwidth_tbs"]
    return bench_bw / target_bw


def ddp_efficiency(n_gpus: int) -> float:
    """Estimate DDP efficiency for a given GPU count.

    Uses graduated scaling: intra-node (<=4 GPUs) gets NVLink efficiency,
    inter-node degrades ~1% per additional node beyond 2 to account for
    increasing all-reduce communication overhead.
    """
    if n_gpus <= 1:
        return 1.0
    if n_gpus <= GPUS_PER_NODE:
        return DDP_EFFICIENCY_INTRA_NODE
    n_nodes = max(1, n_gpus // GPUS_PER_NODE)
    # ~1% degradation per additional node beyond 2
    return max(0.70, DDP_EFFICIENCY_INTER_NODE_BASE - 0.01 * (n_nodes - 2))


def estimate_epoch_time(
    dataset_size: int, step_time_s: float, n_gpus: int, gpu_scaling: float
) -> float:
    """Estimate wall-clock time for one epoch in seconds."""
    samples_per_gpu = dataset_size / n_gpus
    eff = ddp_efficiency(n_gpus)
    return (samples_per_gpu * step_time_s * gpu_scaling) / eff


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}hr"
    return f"{seconds / 86400:.1f}days"


def generate_report(results: dict, output_path: str) -> None:
    """Generate the compute estimates markdown report."""
    metadata = results["metadata"]
    model_summary = results["model_summary"]
    config = results["config"]
    phase1 = results["phase1_data_loading"]
    phase2 = results["phase2_gpu_compute"]
    phase3 = results["phase3_end_to_end"]

    benchmark_gpu = detect_benchmark_gpu(metadata)
    train_epochs = int(config.get("epochs", 50))

    # Use end-to-end step time as the primary metric
    step_time_s = phase3.get("mean_time_s", 0)

    # Bottleneck detection
    data_throughput = phase1.get("samples_per_sec", 0)
    compute_throughput = phase2.get("overall", {}).get("samples_per_sec", 0)
    e2e_throughput = phase3.get("samples_per_sec", 0)

    if data_throughput > 0 and compute_throughput > 0:
        if data_throughput < compute_throughput * 0.8:
            bottleneck = "DATA LOADING"
            bottleneck_detail = (
                f"Data loading ({data_throughput:.1f} samples/s) is slower than "
                f"GPU compute ({compute_throughput:.1f} samples/s). "
                "Consider increasing num_workers or switching to a faster data format."
            )
        elif compute_throughput < data_throughput * 0.8:
            bottleneck = "GPU COMPUTE"
            bottleneck_detail = (
                f"GPU compute ({compute_throughput:.1f} samples/s) is slower than "
                f"data loading ({data_throughput:.1f} samples/s). "
                "Training is GPU-bound; more/faster GPUs will help."
            )
        else:
            bottleneck = "BALANCED"
            bottleneck_detail = (
                "Data loading and GPU compute throughput are roughly matched."
            )
    else:
        bottleneck = "UNKNOWN"
        bottleneck_detail = "Could not determine bottleneck (missing throughput data)."

    # Build report
    lines = []
    lines.append("# ElectrAI Compute Estimates")
    lines.append("")
    lines.append(
        "Auto-generated from benchmark results. "
        "See `src/electrai/scripts/analyze_benchmark.py`."
    )
    lines.append("")

    # Section 1: Model Configuration
    lines.append("## 1. Model Configuration")
    lines.append("")
    model_cfg = config.get("model", {})
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Architecture | ResUNet3D |")
    lines.append(f"| Depth | {model_cfg.get('depth', 'N/A')} |")
    lines.append(f"| Channels | {model_cfg.get('n_channels', 'N/A')} |")
    lines.append(f"| Residual blocks | {model_cfg.get('n_residual_blocks', 'N/A')} |")
    lines.append(f"| Kernel size | {model_cfg.get('kernel_size', 'N/A')} |")
    lines.append(f"| Total parameters | {model_summary['total_params']:,} |")
    lines.append(f"| Trainable parameters | {model_summary['trainable_params']:,} |")
    lines.append(f"| Model size | {model_summary['model_size_mb']:.2f} MB |")
    lines.append("")

    # Section 2: Benchmark Environment
    lines.append("## 2. Benchmark Environment")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Date | {metadata.get('timestamp', 'N/A')} |")
    lines.append(f"| Hostname | {metadata.get('hostname', 'N/A')} |")
    lines.append(f"| GPU | {metadata.get('gpu_name', 'N/A')} |")
    lines.append(f"| GPU Memory | {metadata.get('gpu_memory_gb', 'N/A')} GB |")
    lines.append(f"| CUDA Version | {metadata.get('cuda_version', 'N/A')} |")
    lines.append(f"| PyTorch Version | {metadata.get('torch_version', 'N/A')} |")
    lines.append("")

    # Section 3: Data Loading Performance
    lines.append("## 3. Data Loading Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Samples measured | {phase1.get('n_samples', 'N/A')} |")
    lines.append(f"| Mean load time | {phase1.get('mean_time_s', 'N/A')}s |")
    lines.append(f"| Median load time | {phase1.get('median_time_s', 'N/A')}s |")
    lines.append(f"| P95 load time | {phase1.get('p95_time_s', 'N/A')}s |")
    lines.append(f"| Throughput | {phase1.get('samples_per_sec', 'N/A')} samples/sec |")
    lines.append(f"| DataLoader workers | {phase1.get('num_workers', 'N/A')} |")
    lines.append("")

    shape_dist = phase1.get("shape_distribution", {})
    if shape_dist:
        lines.append("### Shape Distribution")
        lines.append("")
        lines.append("| Shape | Count |")
        lines.append("|-------|-------|")
        for shape, count in list(shape_dist.items())[:15]:
            lines.append(f"| {shape} | {count} |")
        if len(shape_dist) > 15:
            lines.append(f"| ... | ({len(shape_dist) - 15} more) |")
        lines.append("")

    # Section 4: GPU Memory Profile
    lines.append("## 4. GPU Memory Profile")
    lines.append("")
    peak_mem = phase2.get("peak_memory", {})
    if peak_mem:
        lines.append("| Shape | Peak Allocated (MB) | Peak Reserved (MB) |")
        lines.append("|-------|--------------------:|-------------------:|")
        for shape, mem in sorted(peak_mem.items()):
            lines.append(
                f"| {shape} | {mem['peak_allocated_mb']:,.0f} | "
                f"{mem['peak_reserved_mb']:,.0f} |"
            )
        lines.append("")

    gpu_overall = phase2.get("overall", {})
    if gpu_overall:
        lines.append("### GPU Compute Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean step time | {gpu_overall.get('mean_ms', 'N/A')}ms |")
        lines.append(f"| Median step time | {gpu_overall.get('median_ms', 'N/A')}ms |")
        lines.append(f"| P95 step time | {gpu_overall.get('p95_ms', 'N/A')}ms |")
        lines.append(
            f"| Throughput | {gpu_overall.get('samples_per_sec', 'N/A')} samples/sec |"
        )
        lines.append("")

    # Section 5: Bottleneck Analysis
    lines.append("## 5. Bottleneck Analysis")
    lines.append("")
    lines.append(f"**Bottleneck: {bottleneck}**")
    lines.append("")
    lines.append(f"{bottleneck_detail}")
    lines.append("")
    lines.append("| Phase | Throughput (samples/sec) |")
    lines.append("|-------|------------------------:|")
    lines.append(f"| Data loading only | {data_throughput:.2f} |")
    lines.append(f"| GPU compute only | {compute_throughput:.2f} |")
    lines.append(f"| End-to-end | {e2e_throughput:.2f} |")
    lines.append("")

    # Section 6: Training Time Projections
    lines.append("## 6. Training Time Projections")
    lines.append("")
    lines.append(
        f"Projections assume {train_epochs} epochs, batch_size=1, "
        f"end-to-end step time of {step_time_s:.4f}s on benchmark GPU. "
        f"DDP efficiency degrades ~1%/node beyond 2 nodes (model is ~{model_summary['model_size_mb']:.1f} MB, "
        f"so gradient all-reduce is small; network overhead is dominated by synchronization)."
    )
    lines.append("")

    if benchmark_gpu:
        gpu_types = list(GPU_SPECS.keys())
    else:
        lines.append(
            f"> **Note:** Benchmark GPU ({metadata.get('gpu_name', 'unknown')}) "
            f"not recognized. Showing raw times without cross-GPU scaling. "
            f"Re-run on A100/H100/H200 for accurate projections."
        )
        lines.append("")
        gpu_types = [metadata.get("gpu_name", "benchmark GPU")]

    for ds_size in DATASET_SIZES:
        lines.append(f"### {ds_size:,} structures")
        lines.append("")

        # Header
        header = "| GPUs |"
        separator = "|-----:|"
        for gpu_type in gpu_types:
            header += f" {gpu_type} (1 epoch) | {gpu_type} ({train_epochs} epochs) |"
            separator += "---:|---:|"

        lines.append(header)
        lines.append(separator)

        for n_gpus in GPU_COUNTS:
            row = f"| {n_gpus} |"
            for gpu_type in gpu_types:
                if benchmark_gpu and gpu_type in GPU_SPECS:
                    scaling = compute_scaling_factor(benchmark_gpu, gpu_type)
                else:
                    scaling = 1.0

                epoch_time = estimate_epoch_time(ds_size, step_time_s, n_gpus, scaling)
                total_time = epoch_time * train_epochs
                row += f" {format_time(epoch_time)} | {format_time(total_time)} |"

            lines.append(row)

        lines.append("")

    # Section 7: Storage Requirements
    lines.append("## 7. Storage Requirements")
    lines.append("")

    # Use measured file sizes if available, otherwise fall back to estimate
    file_sizes = phase1.get("file_sizes", {})
    if file_sizes and file_sizes.get("mean_mb"):
        mean_chgcar_mb = file_sizes["mean_mb"]
        lines.append(
            f"Measured mean CHGCAR file size: {mean_chgcar_mb:.1f} MB "
            f"(sampled {file_sizes['n_files_sampled']} files, "
            f"range {file_sizes['min_mb']:.1f}-{file_sizes['max_mb']:.1f} MB)"
        )
    else:
        mean_chgcar_mb = 50  # conservative estimate
        lines.append(
            f"Estimated mean CHGCAR file size: ~{mean_chgcar_mb} MB "
            f"(no file size data available; re-run benchmark with data access)"
        )
    lines.append(
        f"\nPer structure storage (data + label): ~{mean_chgcar_mb * 2:.0f} MB"
    )
    lines.append("")
    lines.append("| Dataset Size | Storage (data + labels) |")
    lines.append("|-------------:|------------------------:|")
    for ds_size in DATASET_SIZES:
        storage_gb = (ds_size * mean_chgcar_mb * 2) / 1024
        lines.append(f"| {ds_size:,} | {storage_gb:,.0f} GB |")
    lines.append("")

    # Section 8: Recommended Cluster Configurations
    lines.append("## 8. Recommended Cluster Configurations")
    lines.append("")

    # For 6K and 100K, recommend configurations that keep epoch time reasonable
    target_epoch_hours = 1.0  # target: 1 hour per epoch
    for ds_size in [6000, 100000]:
        lines.append(f"### {ds_size:,} structures")
        lines.append("")
        lines.append(
            f"Target: ~{target_epoch_hours:.0f} hour per epoch "
            f"({train_epochs} epochs = ~{target_epoch_hours * train_epochs:.0f} "
            f"hours total)"
        )
        lines.append("")

        for gpu_type in list(GPU_SPECS.keys()) if benchmark_gpu else ["benchmark GPU"]:
            if benchmark_gpu and gpu_type in GPU_SPECS:
                scaling = compute_scaling_factor(benchmark_gpu, gpu_type)
            else:
                scaling = 1.0

            # Find minimum GPUs to hit target
            for n_gpus in [1, 2, 4, 8, 16, 32, 64]:
                epoch_time = estimate_epoch_time(ds_size, step_time_s, n_gpus, scaling)
                if epoch_time <= target_epoch_hours * 3600:
                    nodes = max(1, n_gpus // 4)
                    lines.append(
                        f"- **{gpu_type}**: {n_gpus} GPUs ({nodes} nodes x "
                        f"{min(n_gpus, 4)} GPUs/node) - "
                        f"{format_time(epoch_time)}/epoch, "
                        f"{format_time(epoch_time * train_epochs)} total"
                    )
                    break
            else:
                lines.append(
                    f"- **{gpu_type}**: >64 GPUs needed for "
                    f"<{target_epoch_hours}hr/epoch"
                )

        lines.append("")

    # Write report
    report = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"Report written to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ElectrAI benchmark results")
    parser.add_argument(
        "--results", type=str, required=True, help="Path to benchmark_results.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/compute_estimates.md",
        help="Output path for the report (default: docs/compute_estimates.md)",
    )
    args = parser.parse_args()

    with Path(args.results).open() as f:
        results = json.load(f)

    generate_report(results, args.output)


if __name__ == "__main__":
    main()
