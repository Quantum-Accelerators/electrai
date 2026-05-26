"""Analyze output saturation / dynamic range compression in model predictions.

Two modes of operation:

1. **From metrics CSV** (Phase 1 output): Reads per-structure max/p99/mean
   columns and produces scatter plots, ratio distributions, and correlation
   analyses across the full test set.

2. **From saved .npy predictions** (Phase 3b): For structures with saved
   prediction arrays and label CHGCARs, produces voxel-level pred-vs-target
   scatter plots that directly show whether the prediction flattens at high
   target densities.

Usage:
    # Mode 1: metrics CSV (after re-running inference with saturation columns)
    uv run python -m electrai.scripts.analyze.analyze_saturation \
        --metrics metrics.csv --output-dir saturation_analysis/

    # Mode 2: voxel scatter from saved predictions
    uv run python -m electrai.scripts.analyze.analyze_saturation \
        --viz-dir ~/viz_worst_output/ --output-dir saturation_analysis/

    # Both modes together
    uv run python -m electrai.scripts.analyze.analyze_saturation \
        --metrics metrics.csv --viz-dir ~/viz_worst_output/ \
        --output-dir saturation_analysis/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode 1: Metrics CSV analysis
# ---------------------------------------------------------------------------


def _load_metrics(path: Path) -> pd.DataFrame:
    """Load metrics CSV and compute derived saturation columns."""
    df = pd.read_csv(path)

    required = {"max_pred", "max_target"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"Metrics CSV missing saturation columns: {missing}. "
            "Re-run inference with the updated test pipeline."
        )

    df["max_ratio"] = df["max_pred"] / df["max_target"]
    if "p99_pred" in df.columns and "p99_target" in df.columns:
        df["p99_ratio"] = df["p99_pred"] / df["p99_target"]
    if "mean_pred" in df.columns and "mean_target" in df.columns:
        df["pred_contrast"] = df["max_pred"] / df["mean_pred"]
        df["target_contrast"] = df["max_target"] / df["mean_target"]

    return df


def _plot_max_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: max_pred vs max_target, colored by NMAE."""
    fig, ax = plt.subplots(figsize=(8, 8))

    nmae_pct = df["nmae"] * 100 if df["nmae"].max() < 1 else df["nmae"]
    sc = ax.scatter(
        df["max_target"],
        df["max_pred"],
        c=nmae_pct,
        cmap="YlOrRd",
        s=4,
        alpha=0.5,
        vmin=0,
        vmax=min(nmae_pct.quantile(0.95), 15),
    )
    lim = max(df["max_target"].max(), df["max_pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("max(target) [e⁻/ų]")
    ax.set_ylabel("max(pred) [e⁻/ų]")
    ax.set_title("Output Saturation: max(pred) vs max(target)")
    ax.set_aspect("equal")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="NMAE (%)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "max_pred_vs_max_target.png", dpi=150)
    plt.close(fig)


def _plot_ratio_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of max_pred / max_target ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ratio = df["max_ratio"].clip(0, 2)
    axes[0].hist(ratio, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
    axes[0].axvline(1.0, color="red", linestyle="--", label="ratio = 1.0")
    axes[0].axvline(
        ratio.median(),
        color="blue",
        linestyle=":",
        label=f"median = {ratio.median():.3f}",
    )
    axes[0].set_xlabel("max(pred) / max(target)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Max Density Ratio Distribution")
    axes[0].legend()

    if "p99_ratio" in df.columns:
        p99_ratio = df["p99_ratio"].clip(0, 2)
        axes[1].hist(p99_ratio, bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
        axes[1].axvline(1.0, color="red", linestyle="--", label="ratio = 1.0")
        axes[1].axvline(
            p99_ratio.median(),
            color="blue",
            linestyle=":",
            label=f"median = {p99_ratio.median():.3f}",
        )
        axes[1].set_xlabel("P99(pred) / P99(target)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("P99 Density Ratio Distribution")
        axes[1].legend()
    else:
        axes[1].text(
            0.5, 0.5, "P99 data not available", transform=axes[1].transAxes, ha="center"
        )
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_dir / "ratio_distribution.png", dpi=150)
    plt.close(fig)


def _plot_ratio_vs_peak_density(df: pd.DataFrame, out_dir: Path) -> None:
    """Binned ratio vs max_target — does compression worsen with peak density?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = 30
    bins = np.linspace(0, df["max_target"].quantile(0.99), n_bins + 1)
    df_cut = df.copy()
    df_cut["target_bin"] = pd.cut(df_cut["max_target"], bins=bins)
    grouped = df_cut.groupby("target_bin", observed=True)["max_ratio"]
    medians = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    counts = grouped.count()

    bin_centers = [(iv.left + iv.right) / 2 for iv in medians.index]

    ax.fill_between(bin_centers, q25.values, q75.values, alpha=0.3, label="IQR")
    ax.plot(bin_centers, medians.values, "b-o", markersize=4, label="Median ratio")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Perfect (1.0)")
    ax.set_xlabel("max(target) [e⁻/ų]")
    ax.set_ylabel("max(pred) / max(target)")
    ax.set_title("Saturation vs Peak Target Density")
    ax.legend()

    # Secondary axis: sample count per bin
    ax2 = ax.twinx()
    ax2.bar(
        bin_centers,
        counts.values,
        width=(bins[1] - bins[0]) * 0.8,
        alpha=0.1,
        color="gray",
    )
    ax2.set_ylabel("Structures per bin", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    fig.savefig(out_dir / "ratio_vs_peak_density.png", dpi=150)
    plt.close(fig)


def _plot_contrast_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: pred contrast ratio vs target contrast ratio."""
    if "pred_contrast" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    nmae_pct = df["nmae"] * 100 if df["nmae"].max() < 1 else df["nmae"]
    sc = ax.scatter(
        df["target_contrast"],
        df["pred_contrast"],
        c=nmae_pct,
        cmap="YlOrRd",
        s=4,
        alpha=0.5,
        vmin=0,
        vmax=min(nmae_pct.quantile(0.95), 15),
    )
    lim = max(df["target_contrast"].max(), df["pred_contrast"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, min(lim, df["target_contrast"].quantile(0.99) * 1.2))
    ax.set_ylim(0, min(lim, df["pred_contrast"].quantile(0.99) * 1.2))
    ax.set_xlabel("target contrast (max / mean)")
    ax.set_ylabel("pred contrast (max / mean)")
    ax.set_title("Spatial Dynamic Range: Prediction vs Target")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="NMAE (%)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "contrast_comparison.png", dpi=150)
    plt.close(fig)


def _plot_saturation_vs_nmae(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: max_ratio vs NMAE — does saturation predict error?"""
    fig, ax = plt.subplots(figsize=(8, 6))
    nmae_pct = df["nmae"] * 100 if df["nmae"].max() < 1 else df["nmae"]
    ax.scatter(df["max_ratio"].clip(0, 2), nmae_pct, s=4, alpha=0.3)
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("max(pred) / max(target)")
    ax.set_ylabel("NMAE (%)")
    ax.set_title("Saturation Ratio vs NMAE")
    fig.tight_layout()
    fig.savefig(out_dir / "saturation_vs_nmae.png", dpi=150)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a text summary of saturation statistics."""
    lines = ["# Output Saturation Summary\n"]

    lines.append("## Max Density Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Structures analyzed | {len(df)} |")
    lines.append(f"| Median max(target) | {df['max_target'].median():.2f} e/ų |")
    lines.append(f"| Median max(pred) | {df['max_pred'].median():.2f} e/ų |")
    lines.append(f"| Median max ratio | {df['max_ratio'].median():.4f} |")
    lines.append(f"| Mean max ratio | {df['max_ratio'].mean():.4f} |")
    lines.append(f"| Std max ratio | {df['max_ratio'].std():.4f} |")
    lines.append(f"| Max max(target) | {df['max_target'].max():.2f} e/ų |")
    lines.append(f"| Max max(pred) | {df['max_pred'].max():.2f} e/ų |")
    lines.append("")

    # Binned summary: how does ratio change with peak density?
    lines.append("## Ratio by Peak Density Bin\n")
    lines.append("| Target max bin | Count | Median ratio | Mean ratio |")
    lines.append("|---------------|-------|-------------|-----------|")
    bins = [0, 1, 2, 5, 10, 20, 50, np.inf]
    labels = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50+"]
    df_tmp = df.copy()
    df_tmp["bin"] = pd.cut(df_tmp["max_target"], bins=bins, labels=labels)
    for label in labels:
        subset = df_tmp[df_tmp["bin"] == label]
        if len(subset) > 0:
            lines.append(
                f"| {label} e/ų | {len(subset)} | "
                f"{subset['max_ratio'].median():.4f} | "
                f"{subset['max_ratio'].mean():.4f} |"
            )
    lines.append("")

    # Structures with severe saturation
    severe = df[df["max_ratio"] < 0.5].sort_values("max_ratio")
    if len(severe) > 0:
        lines.append(
            f"## Structures with Severe Saturation (ratio < 0.5): {len(severe)}\n"
        )
        lines.append("| Index | max(target) | max(pred) | Ratio | NMAE |")
        lines.append("|-------|------------|----------|-------|------|")
        nmae_col = "nmae"
        for _, row in severe.head(20).iterrows():
            nmae_val = row[nmae_col] * 100 if row[nmae_col] < 1 else row[nmae_col]
            lines.append(
                f"| {row['index']} | {row['max_target']:.2f} | "
                f"{row['max_pred']:.2f} | {row['max_ratio']:.4f} | "
                f"{nmae_val:.2f}% |"
            )
    lines.append("")

    (out_dir / "saturation_summary.md").write_text("\n".join(lines))
    logger.info("Wrote saturation_summary.md")


def analyze_metrics(metrics_path: Path, out_dir: Path) -> None:
    """Run all metrics-based saturation analyses."""
    import matplotlib as mpl

    mpl.use("Agg")

    df = _load_metrics(metrics_path)
    logger.info("Loaded %d structures from %s", len(df), metrics_path)

    _plot_max_scatter(df, out_dir)
    _plot_ratio_distribution(df, out_dir)
    _plot_ratio_vs_peak_density(df, out_dir)
    _plot_contrast_comparison(df, out_dir)
    _plot_saturation_vs_nmae(df, out_dir)
    _write_summary(df, out_dir)

    logger.info("Metrics-based saturation analysis complete.")


# ---------------------------------------------------------------------------
# Mode 2: Voxel-level scatter from saved .npy predictions
# ---------------------------------------------------------------------------


def _load_pair(chgcar_path: Path, pred_path: Path) -> dict | None:
    """Load a label CHGCAR and predicted .npy, return flattened arrays."""
    from pymatgen.io.vasp.outputs import Chgcar

    chgcar = Chgcar.from_file(str(chgcar_path))
    volume = chgcar.structure.lattice.volume
    label = chgcar.data["total"] / volume  # e⁻/ų

    pred = np.load(pred_path)
    grid = label.shape
    if pred.shape != grid:
        if pred.shape == (1, *grid):
            pred = pred[0]
        elif pred.shape == (*grid, 1):
            pred = pred[..., 0]
        else:
            logger.warning(
                "Shape mismatch: label=%s pred=%s for %s",
                grid,
                pred.shape,
                pred_path.name,
            )
            return None

    task_id = chgcar_path.stem
    formula = chgcar.structure.composition.reduced_formula

    return {
        "task_id": task_id,
        "formula": formula,
        "label": label.ravel(),
        "pred": pred.ravel(),
        "max_target": label.max(),
        "max_pred": pred.max(),
    }


def _plot_voxel_scatter(data: dict, out_dir: Path) -> None:
    """Pred vs target for all voxels of a single structure."""
    label = data["label"]
    pred = data["pred"]
    tid = data["task_id"]
    formula = data["formula"]

    # Subsample if too many voxels (128^3 = 2M)
    n = len(label)
    max_points = 200_000
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        label_s = label[idx]
        pred_s = pred[idx]
        sample_note = f" ({max_points // 1000}K of {n // 1000}K voxels)"
    else:
        label_s = label
        pred_s = pred
        sample_note = ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: full range scatter
    ax = axes[0]
    ax.scatter(label_s, pred_s, s=0.5, alpha=0.05, rasterized=True)
    lim = max(label.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r-", linewidth=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Target density [e⁻/ų]")
    ax.set_ylabel("Predicted density [e⁻/ų]")
    ax.set_title("Full range")
    ax.set_aspect("equal")
    ax.legend()

    # Add binned median line
    n_bins = 50
    bin_edges = np.linspace(0, label.max() * 1.01, n_bins + 1)
    bin_idx = np.digitize(label, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_medians = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_medians[b] = np.median(pred[mask])
            bin_counts[b] = mask.sum()
    valid = bin_counts > 10
    ax.plot(
        bin_centers[valid], bin_medians[valid], "g-", linewidth=2, label="Binned median"
    )
    ax.legend()

    # Right: zoom on high-density tail (top 5% of target values)
    ax = axes[1]
    threshold = np.percentile(label, 95)
    high_mask = label >= threshold
    if high_mask.sum() > max_points:
        rng = np.random.default_rng(42)
        hi_idx = rng.choice(np.where(high_mask)[0], max_points, replace=False)
    else:
        hi_idx = np.where(high_mask)[0]
    ax.scatter(label[hi_idx], pred[hi_idx], s=1, alpha=0.1, rasterized=True)
    hi_lim = max(label[high_mask].max(), pred[high_mask].max()) * 1.05
    ax.plot([threshold, hi_lim], [threshold, hi_lim], "r-", linewidth=1, label="y = x")
    ax.set_xlim(threshold, hi_lim)
    ax.set_ylim(0, hi_lim)
    ax.set_xlabel("Target density [e⁻/ų]")
    ax.set_ylabel("Predicted density [e⁻/ų]")
    ax.set_title("High-density tail (top 5%)")
    ax.legend()

    # Binned median for high-density region
    hi_bin_edges = np.linspace(threshold, label.max() * 1.01, 30)
    hi_bin_idx = np.digitize(label[high_mask], hi_bin_edges) - 1
    hi_bin_idx = np.clip(hi_bin_idx, 0, len(hi_bin_edges) - 2)
    hi_bin_centers = (hi_bin_edges[:-1] + hi_bin_edges[1:]) / 2
    hi_bin_medians = np.zeros(len(hi_bin_edges) - 1)
    hi_bin_counts = np.zeros(len(hi_bin_edges) - 1)
    pred_high = pred[high_mask]
    for b in range(len(hi_bin_edges) - 1):
        mask = hi_bin_idx == b
        if mask.sum() > 0:
            hi_bin_medians[b] = np.median(pred_high[mask])
            hi_bin_counts[b] = mask.sum()
    valid = hi_bin_counts > 5
    ax.plot(
        hi_bin_centers[valid],
        hi_bin_medians[valid],
        "g-",
        linewidth=2,
        label="Binned median",
    )
    ax.legend()

    max_ratio = data["max_pred"] / data["max_target"]
    fig.suptitle(
        f"{tid} ({formula}) — Voxel Scatter{sample_note}\n"
        f"max(target)={data['max_target']:.2f}, max(pred)={data['max_pred']:.2f}, "
        f"ratio={max_ratio:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{tid}_voxel_scatter.png", dpi=150)
    plt.close(fig)


def _plot_combined_saturation_curve(all_data: list[dict], out_dir: Path) -> None:
    """Overlay binned median pred-vs-target curves for all structures."""
    fig, ax = plt.subplots(figsize=(10, 8))

    global_max = max(d["max_target"] for d in all_data)

    for data in all_data:
        label = data["label"]
        pred = data["pred"]
        n_bins = 50
        bin_edges = np.linspace(0, global_max * 1.01, n_bins + 1)
        bin_idx = np.digitize(label, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_medians = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 10:
                bin_medians[b] = np.median(pred[mask])
        valid = ~np.isnan(bin_medians)
        ratio = data["max_pred"] / data["max_target"]
        ax.plot(
            bin_centers[valid],
            bin_medians[valid],
            linewidth=1.5,
            label=f"{data['task_id']} ({data['formula']}) r={ratio:.2f}",
        )

    ax.plot([0, global_max], [0, global_max], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Target density [e⁻/ų]")
    ax.set_ylabel("Median predicted density [e⁻/ų]")
    ax.set_title("Saturation Curves: Binned Median Prediction vs Target")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(0, global_max * 1.05)
    ax.set_ylim(0, global_max * 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_saturation_curves.png", dpi=150)
    plt.close(fig)


def analyze_voxels(viz_dir: Path, out_dir: Path) -> None:
    """Run voxel-level saturation analysis from saved predictions."""
    import matplotlib as mpl

    mpl.use("Agg")

    pred_dir = viz_dir / "predictions"
    pred_files = sorted(pred_dir.glob("rank_*_*.npy"))
    if not pred_files:
        logger.error("No prediction .npy files found in %s", pred_dir)
        return

    all_data = []
    for pred_path in pred_files:
        parts = pred_path.stem.split("_", 2)
        task_id = parts[2] if len(parts) >= 3 else pred_path.stem

        chgcar_path = viz_dir / f"{task_id}.CHGCAR"
        if not chgcar_path.exists():
            logger.warning("No CHGCAR found for %s, skipping", task_id)
            continue

        logger.info("Loading %s ...", task_id)
        data = _load_pair(chgcar_path, pred_path)
        if data is None:
            continue
        all_data.append(data)
        _plot_voxel_scatter(data, out_dir)

    if len(all_data) >= 2:
        _plot_combined_saturation_curve(all_data, out_dir)

    # Summary table
    if all_data:
        lines = ["# Voxel-Level Saturation Summary\n"]
        lines.append("| Structure | Formula | max(target) | max(pred) | Ratio |")
        lines.append("|-----------|---------|------------|----------|-------|")
        for d in sorted(all_data, key=lambda x: x["max_pred"] / x["max_target"]):
            ratio = d["max_pred"] / d["max_target"]
            lines.append(
                f"| {d['task_id']} | {d['formula']} | "
                f"{d['max_target']:.2f} | {d['max_pred']:.2f} | {ratio:.3f} |"
            )
        (out_dir / "voxel_saturation_summary.md").write_text("\n".join(lines))
        logger.info("Wrote voxel_saturation_summary.md")

    logger.info(
        "Voxel-level saturation analysis complete (%d structures).", len(all_data)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze output saturation / dynamic range compression."
    )
    parser.add_argument(
        "--metrics", default=None, help="Path to metrics.csv with saturation columns"
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory with CHGCARs and predictions/ (for voxel scatter)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for plots"
    )
    args = parser.parse_args(argv)

    if args.metrics is None and args.viz_dir is None:
        parser.error("Provide at least one of --metrics or --viz-dir")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        analyze_metrics(Path(args.metrics), out_dir)

    if args.viz_dir:
        analyze_voxels(Path(args.viz_dir), out_dir)

    logger.info("All outputs written to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
