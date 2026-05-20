"""Compute summary statistics and distribution plots from test metrics.

Usage:
    uv run python -m electrai.scripts.analyze.summarize \
        --metrics metrics.csv \
        --output-dir summary_output/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PERCENTILES = [75, 90, 95, 99]
_NMAE_THRESHOLDS = [0.01, 0.03, 0.05]


def summarize(metrics_path: str | Path, output_dir: str | Path | None = None) -> str:
    """Read metrics.csv and compute aggregate statistics.

    Returns a formatted summary string. If output_dir is provided,
    also writes summary.txt there.
    """
    df = pd.read_csv(metrics_path)
    nmae = df["nmae"]

    percentile_lines = "\n".join(
        f"  P{p}:    {np.percentile(nmae, p):.4%}" for p in _PERCENTILES
    )
    threshold_lines = "\n".join(
        f"  > {t:.0%}: {(nmae > t).sum()} ({(nmae > t).sum() / len(nmae):.1%} of samples)"
        for t in _NMAE_THRESHOLDS
    )

    sections = [
        f"""\
{"=" * 60}
Test Evaluation Summary
{"=" * 60}
Samples evaluated: {len(df)}

NMAE (Normalized Mean Absolute Error)
{"-" * 40}
  Mean:   {nmae.mean():.4%}
  Median: {nmae.median():.4%}
  Std:    {nmae.std():.4%}
  Min:    {nmae.min():.4%}
  Max:    {nmae.max():.4%}

{percentile_lines}

{threshold_lines}"""
    ]

    # --- Duration statistics ---
    if "avg_duration_ms" in df.columns:
        dur = df["avg_duration_ms"]
        sections.append(f"""\
Inference Timing
{"-" * 40}
  Mean per sample: {dur.mean():.1f} ms
  Total:           {dur.sum() / 1000:.1f} s""")

    # --- Peak density statistics ---
    if "max_pred" in df.columns and "max_target" in df.columns:
        ratio = df["max_pred"] / df["max_target"]
        saturated = (ratio < 0.8).sum()
        sections.append(f"""\
Peak Density (max_pred / max_target)
{"-" * 40}
  Mean ratio:   {ratio.mean():.4f}
  Median ratio: {ratio.median():.4f}
  Saturated (ratio < 0.8): {saturated} ({saturated / len(ratio):.1%})""")

    sections.append("=" * 60)

    summary_text = "\n\n".join(sections)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.txt").write_text(summary_text)

    return summary_text


def plot_distribution(metrics_path: str | Path, output_dir: str | Path) -> None:
    """Generate NMAE distribution plots (histogram + CDF)."""
    import matplotlib as mpl

    mpl.use("Agg")

    df = pd.read_csv(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nmae = df["nmae"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: NMAE histogram ---
    ax = axes[0]
    ax.hist(nmae * 100, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    mean_val = nmae.mean() * 100
    median_val = np.median(nmae) * 100
    p95_val = np.percentile(nmae, 95) * 100
    ax.axvline(mean_val, color="red", ls="--", lw=1.5, label=f"Mean: {mean_val:.2f}%")
    ax.axvline(
        median_val, color="orange", ls="--", lw=1.5, label=f"Median: {median_val:.2f}%"
    )
    ax.axvline(p95_val, color="darkred", ls=":", lw=1.5, label=f"P95: {p95_val:.2f}%")
    ax.set_xlabel("NMAE (%)")
    ax.set_ylabel("Count")
    ax.set_title("NMAE Distribution")
    ax.legend(fontsize=9)

    # --- Panel 2: Empirical CDF ---
    ax = axes[1]
    sorted_nmae = np.sort(nmae) * 100
    cdf = np.arange(1, len(sorted_nmae) + 1) / len(sorted_nmae)
    ax.plot(sorted_nmae, cdf, color="steelblue", lw=2)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(0.95, color="darkred", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("NMAE (%)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Empirical CDF")

    fig.tight_layout()
    fig.savefig(output_dir / "nmae_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "nmae_distribution.png")


def log_to_wandb(metrics_path: str | Path, output_dir: str | Path) -> None:
    """Log distribution plot and per-sample metrics to an active W&B run."""
    import wandb

    if wandb.run is None:
        logger.warning("No active W&B run, skipping W&B logging")
        return

    df = pd.read_csv(metrics_path)
    output_dir = Path(output_dir)

    # Log the distribution PNG as an image
    png_path = output_dir / "nmae_distribution.png"
    if png_path.exists():
        wandb.log({"test/nmae_distribution": wandb.Image(str(png_path))})

    # Log per-sample NMAE as a table for interactive exploration
    table_cols = ["index", "nmae"]
    if "max_pred" in df.columns:
        table_cols.extend(["max_pred", "max_target"])
    table = wandb.Table(dataframe=df[table_cols])
    wandb.log({"test/metrics": table})

    # Log a native W&B histogram for the overview panel
    wandb.log({"test/nmae_histogram": wandb.Histogram(df["nmae"].to_numpy())})

    # Log scalar summary stats
    nmae = df["nmae"]
    wandb.log(
        {
            "test/nmae_mean": nmae.mean(),
            "test/nmae_median": nmae.median(),
            "test/nmae_p95": np.percentile(nmae, 95),
            "test/nmae_p99": np.percentile(nmae, 99),
            "test/nmae_max": nmae.max(),
        }
    )

    logger.info("Logged test metrics to W&B")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize test metrics")
    parser.add_argument(
        "--metrics", type=Path, required=True, help="Path to metrics.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for outputs"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    logger.info("\n%s", summarize(args.metrics, output_dir=args.output_dir))
    plot_distribution(args.metrics, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
