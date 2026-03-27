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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
logger = logging.getLogger(__name__)

_PERCENTILES = [50, 75, 90, 95, 99]
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

    # --- Loss statistics (if loss column exists and differs from NMAE) ---
    if "loss" in df.columns:
        loss = df["loss"]
        loss_differs = not np.allclose(loss.to_numpy(), nmae.to_numpy(), atol=1e-6)
        if loss_differs:
            loss_pct = "\n".join(
                f"  P{p}:    {np.percentile(loss, p):.6f}" for p in _PERCENTILES
            )
            sections.append(f"""\
Training Loss (differs from NMAE)
{"-" * 40}
  Mean:   {loss.mean():.6f}
  Median: {loss.median():.6f}
  Std:    {loss.std():.6f}
  Min:    {loss.min():.6f}
  Max:    {loss.max():.6f}

{loss_pct}""")
        else:
            sections.append("Training Loss: identical to NMAE (loss_fn = normmae)")

    # --- Duration statistics ---
    if "duration_ms" in df.columns:
        dur = df["duration_ms"]
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
    """Generate NMAE distribution plots (histogram + CDF).

    If the loss column differs from NMAE, adds a separate loss panel.
    """
    df = pd.read_csv(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nmae = df["nmae"].to_numpy()

    has_loss = "loss" in df.columns
    loss_differs = has_loss and not np.allclose(df["loss"].to_numpy(), nmae, atol=1e-6)

    n_panels = 3 if loss_differs else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

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

    # --- Panel 3 (optional): Loss distribution ---
    if loss_differs:
        ax = axes[2]
        loss = df["loss"].to_numpy()
        ax.hist(loss, bins=50, color="coral", edgecolor="white", alpha=0.8)
        ax.axvline(
            loss.mean(), color="red", ls="--", lw=1.5, label=f"Mean: {loss.mean():.4f}"
        )
        ax.axvline(
            np.median(loss),
            color="orange",
            ls="--",
            lw=1.5,
            label=f"Median: {np.median(loss):.4f}",
        )
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.set_title("Training Loss Distribution")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "nmae_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "nmae_distribution.png")


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
