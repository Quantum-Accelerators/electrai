"""Analyze NMAE distribution tails and correlate with structural metadata.

Phase 2: Joins metrics.csv with metadata.csv and produces statistical
summaries, correlation analyses, and publication-quality plots.

Usage:
    uv run python -m electrai.scripts.analyze.analyze_tail \
        --metrics metrics.csv \
        --metadata metadata.csv \
        --output-dir analysis_output/ \
        [--tail-threshold 95p] \
        [--split-file split.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# NMAE stored as fraction in metrics.csv (e.g. 0.023 = 2.3%)
NMAE_TO_PERCENT = 100.0

# Periodic table layout for element heatmap (row, col) positions
_PT_LAYOUT: dict[str, tuple[int, int]] = {
    "H": (0, 0),
    "He": (0, 17),
    "Li": (1, 0),
    "Be": (1, 1),
    "B": (1, 12),
    "C": (1, 13),
    "N": (1, 14),
    "O": (1, 15),
    "F": (1, 16),
    "Ne": (1, 17),
    "Na": (2, 0),
    "Mg": (2, 1),
    "Al": (2, 12),
    "Si": (2, 13),
    "P": (2, 14),
    "S": (2, 15),
    "Cl": (2, 16),
    "Ar": (2, 17),
    "K": (3, 0),
    "Ca": (3, 1),
    "Sc": (3, 2),
    "Ti": (3, 3),
    "V": (3, 4),
    "Cr": (3, 5),
    "Mn": (3, 6),
    "Fe": (3, 7),
    "Co": (3, 8),
    "Ni": (3, 9),
    "Cu": (3, 10),
    "Zn": (3, 11),
    "Ga": (3, 12),
    "Ge": (3, 13),
    "As": (3, 14),
    "Se": (3, 15),
    "Br": (3, 16),
    "Kr": (3, 17),
    "Rb": (4, 0),
    "Sr": (4, 1),
    "Y": (4, 2),
    "Zr": (4, 3),
    "Nb": (4, 4),
    "Mo": (4, 5),
    "Tc": (4, 6),
    "Ru": (4, 7),
    "Rh": (4, 8),
    "Pd": (4, 9),
    "Ag": (4, 10),
    "Cd": (4, 11),
    "In": (4, 12),
    "Sn": (4, 13),
    "Sb": (4, 14),
    "Te": (4, 15),
    "I": (4, 16),
    "Xe": (4, 17),
    "Cs": (5, 0),
    "Ba": (5, 1),
    "La": (7, 2),
    "Ce": (7, 3),
    "Pr": (7, 4),
    "Nd": (7, 5),
    "Pm": (7, 6),
    "Sm": (7, 7),
    "Eu": (7, 8),
    "Gd": (7, 9),
    "Tb": (7, 10),
    "Dy": (7, 11),
    "Ho": (7, 12),
    "Er": (7, 13),
    "Tm": (7, 14),
    "Yb": (7, 15),
    "Lu": (5, 2),
    "Hf": (5, 3),
    "Ta": (5, 4),
    "W": (5, 5),
    "Re": (5, 6),
    "Os": (5, 7),
    "Ir": (5, 8),
    "Pt": (5, 9),
    "Au": (5, 10),
    "Tl": (5, 12),
    "Pb": (5, 13),
    "Bi": (5, 14),
    "Po": (5, 15),
    "At": (5, 16),
    "Rn": (5, 17),
    "Fr": (6, 0),
    "Ra": (6, 1),
    "Ac": (8, 2),
    "Th": (8, 3),
    "Pa": (8, 4),
    "U": (8, 5),
    "Np": (8, 6),
    "Pu": (8, 7),
    "Hg": (5, 11),
}


def _parse_threshold(s: str, nmae_series: pd.Series) -> float:
    """Parse tail threshold: '95p' -> 95th percentile, '0.05' -> absolute value."""
    if s.endswith("p"):
        pct = float(s[:-1])
        return float(np.percentile(nmae_series, pct))
    return float(s)


def _load_and_join(
    metrics_path: Path, metadata_path: Path, split_path: Path | None
) -> pd.DataFrame:
    """Load metrics and metadata CSVs, join on index == task_id."""
    metrics = pd.read_csv(metrics_path)
    metadata = pd.read_csv(metadata_path)

    # Ensure join keys are strings
    metrics["index"] = metrics["index"].astype(str)
    metadata["task_id"] = metadata["task_id"].astype(str)

    df = metrics.merge(metadata, left_on="index", right_on="task_id", how="left")

    # Derived columns
    if "num_atoms" in df.columns and "total_electrons" in df.columns:
        df["electrons_per_atom"] = df["total_electrons"] / df["num_atoms"]
    if "num_atoms" in df.columns and "volume" in df.columns:
        df["atoms_per_volume"] = df["num_atoms"] / df["volume"]

    # Lattice shape metrics
    if all(c in df.columns for c in ["a", "b", "c"]):
        abc = df[["a", "b", "c"]]
        df["aspect_ratio"] = abc.max(axis=1) / abc.min(axis=1)
        df["cell_isotropy"] = (abc.prod(axis=1) ** (1 / 3)) / abc.max(axis=1)

    # Voxel resolution anisotropy
    if all(c in df.columns for c in ["a", "b", "c", "grid_x", "grid_y", "grid_z"]):
        df["voxel_edge_a"] = df["a"] / df["grid_x"]
        df["voxel_edge_b"] = df["b"] / df["grid_y"]
        df["voxel_edge_c"] = df["c"] / df["grid_z"]
        vedges = df[["voxel_edge_a", "voxel_edge_b", "voxel_edge_c"]]
        df["voxel_anisotropy"] = vedges.max(axis=1) / vedges.min(axis=1)
        df["max_voxel_edge_ax"] = vedges.max(axis=1)
        df["min_voxel_edge_ax"] = vedges.min(axis=1)

    # Grid shape
    if all(c in df.columns for c in ["grid_x", "grid_y", "grid_z"]):
        grids = df[["grid_x", "grid_y", "grid_z"]]
        df["grid_aspect_ratio"] = grids.max(axis=1) / grids.min(axis=1)

    # Annotate split if provided
    if split_path and split_path.exists():
        with split_path.open() as f:
            splits = json.load(f)
        split_map = {}
        for split_name, indices in splits.items():
            for idx in indices:
                split_map[str(idx)] = split_name
        df["split"] = df["index"].map(split_map).fillna("unknown")

    return df


def _summary_stats(df: pd.DataFrame, threshold: float) -> str:
    """Compute and return summary statistics text."""
    nmae = df["nmae"]
    nmae_pct = nmae * NMAE_TO_PERCENT
    lines = [
        "=== NMAE Summary Statistics ===",
        f"Count:    {len(nmae)}",
        f"Mean:     {nmae_pct.mean():.4f}%",
        f"Median:   {nmae_pct.median():.4f}%",
        f"Std:      {nmae_pct.std():.4f}%",
        f"Min:      {nmae_pct.min():.4f}%",
        f"Max:      {nmae_pct.max():.4f}%",
        "",
        "Percentile breakdown:",
    ]
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(nmae_pct, p)
        lines.append(f"  P{p}:   {val:.4f}%")
    lines.append("")
    lines.append("Count above thresholds:")
    for t in [1.0, 3.0, 5.0]:
        count = (nmae_pct > t).sum()
        frac = count / len(nmae_pct) * 100
        lines.append(f"  >{t}%:  {count} ({frac:.1f}%)")
    lines.append("")
    lines.append(f"Tail threshold: {threshold * NMAE_TO_PERCENT:.4f}%")
    n_tail = (nmae >= threshold).sum()
    lines.append(f"Structures in tail: {n_tail} ({n_tail / len(nmae) * 100:.1f}%)")
    return "\n".join(lines)


def _element_analysis(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-element NMAE stats and tail enrichment ratios."""
    rows_with_elements = df.dropna(subset=["elements"])
    exploded = rows_with_elements.assign(
        element=rows_with_elements["elements"].str.split(";")
    ).explode("element")

    elem_stats = (
        exploded.groupby("element")["nmae"]
        .agg(["mean", "median", "count", lambda x: np.percentile(x, 95)])
        .rename(columns={"<lambda_0>": "p95"})
        .sort_values("mean", ascending=False)
    )
    elem_stats.columns = ["mean_nmae", "median_nmae", "count", "p95_nmae"]

    # Enrichment: fraction of element in tail / overall tail fraction
    tail_frac = (df["nmae"] >= threshold).mean()
    if tail_frac > 0:
        tail_mask = exploded["nmae"] >= threshold
        elem_tail_frac = tail_mask.groupby(exploded["element"]).mean()
        elem_stats["enrichment"] = elem_tail_frac / tail_frac
    else:
        elem_stats["enrichment"] = 0.0

    return elem_stats, exploded


def _correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations of NMAE vs numeric features."""
    features = [
        "num_atoms",
        "volume",
        "mean_voxel_edge",
        "ngridpts",
        "total_electrons",
        "num_elements",
        "electrons_per_atom",
        "aspect_ratio",
        "voxel_anisotropy",
        "max_voxel_edge_ax",
        "grid_aspect_ratio",
    ]
    available = [f for f in features if f in df.columns]
    rows = []
    for feat in available:
        valid = df[["nmae", feat]].dropna()
        if len(valid) < 3:
            continue
        sp_r, sp_p = stats.spearmanr(valid["nmae"], valid[feat])
        pe_r, pe_p = stats.pearsonr(valid["nmae"], valid[feat])
        rows.append(
            {
                "feature": feat,
                "spearman_r": sp_r,
                "spearman_p": sp_p,
                "pearson_r": pe_r,
                "pearson_p": pe_p,
            }
        )
    return pd.DataFrame(rows)


def _composition_families(df: pd.DataFrame) -> pd.DataFrame:
    """Top formulas by mean NMAE (min 3 samples)."""
    valid = df.dropna(subset=["formula"])
    grouped = valid.groupby("formula")["nmae"].agg(["mean", "median", "count", "std"])
    grouped = grouped[grouped["count"] >= 3].sort_values("mean", ascending=False)
    return grouped.head(50)


# --- Plotting functions ---


def _plot_distribution(df: pd.DataFrame, threshold: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    nmae_pct = df["nmae"] * NMAE_TO_PERCENT
    ax.hist(nmae_pct, bins=100, edgecolor="black", linewidth=0.3, alpha=0.7)
    thresh_pct = threshold * NMAE_TO_PERCENT
    ax.axvline(
        thresh_pct,
        color="red",
        linestyle="--",
        label=f"Tail threshold ({thresh_pct:.2f}%)",
    )
    ax.set_xlabel("NMAE (%)")
    ax.set_ylabel("Count")
    ax.set_title("NMAE Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "nmae_distribution.png", dpi=150)
    plt.close(fig)


def _plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    fname: str,
    out_dir: Path,
    color_col: str | None = None,
    xlabel: str | None = None,
    log_x: bool = False,
) -> None:
    valid = df.dropna(subset=["nmae", x_col])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    nmae_pct = valid["nmae"] * NMAE_TO_PERCENT

    if color_col and color_col in valid.columns:
        for label, group in valid.groupby(color_col):
            ax.scatter(
                group[x_col],
                group["nmae"] * NMAE_TO_PERCENT,
                label=str(label),
                alpha=0.4,
                s=10,
            )
        ax.legend(title=color_col)
    else:
        ax.scatter(valid[x_col], nmae_pct, alpha=0.4, s=10)

    if log_x:
        ax.set_xscale("log")

    sp_r, _ = stats.spearmanr(valid[x_col], valid["nmae"])
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel("NMAE (%)")
    ax.set_title(f"NMAE vs {xlabel or x_col} (Spearman r={sp_r:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def _plot_element_heatmap(elem_stats: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    max_row = max(r for r, _ in _PT_LAYOUT.values()) + 1
    max_col = max(c for _, c in _PT_LAYOUT.values()) + 1
    grid = np.full((max_row, max_col), np.nan)

    for elem, (r, c) in _PT_LAYOUT.items():
        if elem in elem_stats.index:
            grid[r, c] = elem_stats.loc[elem, "mean_nmae"] * NMAE_TO_PERCENT

    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    for elem, (r, c) in _PT_LAYOUT.items():
        if elem in elem_stats.index:
            val = elem_stats.loc[elem, "mean_nmae"] * NMAE_TO_PERCENT
            ax.text(c, r, f"{elem}\n{val:.2f}%", ha="center", va="center", fontsize=6)
        else:
            ax.text(c, r, elem, ha="center", va="center", fontsize=6, color="gray")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Mean NMAE by Element (Periodic Table)")
    fig.colorbar(im, ax=ax, label="Mean NMAE (%)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "element_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_enrichment(elem_stats: pd.DataFrame, out_dir: Path) -> None:
    if "enrichment" not in elem_stats.columns:
        return
    top = elem_stats.dropna(subset=["enrichment"]).nlargest(30, "enrichment")
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top["enrichment"][::-1])
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="baseline (1.0)")
    ax.set_xlabel("Tail Enrichment Ratio")
    ax.set_title("Element Tail Enrichment (top 30)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "element_enrichment.png", dpi=150)
    plt.close(fig)


def _plot_correlation_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    features = [
        "nmae",
        "num_atoms",
        "volume",
        "mean_voxel_edge",
        "ngridpts",
        "total_electrons",
        "num_elements",
        "electrons_per_atom",
        "aspect_ratio",
        "voxel_anisotropy",
        "max_voxel_edge_ax",
        "grid_aspect_ratio",
    ]
    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        return
    corr = df[available].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(available, fontsize=8)
    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(
                j,
                i,
                f"{corr.to_numpy()[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, label="Spearman r", shrink=0.8)
    ax.set_title("Correlation Matrix (Spearman)")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def _plot_composition_families(comp_df: pd.DataFrame, out_dir: Path) -> None:
    top20 = comp_df.head(20)
    if top20.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    nmae_pct = top20["mean"] * NMAE_TO_PERCENT
    ax.barh(top20.index[::-1], nmae_pct[::-1])
    ax.set_xlabel("Mean NMAE (%)")
    ax.set_title("Top 20 Worst Formulas by Mean NMAE (min 3 samples)")
    fig.tight_layout()
    fig.savefig(out_dir / "composition_families.png", dpi=150)
    plt.close(fig)


def _plot_by_functional(df: pd.DataFrame, out_dir: Path) -> None:
    if "functional" not in df.columns:
        return
    valid = df.dropna(subset=["functional"])
    valid = valid[valid["functional"] != ""]
    if valid.empty:
        return
    functionals = sorted(valid["functional"].unique())
    data_groups = [
        valid[valid["functional"] == f]["nmae"] * NMAE_TO_PERCENT for f in functionals
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data_groups, tick_labels=functionals)
    ax.set_ylabel("NMAE (%)")
    ax.set_title("NMAE by Functional")
    fig.tight_layout()
    fig.savefig(out_dir / "nmae_by_functional.png", dpi=150)
    plt.close(fig)


def _shape_summary(df: pd.DataFrame, threshold: float) -> str:
    """Binned summary tables for lattice shape and voxel anisotropy metrics."""
    lines: list[str] = []

    # Aspect ratio bins
    if "aspect_ratio" in df.columns:
        lines.append("\n=== Aspect Ratio Binned Summary ===")
        ar_bins = [1.0, 1.1, 1.5, 3.0, 5.0, 10.0, np.inf]
        ar_labels = [
            "1.0-1.1 (nearly cubic)",
            "1.1-1.5 (mildly anisotropic)",
            "1.5-3.0 (moderately anisotropic)",
            "3.0-5.0 (highly anisotropic)",
            "5.0-10.0 (very elongated)",
            ">10.0 (extremely elongated)",
        ]
        df["_ar_bin"] = pd.cut(df["aspect_ratio"], bins=ar_bins, labels=ar_labels)
        lines.append(
            f"  {'Bin':42s}  {'Count':>6s}  {'Mean%':>7s}  "
            f"{'Med%':>7s}  {'P95%':>7s}  {'Tail%':>6s}"
        )
        for label in ar_labels:
            group = df[df["_ar_bin"] == label]
            if len(group) == 0:
                continue
            nmae_pct = group["nmae"] * NMAE_TO_PERCENT
            tail_pct = (group["nmae"] >= threshold).mean() * 100
            lines.append(
                f"  {label:42s}  {len(group):6d}  {nmae_pct.mean():7.3f}  "
                f"{nmae_pct.median():7.3f}  {np.percentile(nmae_pct, 95):7.3f}  "
                f"{tail_pct:5.1f}%"
            )
        df = df.drop(columns=["_ar_bin"])

    # Voxel anisotropy bins
    if "voxel_anisotropy" in df.columns:
        lines.append("\n=== Voxel Anisotropy Binned Summary ===")
        va_bins = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, np.inf]
        va_labels = ["1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]
        df["_va_bin"] = pd.cut(df["voxel_anisotropy"], bins=va_bins, labels=va_labels)
        lines.append(
            f"  {'Bin':42s}  {'Count':>6s}  {'Mean%':>7s}  "
            f"{'Med%':>7s}  {'P95%':>7s}  {'Tail%':>6s}"
        )
        for label in va_labels:
            group = df[df["_va_bin"] == label]
            if len(group) == 0:
                continue
            nmae_pct = group["nmae"] * NMAE_TO_PERCENT
            tail_pct = (group["nmae"] >= threshold).mean() * 100
            lines.append(
                f"  {label:42s}  {len(group):6d}  {nmae_pct.mean():7.3f}  "
                f"{nmae_pct.median():7.3f}  {np.percentile(nmae_pct, 95):7.3f}  "
                f"{tail_pct:5.1f}%"
            )
        df = df.drop(columns=["_va_bin"])

    return "\n".join(lines)


def _plot_shape_vs_size_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """2D heatmap of mean NMAE for aspect_ratio bins x num_atoms bins."""
    if "aspect_ratio" not in df.columns or "num_atoms" not in df.columns:
        return

    ar_bins = [1.0, 1.1, 1.5, 3.0, 5.0, 10.0, np.inf]
    ar_labels = ["1.0-1.1", "1.1-1.5", "1.5-3.0", "3.0-5.0", "5.0-10.0", ">10.0"]
    atom_bins = [0, 4, 8, 16, 32, 64, np.inf]
    atom_labels = ["1-4", "5-8", "9-16", "17-32", "33-64", "65+"]

    df["_ar_bin"] = pd.cut(df["aspect_ratio"], bins=ar_bins, labels=ar_labels)
    df["_atom_bin"] = pd.cut(df["num_atoms"], bins=atom_bins, labels=atom_labels)

    pivot = (
        df.pivot_table(
            index="_ar_bin",
            columns="_atom_bin",
            values="nmae",
            aggfunc="mean",
            observed=False,
        )
        * NMAE_TO_PERCENT
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(atom_labels)))
    ax.set_yticks(range(len(ar_labels)))
    ax.set_xticklabels(atom_labels, fontsize=9)
    ax.set_yticklabels(ar_labels, fontsize=9)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Aspect Ratio")
    ax.set_title("Mean NMAE (%) by Aspect Ratio x Number of Atoms")

    # Annotate cells with value and count
    counts = df.pivot_table(
        index="_ar_bin",
        columns="_atom_bin",
        values="nmae",
        aggfunc="count",
        observed=False,
        fill_value=0,
    )
    for i in range(len(ar_labels)):
        for j in range(len(atom_labels)):
            val = data[i, j]
            cnt = int(counts.to_numpy()[i, j])
            if cnt > 0 and not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}%\nn={cnt}", ha="center", va="center", fontsize=7
                )

    fig.colorbar(im, ax=ax, label="Mean NMAE (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "shape_vs_size_heatmap.png", dpi=150)
    plt.close(fig)

    df = df.drop(columns=["_ar_bin", "_atom_bin"])


def main(argv: list[str] | None = None) -> None:
    import matplotlib as mpl

    mpl.use("Agg")

    parser = argparse.ArgumentParser(
        description="Analyze NMAE tail and correlate with structural metadata."
    )
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--tail-threshold",
        default="95p",
        help="Tail threshold: percentile (e.g. '95p') or absolute NMAE (e.g. '0.05')",
    )
    parser.add_argument(
        "--split-file", default=None, help="Path to split.json for train/val annotation"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load & join
    split_path = Path(args.split_file) if args.split_file else None
    df = _load_and_join(Path(args.metrics), Path(args.metadata), split_path)
    logger.info(
        "Joined dataset: %d rows, %d with metadata",
        len(df),
        df["task_id"].notna().sum(),
    )

    threshold = _parse_threshold(args.tail_threshold, df["nmae"])
    df["in_tail"] = df["nmae"] >= threshold

    # Step 2: Analyses
    summary_text = _summary_stats(df, threshold)
    logger.info("\n%s", summary_text)

    elem_stats, _ = _element_analysis(df, threshold)
    corr_df = _correlations(df)
    comp_df = _composition_families(df)

    # Add correlations to summary
    if not corr_df.empty:
        summary_text += "\n\n=== Correlations with NMAE ===\n"
        for _, row in corr_df.iterrows():
            summary_text += (
                f"  {row['feature']:25s}  "
                f"Spearman r={row['spearman_r']:+.4f} (p={row['spearman_p']:.2e})  "
                f"Pearson r={row['pearson_r']:+.4f} (p={row['pearson_p']:.2e})\n"
            )

    # Add shape summary tables
    shape_text = _shape_summary(df, threshold)
    if shape_text:
        summary_text += "\n" + shape_text

    # Step 3: Write outputs
    df.to_csv(out_dir / "enriched_metrics.csv", index=False)
    logger.info("Wrote enriched_metrics.csv")

    with (out_dir / "summary.txt").open("w") as f:
        f.write(summary_text)
    logger.info("Wrote summary.txt")

    outliers = df.nlargest(50, "nmae")
    outliers.to_csv(out_dir / "outlier_structures.csv", index=False)
    logger.info("Wrote outlier_structures.csv")

    # Plots
    _plot_distribution(df, threshold, out_dir)
    _plot_scatter(
        df, "num_atoms", "nmae_vs_num_atoms.png", out_dir, xlabel="Number of Atoms"
    )
    _plot_scatter(
        df,
        "volume",
        "nmae_vs_volume.png",
        out_dir,
        color_col="functional",
        xlabel="Volume (Ang^3)",
    )
    _plot_scatter(
        df,
        "mean_voxel_edge",
        "nmae_vs_voxel_resolution.png",
        out_dir,
        xlabel="Mean Voxel Edge (Ang)",
    )
    _plot_scatter(
        df,
        "aspect_ratio",
        "nmae_vs_aspect_ratio.png",
        out_dir,
        xlabel="Aspect Ratio (max/min lattice param)",
        log_x=True,
    )
    _plot_scatter(
        df,
        "voxel_anisotropy",
        "nmae_vs_voxel_anisotropy.png",
        out_dir,
        xlabel="Voxel Anisotropy (max/min voxel edge)",
        log_x=True,
    )
    _plot_element_heatmap(elem_stats, out_dir)
    _plot_enrichment(elem_stats, out_dir)
    _plot_correlation_matrix(df, out_dir)
    _plot_composition_families(comp_df, out_dir)
    _plot_by_functional(df, out_dir)
    _plot_shape_vs_size_heatmap(df, out_dir)

    logger.info("All outputs written to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
