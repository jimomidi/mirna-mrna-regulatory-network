"""
Scatter plots (with linear fit) comparing the expression correlation between
pairs of genes in normal vs. tumor samples.

For every requested gene, all pairwise combinations are plotted side by
side, each panel showing the normal-sample points/fit and the tumor-sample
points/fit together with their Pearson correlation coefficients.

Usage
-----
    python -m differential_expression.gene_gene_correlation_scatter \
        --genes GENE_A GENE_B GENE_C \
        --output results/gene_correlation_scatter.png
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

import config
import utils


def prepare_expression(
    genes: list[str], tumor_file: Path, normal_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_file)
    df_normal = df_normal[df_normal["Description"].isin(genes)].set_index("Description")
    expr_normal = df_normal.drop(columns=["Name"]).T.apply(pd.to_numeric, errors="coerce")
    expr_normal = utils.filter_low_expression(expr_normal, threshold=1.0, max_low_fraction=0.8, axis=1)
    expr_normal = utils.log2_transform(expr_normal)

    df_tumor = utils.load_tumor_expression(tumor_file)
    df_tumor = df_tumor[df_tumor["gene_name"].isin(genes)].copy()
    df_tumor["log2_fpkm"] = utils.log2_transform(df_tumor["fpkm_unstranded"])
    pivot_tumor = df_tumor.pivot_table(index="case_id", columns="gene_name", values="fpkm_unstranded", aggfunc="mean")
    expr_tumor = utils.filter_low_expression(pivot_tumor, threshold=1.0, max_low_fraction=0.8, axis=1)
    expr_tumor = utils.log2_transform(expr_tumor)

    return expr_normal, expr_tumor


def plot_pair(ax, expr_normal, expr_tumor, gene_x, gene_y, normal_color, tumor_color):
    if gene_x in expr_normal.columns and gene_y in expr_normal.columns:
        x, y = expr_normal[gene_x], expr_normal[gene_y]
        r, _ = pearsonr(x, y)
        slope, intercept, *_ = linregress(x, y)
        ax.scatter(x, y, color=normal_color, s=100, alpha=0.7, label=f"Normal (r={r:.2f})")
        ax.plot(x, slope * x + intercept, linestyle="--", color=normal_color)

    if gene_x in expr_tumor.columns and gene_y in expr_tumor.columns:
        x, y = expr_tumor[gene_x], expr_tumor[gene_y]
        r, _ = pearsonr(x, y)
        slope, intercept, *_ = linregress(x, y)
        ax.scatter(x, y, color=tumor_color, s=100, alpha=0.7, label=f"Tumor (r={r:.2f})")
        ax.plot(x, slope * x + intercept, linestyle="--", color=tumor_color)

    ax.set_title(f"{gene_x} vs. {gene_y}")
    ax.set_xlabel(f"log2({gene_x} + {config.PSEUDOCOUNT})")
    ax.set_ylabel(f"log2({gene_y} + {config.PSEUDOCOUNT})")
    ax.legend(frameon=False)


def run(genes: list[str], tumor_file: Path, normal_file: Path, output_path: Path) -> None:
    utils.set_plot_style()
    plt.rcParams.update({
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.titlesize": 20, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 14,
    })

    expr_normal, expr_tumor = prepare_expression(genes, tumor_file, normal_file)

    cmap = utils.get_cmap("PiYG_r", 5)
    tumor_color, normal_color = cmap(0), cmap(4)

    gene_pairs = list(combinations(genes, 2))
    n_cols = min(len(gene_pairs), 3)
    n_rows = int(np.ceil(len(gene_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (gene_x, gene_y) in zip(axes_flat, gene_pairs):
        plot_pair(ax, expr_normal, expr_tumor, gene_x, gene_y, normal_color, tumor_color)
    for ax in axes_flat[len(gene_pairs):]:
        fig.delaxes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", nargs="+", default=config.GENES_OF_INTEREST)
    parser.add_argument("--tumor-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument("--normal-file", type=Path, default=config.NORMAL_EXPRESSION_FILE)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "gene_correlation_scatter.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.genes, args.tumor_file, args.normal_file, args.output)
