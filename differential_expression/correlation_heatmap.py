"""
Side-by-side correlation heatmaps (normal, tumor, and the tumor-minus-normal
difference) for a list of genes.

Usage
-----
    python -m differential_expression.correlation_heatmap \
        --genes GENE_A GENE_B GENE_C GENE_D \
        --output results/correlation_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
import utils


def compute_correlation_matrices(
    genes: list[str], tumor_file: Path, normal_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_normal = pd.read_csv(normal_file)
    df_normal = df_normal[df_normal["Description"].isin(genes)].set_index("Description")
    expr_normal = df_normal.drop(columns=["Name"]).T.apply(pd.to_numeric, errors="coerce")
    expr_normal = utils.filter_low_expression(expr_normal, threshold=1.0, max_low_fraction=0.8, axis=1)
    expr_normal = utils.log2_transform(expr_normal)

    df_tumor = utils.load_tumor_expression(tumor_file)
    df_tumor = df_tumor[df_tumor["gene_name"].isin(genes)]
    pivot_tumor = df_tumor.pivot_table(index="case_id", columns="gene_name", values="fpkm_unstranded", aggfunc="mean")
    expr_tumor = utils.filter_low_expression(pivot_tumor, threshold=1.0, max_low_fraction=0.8, axis=1)
    expr_tumor = utils.log2_transform(expr_tumor)

    corr_normal = expr_normal.corr()
    corr_tumor = expr_tumor.corr()
    corr_diff = corr_tumor - corr_normal

    order = [g for g in genes if g in corr_normal.columns and g in corr_tumor.columns]
    corr_normal = corr_normal.loc[order, order].iloc[::-1]
    corr_tumor = corr_tumor.loc[order, order].iloc[::-1]
    corr_diff = corr_diff.loc[order, order].iloc[::-1]
    return corr_normal, corr_tumor, corr_diff


def plot_heatmaps(
    corr_normal: pd.DataFrame, corr_tumor: pd.DataFrame, corr_diff: pd.DataFrame, output_path: Path
) -> None:
    utils.set_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.3})
    cmap, vmin, vmax = "RdBu_r", -1, 1
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])

    matrices = [corr_normal, corr_tumor, corr_diff]
    titles = ["Correlation - Normal", "Correlation - Tumor", "Correlation Difference"]

    sns.heatmap(
        matrices[0], annot=True, cmap=cmap, center=0, square=True, fmt=".2f",
        linewidths=0.8, linecolor="black", ax=axes[0], cbar=True, cbar_ax=cbar_ax,
        vmin=vmin, vmax=vmax, annot_kws={"size": 14},
    )
    axes[0].set_title(titles[0], fontsize=16, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    for ax, matrix, title in zip(axes[1:], matrices[1:], titles[1:]):
        sns.heatmap(
            matrix, annot=True, cmap=cmap, center=0, square=True, fmt=".2f",
            linewidths=0.8, linecolor="black", ax=ax, cbar=False, vmin=vmin, vmax=vmax,
            annot_kws={"size": 14},
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to: {output_path}")


def run(genes: list[str], tumor_file: Path, normal_file: Path, output_path: Path) -> None:
    corr_normal, corr_tumor, corr_diff = compute_correlation_matrices(genes, tumor_file, normal_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(corr_normal, corr_tumor, corr_diff, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", nargs="+", default=config.GENES_OF_INTEREST)
    parser.add_argument("--tumor-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument("--normal-file", type=Path, default=config.NORMAL_EXPRESSION_FILE)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "correlation_heatmap.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.genes, args.tumor_file, args.normal_file, args.output)
