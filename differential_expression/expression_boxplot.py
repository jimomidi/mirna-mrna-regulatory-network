"""
Compare normal vs. tumor expression for a list of genes with a paired
boxplot + violin plot and a Welch's t-test per gene.

Note on batch-effect correction
--------------------------------
An earlier version of this script first regressed expression on a
normal/tumor indicator (ordinary least squares) and used the residuals in
place of raw expression, framed as "batch-effect correction" for combining
two independently processed cohorts (a normal-tissue reference cohort and a
tumor cohort). That step has been removed: the "batch" indicator used was
identical to the normal/tumor group being compared (every normal sample
comes from one cohort and every tumor sample from the other), so regressing
on it and taking residuals removes, by construction, exactly the between-group
mean difference this comparison is meant to test - both corrected groups end
up with an (numerically) equal mean regardless of the true underlying
difference. Any "significant" result computed downstream from those
residuals (after the log2 transform) would reflect an artifact of unequal
within-group variance interacting with a nonlinear transform, not a genuine
expression difference. This is scientifically inappropriate whenever the
correction variable is fully confounded with the outcome being tested, which
is the case here, so the comparison now uses raw expression directly.

Usage
-----
    python -m differential_expression.expression_boxplot \
        --genes GENE_A GENE_B GENE_C \
        --output results/expression_boxplot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

import config
import utils


def build_combined_matrix(
    tumor_df: pd.DataFrame, normal_df: pd.DataFrame, genes: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Return a genes x samples matrix combining normal and tumor samples, and
    the matching per-sample batch label list."""
    expr_data: dict[str, np.ndarray] = {}
    batch_labels: list[str] = []
    sample_names: list[str] = []

    for gene in genes:
        normal_row = normal_df[normal_df["Description"] == gene]
        if normal_row.empty:
            continue
        normal_vals = normal_row.iloc[:, 2:].values.flatten()
        tumor_vals = tumor_df[tumor_df["gene_name"] == gene]["tpm_unstranded"].values

        all_vals = np.concatenate([normal_vals, tumor_vals])
        expr_data[gene] = all_vals

        if not batch_labels:
            batch_labels = ["Normal"] * len(normal_vals) + ["Tumor"] * len(tumor_vals)
            sample_names = [f"N{i}" for i in range(len(normal_vals))] + [f"T{i}" for i in range(len(tumor_vals))]

    return pd.DataFrame(expr_data, index=sample_names).T, batch_labels


def run(genes: list[str], tumor_file: Path, normal_file: Path, output_path: Path) -> None:
    utils.set_plot_style()
    plt.rcParams["font.size"] = 16

    tumor_df = utils.load_tumor_expression(tumor_file)
    tumor_df = tumor_df.dropna(subset=["gene_name", "tpm_unstranded"])
    tumor_df["tpm_unstranded"] = tumor_df["tpm_unstranded"].astype(float)
    normal_df = pd.read_csv(normal_file)

    expr_df, batch_labels = build_combined_matrix(tumor_df, normal_df, genes)

    keep = [
        gene for gene in expr_df.index
        if np.sum(expr_df.loc[gene] >= config.LOW_EXPRESSION_TPM_THRESHOLD) / expr_df.shape[1]
        >= config.MIN_NONZERO_FRACTION
    ]
    expr_df = expr_df.loc[keep]
    if expr_df.empty:
        raise ValueError("No gene passed the low-expression filter.")

    fig, axes = plt.subplots(nrows=2, ncols=len(keep), figsize=(5 * len(keep), 10), squeeze=False)

    print("Welch's t-test p-values:")
    for idx, gene in enumerate(keep):
        log_vals = np.log2(expr_df.loc[gene].astype(float).values + 1)
        gene_df = pd.DataFrame({"expression": log_vals, "group": batch_labels})

        normal_vals = gene_df[gene_df["group"] == "Normal"]["expression"].dropna()
        tumor_vals = gene_df[gene_df["group"] == "Tumor"]["expression"].dropna()
        if len(normal_vals) < 2 or len(tumor_vals) < 2 or normal_vals.std() == 0 or tumor_vals.std() == 0:
            print(f"  {gene}: not enough data / zero variance, skipped")
            continue

        stat, pval = ttest_ind(normal_vals, tumor_vals, equal_var=False)
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {gene}: p = {pval:.4e} ({stars})")

        palette = {"Normal": "#053061", "Tumor": "#67001f"}
        sns.boxplot(
            x="group", y="expression", data=gene_df, ax=axes[0, idx], showfliers=False,
            color="black", medianprops=dict(color="white", linewidth=1),
        )
        sns.stripplot(
            x="group", y="expression", hue="group", data=gene_df, ax=axes[0, idx],
            size=6, alpha=0.6, palette=palette, legend=False,
        )
        axes[0, idx].set_title(gene, fontweight="bold")
        axes[0, idx].set_xlabel("")
        axes[0, idx].set_ylabel("log2(TPM + 1)" if idx == 0 else "")

        sns.violinplot(x="group", y="expression", data=gene_df, ax=axes[1, idx], color="black")
        sns.stripplot(
            x="group", y="expression", hue="group", data=gene_df, ax=axes[1, idx],
            size=6, alpha=0.6, palette=palette, legend=False,
        )
        ymax = gene_df["expression"].max()
        axes[1, idx].text(0.5, ymax - 0.5, stars, ha="center", va="top", fontsize=18, fontweight="bold")

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
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "expression_boxplot.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.genes, args.tumor_file, args.normal_file, args.output)
