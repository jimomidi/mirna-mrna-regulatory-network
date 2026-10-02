"""
Single-gene and multi-gene diagnostic ROC analysis (tumor vs. normal
classification from expression level alone).

For each gene, a ROC curve is computed by treating raw expression as the
classification score (normal = 0, tumor = 1), together with a Mann-Whitney U
test p-value. A combined multi-gene logistic regression model (expression of
every requested gene, standardized, as predictors) is also fit and its ROC
curve plotted alongside the single-gene panels.

Usage
-----
    python -m classification_and_survival.roc_diagnostic_analysis \
        --genes GENE_A GENE_B GENE_C \
        --output results/roc_diagnostic.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import utils
from differential_expression.expression_boxplot import build_combined_matrix


def run(genes: list[str], tumor_file: Path, normal_file: Path, output_path: Path) -> None:
    utils.set_plot_style()
    plt.rcParams["font.size"] = 20

    tumor_df = utils.load_tumor_expression(tumor_file)
    normal_df = pd.read_csv(normal_file)
    expr_df, batch_labels = build_combined_matrix(tumor_df, normal_df, genes)
    genes_available = list(expr_df.index)

    fig, axes = plt.subplots(1, len(genes_available) + 1, figsize=(6 * (len(genes_available) + 1), 6))
    y_true = np.array([1 if b == "Tumor" else 0 for b in batch_labels])

    for i, gene in enumerate(genes_available):
        ax = axes[i]
        y_score = expr_df.loc[gene].astype(float).values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        expr_normal = y_score[y_true == 0]
        expr_tumor = y_score[y_true == 1]
        _, p_value = mannwhitneyu(expr_normal, expr_tumor, alternative="two-sided")

        ax.plot(fpr, tpr, color="dodgerblue", lw=3)
        ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
        ax.set_xlim(-0.01, 1)
        ax.set_ylim(0, 1)
        ax.set_title(gene, fontsize=28, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=18, fontweight="bold")
        if i == 0:
            ax.set_ylabel("True Positive Rate", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.26, f"AUC = {roc_auc:.2f}", transform=ax.transAxes, fontsize=18)
        ax.text(0.5, 0.15, f"p = {p_value:.2e}", transform=ax.transAxes, fontsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[-1]
    X = expr_df.T.values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_true)
    y_score = model.predict_proba(X_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    _, p_value = mannwhitneyu(y_score[y_true == 1], y_score[y_true == 0], alternative="two-sided")

    ax.plot(fpr, tpr, color="dodgerblue", lw=3)
    ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-gene Logistic Regression", fontsize=24, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.26, f"AUC = {roc_auc:.2f}", transform=ax.transAxes, fontsize=18)
    ax.text(0.5, 0.15, f"p = {p_value:.2e}", transform=ax.transAxes, fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "roc_diagnostic.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.genes, args.tumor_file, args.normal_file, args.output)
