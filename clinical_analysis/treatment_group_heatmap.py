"""
Compare expression of a gene set between two treatment-defined case groups
derived directly from the clinical table (e.g. cases that received a
secondary therapy in addition to the primary one, vs. cases that received
only the primary therapy), and visualize the top differentially expressed
genes as two min-max-normalized heatmaps.

Case groups are derived programmatically from the `treatment_type` /
`treatment_or_therapy` columns of the clinical table rather than hard-coded
case ID lists, so the analysis reproduces on any cohort with the same
clinical schema.

Usage
-----
    python -m clinical_analysis.treatment_group_heatmap \
        --genes GENE_A GENE_B GENE_C \
        --primary-treatment Pharmaceutical --secondary-treatment Radiation \
        --output results/treatment_group_heatmap.png
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


def run(
    genes: list[str],
    tumor_file: Path,
    clinical_file: Path,
    primary_treatment: str,
    secondary_treatment: str,
    top_n: int,
    output_path: Path,
) -> None:
    utils.set_plot_style()
    plt.rcParams["font.size"] = 16

    tumor_df = utils.load_tumor_expression(tumor_file)
    clinical = utils.load_clinical(clinical_file)

    genes_subset = tumor_df[tumor_df["gene_name"].isin(genes)]
    gene_id_to_name = genes_subset[["gene_id", "gene_name"]].drop_duplicates().set_index("gene_id")["gene_name"].to_dict()

    primary_cases = utils.cases_by_treatment(clinical, contains=primary_treatment)
    rna_primary = tumor_df[
        tumor_df["case_id"].isin(primary_cases) & tumor_df["gene_id"].isin(gene_id_to_name.keys())
    ]
    expr_matrix = rna_primary.pivot_table(index="case_id", columns="gene_id", values="tpm_unstranded")
    expr_matrix = expr_matrix.loc[:, (expr_matrix != 0).any(axis=0)]
    valid = (expr_matrix > config.PSEUDOCOUNT).mean(axis=0) >= config.MIN_NONZERO_FRACTION
    expr_matrix = expr_matrix.loc[:, valid]

    group_a_cases = sorted(utils.cases_by_treatment(clinical, contains=primary_treatment, also_received=secondary_treatment))
    group_b_cases = sorted(utils.cases_by_treatment(clinical, contains=primary_treatment, exclude=secondary_treatment))
    group_a_cases = [c for c in group_a_cases if c in expr_matrix.index]
    group_b_cases = [c for c in group_b_cases if c in expr_matrix.index]

    results = []
    epsilon = 1e-5
    for gene in expr_matrix.columns:
        g1 = expr_matrix.loc[group_a_cases, gene].dropna()
        g2 = expr_matrix.loc[group_b_cases, gene].dropna()
        if len(g1) > 1 and len(g2) > 1:
            logfc = np.log2((g1.mean() + epsilon) / (g2.mean() + epsilon))
            _, pval = ttest_ind(g1, g2, equal_var=False)
            score = logfc * -np.log10(pval + 1e-10)
            results.append((gene, gene_id_to_name.get(gene, gene), logfc, pval, score))

    df_results = pd.DataFrame(results, columns=["gene_id", "gene_name", "logFC", "p_value", "EffectScore"])
    top_genes = df_results.sort_values("EffectScore", ascending=False).head(top_n)

    expr_top = expr_matrix[top_genes["gene_id"].tolist()]
    expr_top.columns = [gene_id_to_name.get(c, c) for c in expr_top.columns]

    def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.min().min()) / (df.max().max() - df.min().min())

    datasets = [min_max_normalize(expr_top.loc[group_a_cases]), min_max_normalize(expr_top.loc[group_b_cases])]
    titles = [f"{secondary_treatment} & {primary_treatment} Therapy", f"{primary_treatment} Therapy Only"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for i in range(2):
        sns.heatmap(datasets[i], ax=axes[i], cmap="PiYG_r", cbar_kws={"label": " "}, linewidths=0.5, linecolor="white")
        axes[i].set_title(titles[i], fontsize=22, fontweight="bold")
        axes[i].set_ylabel("Case ID", fontsize=16, fontweight="bold")
        axes[i].tick_params(axis="x", labelrotation=90, labelsize=14)
        axes[i].tick_params(axis="y", labelrotation=0, labelsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Figure written to: {output_path}")

    table_path = output_path.with_suffix(".csv")
    df_results.sort_values("EffectScore", ascending=False).to_csv(table_path, index=False)
    print(f"Differential expression table written to: {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", nargs="+", default=config.GENES_OF_INTEREST)
    parser.add_argument("--tumor-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument("--clinical-file", type=Path, default=config.CLINICAL_FILE)
    parser.add_argument("--primary-treatment", type=str, default="Pharmaceutical")
    parser.add_argument("--secondary-treatment", type=str, default="Radiation")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "treatment_group_heatmap.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.genes, args.tumor_file, args.clinical_file,
        args.primary_treatment, args.secondary_treatment, args.top_n, args.output,
    )
