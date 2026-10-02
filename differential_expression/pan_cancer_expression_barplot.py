"""
Pan-cancer log2 fold-change bar chart for a set of genes.

Expects, for each gene, a single-column CSV of paired [normal, tumor,
normal, tumor, ...] median expression values - one pair per cancer type, in
the same order as `--cancer-types` - as exported by pan-cancer expression
browsers such as GEPIA2. log2 fold-change (tumor vs. normal) is computed for
every gene/cancer-type combination and plotted as a grouped bar chart.

Usage
-----
    python -m differential_expression.pan_cancer_expression_barplot \
        --gene-files GENE_A=data/pan_cancer/GENE_A.csv GENE_B=data/pan_cancer/GENE_B.csv \
        --cancer-types BLCA BRCA CESC CHOL COAD \
        --output results/pan_cancer_log2fc.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils

DEFAULT_COLORS = ["#1f77b4", "#e377c2", "#2ca02c", "#ffcc00", "#9467bd", "#8c564b"]


def compute_log2fc_table(gene_files: dict[str, Path], cancer_types: list[str]) -> pd.DataFrame:
    """Return a long-format (Cancer, Gene, log2FC) table from paired normal/tumor exports."""
    records = []
    for gene, path in gene_files.items():
        values = pd.read_csv(path, header=None).iloc[:, 0].tolist()
        n_pairs = min(len(values) // 2, len(cancer_types))
        for i in range(n_pairs):
            normal_val, tumor_val = values[2 * i], values[2 * i + 1]
            log2fc = np.log2(tumor_val / normal_val) if normal_val > 0 else np.nan
            records.append({"Cancer": cancer_types[i], "Gene": gene, "log2FC": log2fc})
    return pd.DataFrame(records)


def plot_barplot(df_long: pd.DataFrame, output_path: Path) -> None:
    utils.set_plot_style()
    plt.rcParams["axes.labelweight"] = "bold"

    df_wide = df_long.pivot(index="Cancer", columns="Gene", values="log2FC")
    genes = list(df_wide.columns)
    n_cancer = len(df_wide)
    bar_width = 0.8 / max(len(genes), 1)
    x = np.arange(n_cancer)

    fig = plt.figure(figsize=(max(12, n_cancer), 6))
    for i, gene in enumerate(genes):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        plt.bar(x + i * bar_width, df_wide[gene], width=bar_width, color=color,
                label=gene, edgecolor="black", linewidth=1.2, zorder=3)

    plt.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=1)
    plt.axhline(1, color="gray", linewidth=1.0, linestyle="--", zorder=0)
    plt.axhline(-1, color="gray", linewidth=1.0, linestyle="--", zorder=0)

    plt.xticks(x + bar_width * (len(genes) - 1) / 2, df_wide.index, rotation=45, fontsize=14)
    plt.ylabel("log2FC (Tumor vs. Normal)", fontsize=18, weight="bold", labelpad=15)
    plt.xlim(x[0] - 0.25, x[-1] + 0.75)
    plt.legend(frameon=False, loc="upper center", ncol=len(genes))

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to: {output_path}")


def run(gene_files: dict[str, Path], cancer_types: list[str], output_path: Path) -> None:
    df_long = compute_log2fc_table(gene_files, cancer_types)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path = output_path.with_suffix(".csv")
    df_long.to_csv(table_path, index=False)
    print(f"log2FC table written to: {table_path}")

    plot_barplot(df_long, output_path)


def parse_gene_file_args(pairs: list[str]) -> dict[str, Path]:
    result = {}
    for pair in pairs:
        gene, path = pair.split("=", 1)
        result[gene] = Path(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gene-files", nargs="+", required=True,
        help="GENE=path/to/file.csv pairs, one per gene of interest.",
    )
    parser.add_argument("--cancer-types", nargs="+", required=True, help="Ordered cancer-type labels.")
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "pan_cancer_log2fc.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(parse_gene_file_args(args.gene_files), args.cancer_types, args.output)
