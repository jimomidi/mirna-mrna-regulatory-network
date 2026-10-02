"""
Volcano plot of differential expression results (log2 fold-change vs.
statistical significance), with optional highlighting of a gene set of
interest.

Accepts a differential-expression results table such as a limma/DESeq2
export (one row per feature, with a fold-change column and a p-value
column). If `--highlight-genes` is supplied together with `--gene-id-map`
(a table mapping the result table's row identifier to a gene symbol, e.g.
the tumor expression file's gene_id/gene_name columns), each highlighted
gene is drawn as a distinct colored point with its own legend entry;
otherwise the plot simply colors points by up/down/not-significant.

Usage
-----
    python -m differential_expression.volcano_plot \
        --input data/differential_expression/volcano_input.csv \
        --output results/volcano.png

    python -m differential_expression.volcano_plot \
        --input data/differential_expression/volcano_input.csv \
        --highlight-genes GENE_A GENE_B GENE_C \
        --gene-id-map data/expression/tumor_expression.csv \
        --output results/volcano_highlighted.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


def load_results(path: Path, id_col: Optional[str], logfc_col: str, pval_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0 if id_col is None else None)
    if id_col is not None:
        df = df.set_index(id_col)
    df.columns = df.columns.str.strip()
    return df


def build_gene_id_map(mapping_path: Path) -> dict:
    """Build a {gene_id: gene_name} lookup from a tumor-expression-style table."""
    df = pd.read_csv(mapping_path)
    df = df[df["gene_id"].astype(str).str.startswith("ENSG") & df["gene_name"].notna()]
    df["gene_id"] = df["gene_id"].astype(str).str.split(".").str[0]
    return df.drop_duplicates("gene_id").set_index("gene_id")["gene_name"].to_dict()


def plot_basic_volcano(
    df: pd.DataFrame,
    logfc_col: str,
    pval_col: str,
    logfc_threshold: float,
    pval_threshold: float,
    output_path: Path,
) -> None:
    utils.set_plot_style()

    logfc = df[logfc_col]
    neg_log_pval = -np.log10(df[pval_col])
    up = (logfc > logfc_threshold) & (df[pval_col] < pval_threshold)
    down = (logfc < -logfc_threshold) & (df[pval_col] < pval_threshold)
    neutral = ~(up | down)

    plt.figure(figsize=(8, 6))
    plt.scatter(logfc[neutral], neg_log_pval[neutral], c="grey", alpha=0.6, s=30, label="Not significant")
    plt.scatter(logfc[up], neg_log_pval[up], c="red", alpha=0.8, s=30, label=f"Up ({up.sum()})")
    plt.scatter(logfc[down], neg_log_pval[down], c="blue", alpha=0.8, s=30, label=f"Down ({down.sum()})")

    plt.axhline(-np.log10(pval_threshold), color="black", linestyle="--", linewidth=1)
    plt.axvline(logfc_threshold, color="black", linestyle="--", linewidth=1)
    plt.axvline(-logfc_threshold, color="black", linestyle="--", linewidth=1)

    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(p-value)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"Figure written to: {output_path}")


def plot_highlighted_volcano(
    df: pd.DataFrame,
    logfc_col: str,
    pval_col: str,
    logfc_threshold: float,
    pval_threshold: float,
    highlight_genes: list[str],
    gene_id_map: dict,
    output_path: Path,
) -> None:
    utils.set_plot_style()

    name_to_ids: dict[str, list[str]] = {}
    for gene_id, gene_name in gene_id_map.items():
        name_to_ids.setdefault(gene_name.upper(), []).append(gene_id)

    plt.figure(figsize=(7, 6.5))
    plt.scatter(df[logfc_col], -np.log10(df[pval_col]), color="grey", alpha=0.7, s=20)

    palette = utils.get_cmap("tab20", max(len(highlight_genes), 1))
    legend_handles = []
    for i, gene in enumerate(highlight_genes):
        gene_ids = name_to_ids.get(gene.upper(), [])
        color = palette(i)
        for gene_id in gene_ids:
            if gene_id not in df.index:
                continue
            x, y = df.loc[gene_id, logfc_col], -np.log10(df.loc[gene_id, pval_col])
            plt.scatter(x, y, color=color, s=90, edgecolor="black", linewidth=0.5)
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", label=gene,
                       markerfacecolor=color, markeredgecolor="black", markersize=8)
        )

    plt.axhline(-np.log10(pval_threshold), color="black", linestyle="--", linewidth=1)
    plt.axvline(logfc_threshold, color="black", linestyle="--", linewidth=1)
    plt.axvline(-logfc_threshold, color="black", linestyle="--", linewidth=1)

    plt.xlabel("log2(FC)")
    plt.ylabel("-log10(p-value)")
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Figure written to: {output_path}")


def run(
    input_path: Path,
    id_col: Optional[str],
    logfc_col: str,
    pval_col: str,
    highlight_genes: Optional[list[str]],
    gene_id_map_path: Optional[Path],
    output_path: Path,
) -> None:
    df = load_results(input_path, id_col, logfc_col, pval_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if highlight_genes:
        if gene_id_map_path is None:
            raise ValueError("--gene-id-map is required when --highlight-genes is used.")
        gene_id_map = build_gene_id_map(gene_id_map_path)
        plot_highlighted_volcano(
            df, logfc_col, pval_col, config.LOGFC_THRESHOLD, config.PVALUE_THRESHOLD,
            highlight_genes, gene_id_map, output_path,
        )
    else:
        plot_basic_volcano(
            df, logfc_col, pval_col, config.LOGFC_THRESHOLD, config.PVALUE_THRESHOLD, output_path
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Differential expression results CSV.")
    parser.add_argument(
        "--id-col", type=str, default=None,
        help="Column holding the feature identifier (defaults to the first column / CSV index).",
    )
    parser.add_argument("--logfc-col", type=str, default="logFC")
    parser.add_argument("--pval-col", type=str, default="P.Value")
    parser.add_argument("--highlight-genes", nargs="+", default=None)
    parser.add_argument(
        "--gene-id-map", type=Path, default=None,
        help="Tumor-expression-style table providing gene_id -> gene_name (required with --highlight-genes).",
    )
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "volcano.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.id_col, args.logfc_col, args.pval_col, args.highlight_genes, args.gene_id_map, args.output)
