"""
Correlate one or more miRNAs against messenger RNA expression to identify
putative regulatory targets by expression anti-correlation.

For each requested miRNA, this script correlates its per-case log2 expression
against either a supplied candidate gene list (e.g. genes predicted to be
targeted by that miRNA, from `mirna_target_prediction_filter.py`) or, if no
candidate list is given, every gene passing the expression filter - useful
for an exploratory genome-wide scan. Pairs with a paired sample count below
`--min-samples` are skipped. Significant negative correlations (the expected
direction for miRNA-mediated repression) are reported separately.

Usage
-----
    python -m preprocessing.mirna_mrna_correlation_filter \
        --mirna hsa-mir-example-1 \
        --candidate-genes results/mirna_example_1_targets.csv \
        --output results/mirna_example_1_correlations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config


def _filter_low_expression_long(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    min_nonzero_fraction: float,
    detection_threshold: float = 1.0,
) -> pd.DataFrame:
    """Drop features (miRNAs or genes) expressed above `detection_threshold` in
    fewer than `min_nonzero_fraction` of samples, matching the exploratory
    analysis's low-expression filter."""
    matrix = df.pivot_table(index=id_col, columns="case_id", values=value_col)
    keep = matrix.gt(detection_threshold).sum(axis=1) >= min_nonzero_fraction * matrix.shape[1]
    keep_ids = matrix.index[keep]
    return df[df[id_col].isin(keep_ids)]


def correlate_mirna_targets(
    mirna_list: Iterable[str],
    df_mirna: pd.DataFrame,
    df_mrna: pd.DataFrame,
    candidate_genes: Optional[Iterable[str]] = None,
    min_nonzero_fraction: float = config.MIN_NONZERO_FRACTION,
    min_paired_samples: int = config.MIN_PAIRED_SAMPLES,
) -> pd.DataFrame:
    """Return a long table of (miRNA, gene, r, p) for every requested pair."""
    df_mirna = _filter_low_expression_long(
        df_mirna, "miRNA_ID", "reads_per_million_miRNA_mapped", min_nonzero_fraction
    )
    df_mrna = _filter_low_expression_long(df_mrna, "gene_name", "fpkm_unstranded", min_nonzero_fraction)

    df_mirna = df_mirna.copy()
    df_mirna["log2_rpm"] = np.log2(df_mirna["reads_per_million_miRNA_mapped"] + config.PSEUDOCOUNT)

    df_mrna = df_mrna.copy()
    df_mrna["gene_id"] = df_mrna["gene_id"].astype(str).str.replace(r"\..*", "", regex=True)
    df_mrna["log2_fpkm"] = np.log2(df_mrna["fpkm_unstranded"] + config.PSEUDOCOUNT)

    genes_to_test = list(candidate_genes) if candidate_genes is not None else df_mrna["gene_name"].unique()

    results = []
    for mirna in mirna_list:
        mirna_expr = (
            df_mirna[df_mirna["miRNA_ID"] == mirna][["case_id", "log2_rpm"]]
            .drop_duplicates("case_id")
        )
        if mirna_expr.empty:
            continue

        for gene in genes_to_test:
            gene_expr = (
                df_mrna[df_mrna["gene_name"] == gene][["case_id", "log2_fpkm"]]
                .drop_duplicates("case_id")
            )
            if gene_expr.empty:
                continue

            merged = pd.merge(mirna_expr, gene_expr, on="case_id")
            if len(merged) < min_paired_samples:
                continue

            r, p = pearsonr(merged["log2_rpm"], merged["log2_fpkm"])
            results.append({"miRNA": mirna, "gene": gene, "r": r, "p": p, "n_samples": len(merged)})

    return pd.DataFrame(results)


def run(
    mirna_list: list[str],
    mirna_file: Path,
    mrna_file: Path,
    candidate_genes_file: Path | None,
    output_path: Path,
) -> None:
    df_mirna = pd.read_csv(mirna_file)
    df_mrna = pd.read_csv(mrna_file)

    candidate_genes = None
    if candidate_genes_file is not None:
        candidate_genes = pd.read_csv(candidate_genes_file)["gene"].dropna().astype(str).unique().tolist()

    results = correlate_mirna_targets(mirna_list, df_mirna, df_mrna, candidate_genes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"{len(results)} miRNA-gene pairs evaluated. Output written to: {output_path}")

    significant = results[
        (results["r"] < config.CORRELATION_THRESHOLD) & (results["p"] < config.PVALUE_THRESHOLD)
    ]
    print(
        f"{len(significant)} pairs pass the significance filter "
        f"(r < {config.CORRELATION_THRESHOLD}, p < {config.PVALUE_THRESHOLD}):"
    )
    print(significant.sort_values("r").to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirna", nargs="+", required=True, help="One or more miRNA identifiers.")
    parser.add_argument("--mirna-file", type=Path, default=config.TUMOR_MIRNA_FILE)
    parser.add_argument("--mrna-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument(
        "--candidate-genes", type=Path, default=None,
        help="Optional CSV with a 'gene' column restricting the genes tested. "
             "If omitted, every gene in the expression table is tested.",
    )
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "mirna_mrna_correlations.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.mirna, args.mirna_file, args.mrna_file, args.candidate_genes, args.output)
