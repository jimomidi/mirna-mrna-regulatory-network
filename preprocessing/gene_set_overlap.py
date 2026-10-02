"""
Gene set overlap utility.

Computes pairwise and multi-way overlaps between two or more gene-symbol
lists (each supplied as a CSV file with a "gene" column), and optionally
checks whether a separate candidate gene list is present in the union of
the reference sets. This generalizes the ad hoc two-way / three-way set
intersections used throughout the exploratory analysis (e.g. comparing a
predicted-target list against a differential-expression gene list, or
checking overlap between three co-expression module gene lists) into a
single reusable function.

Usage
-----
    python -m preprocessing.gene_set_overlap \
        --input data/gene_sets/set_a.csv data/gene_sets/set_b.csv \
        --output results/gene_overlap.csv

    python -m preprocessing.gene_set_overlap \
        --input data/gene_sets/set_a.csv data/gene_sets/set_b.csv \
        --candidates data/gene_sets/candidate_genes.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

import config


def load_gene_set(path: Path, column: str = "gene") -> set:
    """Read a single-column CSV of gene symbols into a cleaned, upper-cased set."""
    df = pd.read_csv(path)
    return set(df[column].dropna().astype(str).str.strip().str.upper())


def pairwise_overlaps(gene_sets: Dict[str, set]) -> pd.DataFrame:
    """Return a table of pairwise intersection sizes for every pair of gene sets."""
    rows = []
    for (name_a, set_a), (name_b, set_b) in combinations(gene_sets.items(), 2):
        rows.append(
            {
                "set_a": name_a,
                "set_b": name_b,
                "n_a": len(set_a),
                "n_b": len(set_b),
                "n_overlap": len(set_a & set_b),
            }
        )
    return pd.DataFrame(rows)


def full_overlap(gene_sets: Dict[str, set]) -> set:
    """Return the intersection shared by every gene set supplied."""
    sets = list(gene_sets.values())
    if not sets:
        return set()
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return common


def check_candidates(candidates: Sequence[str], gene_sets: Dict[str, set]) -> pd.DataFrame:
    """Report, for each candidate gene, whether it appears in each reference set."""
    union = set().union(*gene_sets.values()) if gene_sets else set()
    rows = []
    for gene in candidates:
        gene_upper = str(gene).strip().upper()
        row = {"gene": gene, "found_in_any_set": gene_upper in union}
        for name, genes in gene_sets.items():
            row[f"in_{name}"] = gene_upper in genes
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    input_paths: List[Path],
    output_path: Path,
    candidates_path: Path | None = None,
) -> None:
    gene_sets = {path.stem: load_gene_set(path) for path in input_paths}

    for name, genes in gene_sets.items():
        print(f"{name}: {len(genes)} unique genes")

    pairwise = pairwise_overlaps(gene_sets)
    print("\nPairwise overlaps:")
    print(pairwise.to_string(index=False))

    if len(gene_sets) > 2:
        common = full_overlap(gene_sets)
        print(f"\nOverlap across all {len(gene_sets)} sets: {len(common)} genes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_path, index=False)
    print(f"\nPairwise overlap table written to: {output_path}")

    if candidates_path is not None:
        candidate_genes = pd.read_csv(candidates_path)["gene"].dropna().tolist()
        report = check_candidates(candidate_genes, gene_sets)
        candidates_out = output_path.with_name(output_path.stem + "_candidate_membership.csv")
        report.to_csv(candidates_out, index=False)
        print(f"Candidate membership report written to: {candidates_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True, type=Path,
        help="Two or more CSV files, each with a 'gene' column, to compare.",
    )
    parser.add_argument(
        "--output", type=Path, default=config.RESULTS_DIR / "gene_overlap.csv",
        help="Where to write the pairwise overlap summary table.",
    )
    parser.add_argument(
        "--candidates", type=Path, default=None,
        help="Optional CSV with a 'gene' column to test for membership in the input sets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.candidates)
