"""
Filter a target-prediction database (e.g. TargetScan "Summary_Counts" export)
down to the predicted targets of a single miRNA that pass a context++ score
threshold, and write the deduplicated (miRNA, target gene) pairs to CSV.

The scoring column and threshold match a TargetScan-style
"Total context++ score" export, where more negative scores indicate a
stronger predicted repressive interaction.

Usage
-----
    python -m preprocessing.mirna_target_prediction_filter \
        --input data/interactions/targetscan_predictions.csv \
        --mirna hsa-mir-example-1 \
        --score-threshold -0.2 \
        --output results/mirna_example_1_targets.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config


def filter_predicted_targets(
    predictions: pd.DataFrame,
    mirna: str,
    score_threshold: float,
    mirna_col: str = "Representative miRNA",
    gene_col: str = "Gene Symbol",
    score_col: str = "Total context++ score",
) -> pd.DataFrame:
    """Return unique (miRNA, gene) pairs for `mirna` with score < `score_threshold`."""
    subset = predictions[predictions[mirna_col] == mirna].copy()
    subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce")
    subset = subset[subset[score_col] < score_threshold]
    return subset[[mirna_col, gene_col]].drop_duplicates()


def run(input_path: Path, mirna: str, score_threshold: float, output_path: Path) -> None:
    predictions = pd.read_csv(input_path)
    targets = filter_predicted_targets(predictions, mirna, score_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_path, index=False)
    print(f"{len(targets)} predicted targets for {mirna} (score < {score_threshold})")
    print(f"Output written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Target-prediction CSV export.")
    parser.add_argument("--mirna", type=str, required=True, help="miRNA identifier to filter on.")
    parser.add_argument(
        "--score-threshold", type=float, default=-0.2,
        help="Keep predictions with a context++ score below this value (default: -0.2).",
    )
    parser.add_argument(
        "--output", type=Path, default=config.RESULTS_DIR / "mirna_targets.csv",
        help="Where to write the filtered target list.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.mirna, args.score_threshold, args.output)
