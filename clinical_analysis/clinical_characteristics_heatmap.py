"""
Per-case clinical characteristics heatmap (gender, age group, AJCC stage,
vital status) for a treatment-defined case group, plus a standalone legend
figure for the categorical color scales.

Case groups are derived from the clinical table's `treatment_type` /
`treatment_or_therapy` columns using the same logic as
`treatment_group_heatmap.py`, so the two analyses stay consistent.

Usage
-----
    python -m clinical_analysis.clinical_characteristics_heatmap \
        --primary-treatment Pharmaceutical --secondary-treatment Radiation \
        --output-prefix results/clinical_characteristics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

import config
import utils

COLOR_MAPS = {
    "gender": {"male": "#87CEEB", "female": "#FF69B4"},
    "age_category": {"<33": "#E0BBE4", "33-48": "#C89BCF", "48-62": "#A678B4", ">62": "#703F7E"},
    "ajcc_pathologic_stage": {
        "Stage I": "#FFB6B6", "Stage II": "#FF7F7F", "Stage III": "#FF4C4C", "Stage IV": "#B22222",
    },
    "vital_status": {"Alive": "#228B22", "Dead": "#A9A9A9"},
}
DISPLAY_LABELS = ["Gender", "Age Group", "AJCC Stage", "Vital Status"]


def categorize_age(age_years: float) -> str | None:
    if pd.isna(age_years):
        return None
    if age_years < 33:
        return "<33"
    if age_years < 48:
        return "33-48"
    if age_years < 62:
        return "48-62"
    return ">62"


def build_clinical_table(clinical: pd.DataFrame, case_ids: list[str]) -> pd.DataFrame:
    cols = ["case_id", "gender", "age_at_diagnosis", "ajcc_pathologic_stage", "vital_status"]
    df = clinical[clinical["case_id"].isin(case_ids)][cols].drop_duplicates(subset="case_id")
    df["age_at_diagnosis"] = df["age_at_diagnosis"] / 365.25
    df["age_category"] = df["age_at_diagnosis"].apply(categorize_age)
    df["case_id"] = pd.Categorical(df["case_id"], categories=case_ids, ordered=True)
    return df.sort_values("case_id")


def plot_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    columns = ["gender", "age_category", "ajcc_pathologic_stage", "vital_status"]
    plot_df = df.set_index("case_id")[columns]

    fig, ax = plt.subplots(figsize=(3, max(4, 0.25 * len(plot_df))))
    for i, case in enumerate(plot_df.index):
        for j, col in enumerate(plot_df.columns):
            value = plot_df.loc[case, col]
            color = COLOR_MAPS[col].get(value, "#FFFFFF")
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white"))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(0, len(plot_df.columns))
    ax.set_ylim(0, len(plot_df))
    ax.set_xticks([x + 0.5 for x in range(len(plot_df.columns))])
    ax.set_xticklabels(DISPLAY_LABELS, rotation=90, ha="center", fontsize=14, fontweight="bold")
    ax.set_yticks([y + 0.5 for y in range(len(plot_df.index))])
    ax.set_yticklabels(plot_df.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to: {output_path}")


def plot_legend(output_path: Path) -> None:
    legend_elements = [
        Patch(facecolor="#87CEEB", label="Male"),
        Patch(facecolor="#FF69B4", label="Female"),
        Patch(facecolor="#E0BBE4", label="Age < 33"),
        Patch(facecolor="#C89BCF", label="Age 33-48"),
        Patch(facecolor="#A678B4", label="Age 48-62"),
        Patch(facecolor="#703F7E", label="Age > 62"),
        Patch(facecolor="#FFB6B6", label="Stage I"),
        Patch(facecolor="#FF7F7F", label="Stage II"),
        Patch(facecolor="#FF4C4C", label="Stage III"),
        Patch(facecolor="#B22222", label="Stage IV"),
        Patch(facecolor="#228B22", label="Alive"),
        Patch(facecolor="#A9A9A9", label="Dead"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="center", ncol=6, frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Legend figure written to: {output_path}")


def run(
    clinical_file: Path,
    primary_treatment: str,
    secondary_treatment: str,
    output_prefix: Path,
) -> None:
    utils.set_plot_style()
    clinical = utils.load_clinical(clinical_file)

    group_a_cases = sorted(utils.cases_by_treatment(clinical, contains=primary_treatment, also_received=secondary_treatment))
    group_b_cases = sorted(utils.cases_by_treatment(clinical, contains=primary_treatment, exclude=secondary_treatment))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_a = build_clinical_table(clinical, group_a_cases)
    plot_heatmap(df_a, f"{secondary_treatment} & {primary_treatment} Therapy", output_prefix.with_name(output_prefix.name + "_group_a.png"))

    df_b = build_clinical_table(clinical, group_b_cases)
    plot_heatmap(df_b, f"{primary_treatment} Therapy Only", output_prefix.with_name(output_prefix.name + "_group_b.png"))

    plot_legend(output_prefix.with_name(output_prefix.name + "_legend.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-file", type=Path, default=config.CLINICAL_FILE)
    parser.add_argument("--primary-treatment", type=str, default="Pharmaceutical")
    parser.add_argument("--secondary-treatment", type=str, default="Radiation")
    parser.add_argument("--output-prefix", type=Path, default=config.RESULTS_DIR / "clinical_characteristics")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.clinical_file, args.primary_treatment, args.secondary_treatment, args.output_prefix)
