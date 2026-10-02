"""
Shared helper functions for loading and preprocessing expression data.

These helpers factor out the data-wrangling steps that were duplicated
across the original exploratory notebooks (parsing the tumor/normal
expression tables, filtering low-expression features, and computing
log2-transformed values). The transformations themselves are unchanged
from the source analysis; only the duplicated boilerplate was consolidated.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config


def load_tumor_expression(path=config.TUMOR_EXPRESSION_FILE) -> pd.DataFrame:
    """Load a TCGA-style tumor expression table and strip Ensembl version suffixes."""
    df = pd.read_csv(path)
    df["gene_id"] = df["gene_id"].astype(str).str.replace(r"\..*", "", regex=True)
    return df


def load_normal_expression(path=config.NORMAL_EXPRESSION_FILE) -> pd.DataFrame:
    """Load a GTEx-style wide-format normal/reference-tissue expression table."""
    return pd.read_csv(path)


def normal_expression_matrix(
    df_normal: pd.DataFrame,
    genes: Iterable[str],
    gene_col: str = "Description",
    drop_cols: Iterable[str] = ("Name",),
) -> pd.DataFrame:
    """Return a samples x genes matrix for the requested genes from a wide-format table."""
    subset = df_normal[df_normal[gene_col].isin(list(genes))].set_index(gene_col)
    subset = subset.drop(columns=[c for c in drop_cols if c in subset.columns])
    matrix = subset.T.apply(pd.to_numeric, errors="coerce")
    return matrix


def tumor_expression_matrix(
    df_tumor: pd.DataFrame,
    genes: Iterable[str],
    value_col: str = "tpm_unstranded",
) -> pd.DataFrame:
    """Return a case_id x gene_name matrix pivoted from a long-format tumor table."""
    subset = df_tumor[df_tumor["gene_name"].isin(list(genes))]
    return subset.pivot_table(index="case_id", columns="gene_name", values=value_col, aggfunc="mean")


def filter_low_expression(
    matrix: pd.DataFrame,
    threshold: float = config.LOW_EXPRESSION_TPM_THRESHOLD,
    max_low_fraction: float = config.MAX_LOW_EXPRESSION_FRACTION,
    axis: int = 1,
) -> pd.DataFrame:
    """
    Drop features (columns, by default) whose expression falls below `threshold`
    in more than `max_low_fraction` of samples.

    `axis=1` expects a samples x features matrix (drops low-expression columns).
    `axis=0` expects a features x samples matrix (drops low-expression rows).
    """
    if axis == 1:
        keep = (matrix < threshold).mean(axis=0) <= max_low_fraction
        return matrix.loc[:, keep]
    keep = (matrix < threshold).mean(axis=1) <= max_low_fraction
    return matrix.loc[keep, :]


def log2_transform(matrix: pd.DataFrame, pseudocount: float = config.PSEUDOCOUNT) -> pd.DataFrame:
    """log2(x + pseudocount) transform, matching the additive pseudocount used throughout."""
    return np.log2(matrix + pseudocount)


def load_clinical(path=config.CLINICAL_FILE) -> pd.DataFrame:
    """Load the per-case clinical/metadata table."""
    return pd.read_csv(path)


def cases_by_treatment(
    clinical: pd.DataFrame,
    contains: str,
    also_received: Optional[str] = None,
    exclude: Optional[str] = None,
) -> set:
    """
    Return the set of case IDs whose `treatment_type` matches `contains`
    (case-insensitive substring) and `treatment_or_therapy == "yes"`.

    `also_received` / `exclude` optionally intersect/subtract cases that also
    match another treatment-type substring, which is how the original
    analysis distinguished, e.g., "radiotherapy + drug therapy" cases from
    "drug therapy only" cases without hard-coding case ID lists.
    """
    received = clinical[
        (clinical["treatment_or_therapy"] == "yes")
        & (clinical["treatment_type"].str.contains(contains, case=False, na=False))
    ]["case_id"]
    cases = set(received)

    if also_received is not None:
        other = clinical[
            (clinical["treatment_or_therapy"] == "yes")
            & (clinical["treatment_type"].str.contains(also_received, case=False, na=False))
        ]["case_id"]
        cases &= set(other)

    if exclude is not None:
        other = clinical[
            (clinical["treatment_or_therapy"] == "yes")
            & (clinical["treatment_type"].str.contains(exclude, case=False, na=False))
        ]["case_id"]
        cases -= set(other)

    return cases


def set_plot_style() -> None:
    """Apply the shared font/style settings used across all figures."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = config.FONT_FAMILY


def get_cmap(name: str, n: int):
    """Return an `n`-color discrete colormap, using the modern matplotlib API
    (`matplotlib.colormaps`) where available and falling back to the older
    `cm.get_cmap` on older matplotlib versions."""
    import matplotlib as mpl

    try:
        return mpl.colormaps[name].resampled(n)
    except AttributeError:  # matplotlib < 3.6
        import matplotlib.cm as cm

        return cm.get_cmap(name, n)
