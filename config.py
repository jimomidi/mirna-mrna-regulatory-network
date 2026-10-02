"""
Central configuration for the miRNA-mRNA regulatory network analysis pipeline.

Every script in this repository imports its default file paths and analysis
parameters from this module. Edit the values below (or override them with
command-line arguments, where a script accepts them) to point the pipeline
at your own dataset.

Directory layout expected by default
-------------------------------------
data/
    expression/
        tumor_expression.csv        Tumor-sample mRNA quantification
                                     (gene_id, gene_name, case_id, tpm_unstranded,
                                     fpkm_unstranded, ...) - TCGA STAR-Counts style.
        normal_expression.csv       Normal/reference-tissue mRNA quantification in
                                     wide format (gene identifier column, one column
                                     per sample) - GTEx-style export.
        tumor_mirna.csv             Tumor-sample miRNA quantification
                                     (miRNA_ID, case_id, reads_per_million_miRNA_mapped,
                                     read_count).
    metadata/
        clinical.csv                Per-case clinical/metadata table (case_id,
                                     age_at_diagnosis, gender, ajcc_pathologic_stage,
                                     vital_status, days_to_death,
                                     days_to_last_follow_up, treatment_or_therapy,
                                     treatment_type).
    interactions/
        interaction_matrix_tumor.csv    Predicted miRNA-target interaction counts
        interaction_matrix_normal.csv   (Var1/miRNA_ID, Var2/gene_id, Freq/weight),
                                         e.g. exported from TargetScan / miRDB /
                                         miRTarBase target-prediction databases.
    gene_sets/
        Arbitrary two-column ("gene") CSV files used by the overlap utilities.
results/
    Output directory used by every script (figures and derived tables).

All paths below are relative to the repository root, so the pipeline runs
unchanged on any machine as long as the expected input files are supplied.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

EXPRESSION_DIR = DATA_DIR / "expression"
METADATA_DIR = DATA_DIR / "metadata"
INTERACTIONS_DIR = DATA_DIR / "interactions"
GENE_SETS_DIR = DATA_DIR / "gene_sets"

# ---------------------------------------------------------------------------
# Default input files (override on the command line where supported)
# ---------------------------------------------------------------------------
TUMOR_EXPRESSION_FILE = EXPRESSION_DIR / "tumor_expression.csv"
NORMAL_EXPRESSION_FILE = EXPRESSION_DIR / "normal_expression.csv"
TUMOR_MIRNA_FILE = EXPRESSION_DIR / "tumor_mirna.csv"
CLINICAL_FILE = METADATA_DIR / "clinical.csv"

# ---------------------------------------------------------------------------
# Genes / miRNAs of interest
# ---------------------------------------------------------------------------
# Populate these with the symbols relevant to your own study. Every script
# that needs a gene or miRNA list reads it from here by default, but also
# accepts an explicit list so the module can be reused programmatically.
GENES_OF_INTEREST = ["GENE_A", "GENE_B", "GENE_C"]
MIRNAS_OF_INTEREST = ["hsa-mir-example-1", "hsa-mir-example-2"]

# ---------------------------------------------------------------------------
# Preprocessing / filtering thresholds
# (kept identical to the exploratory analysis this repository was
# generalized from - see README "Reproducibility" section)
# ---------------------------------------------------------------------------
LOW_EXPRESSION_TPM_THRESHOLD = 1.0     # minimum expression to count a sample as "expressed"
MAX_LOW_EXPRESSION_FRACTION = 0.8      # drop a gene/miRNA if it falls below the threshold
                                        # in more than this fraction of samples
MIN_NONZERO_FRACTION = 0.2             # alternative "at least expressed in X% of samples" filter
PSEUDOCOUNT = 0.01                     # additive pseudocount used before log2 transforms

# Differential expression / correlation thresholds
LOGFC_THRESHOLD = 1.0
PVALUE_THRESHOLD = 0.05
CORRELATION_THRESHOLD = -0.3           # target-filtering correlation cutoff (miRNA vs mRNA)
NETWORK_CORRELATION_THRESHOLD = -0.1   # looser cutoff used when building the regulatory network
MIN_PAIRED_SAMPLES = 3                 # minimum paired samples required to compute a correlation

# Predicted-interaction confidence filter (network construction)
MIN_INTERACTION_WEIGHT = 5

# Random seed used throughout (cross-validation, SMOTE, hyper-parameter search)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
FONT_FAMILY = "Times New Roman"
FIGURE_DPI = 300


def ensure_results_dir() -> Path:
    """Create the results directory if it does not already exist and return it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
