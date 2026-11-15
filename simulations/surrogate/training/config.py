"""
Configuration for the split generator.

Everything tunable lives here: grouping key, stratification, holdout fraction,
fold count, and the optional filtering of samples whose band contains a
resonance.

The split generator consumes a `dataset.npz` produced by `data-processor/` and
writes split files next to it. It does not modify the dataset.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

# Default dataset path, used when --dataset is not given. Relative to the
# repository root.
DEFAULT_DATASET = Path(
    "simulations/surrogate/data-processor/out/dataset.npz"
)


# --------------------------------------------------------------------------- #
# Optional filtering
# --------------------------------------------------------------------------- #
#
# The dataset carries a `resonance_in_band` column in its `band` array. Samples
# flagged 1 have a |S| notch inside the simulated band, which means their phase
# slope is not a smooth function of geometry. They remain valid design points
# but they are poor regression targets for the slope, so they can be excluded.
#
# Filtering happens here rather than in the data processor so the full dataset
# stays intact on disk and the choice remains reversible.

FILTER_RESONANCE_IN_BAND = False

# Column name inside the dataset's `band` array that carries the flag.
RESONANCE_FLAG_COLUMN = "resonance_in_band"


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
#
# Splitting MUST be done at the geometry level, not the sample level. Each
# geometry is a sweep along `a` at fixed (P_ratio, eps_r, H_pillar, ratio), so a
# sample-level split would place different `a` values of the same geometry on
# both sides and the test score would measure interpolation along one axis
# inside a configuration the model has already seen. That is not what the
# surrogate is for.
#
# The group key is therefore the full fixed-parameter tuple. Column names refer
# to the dataset's `X_raw` array.

GROUP_COLUMNS = ["p_ratio", "eps_r", "h_pillar_mm", "ratio"]


# --------------------------------------------------------------------------- #
# Holdout split
# --------------------------------------------------------------------------- #

# Fraction of *geometries* (not samples) held out for final evaluation.
HOLDOUT_FRACTION = 0.20

# Stratify the holdout on these columns so every (P_ratio, eps_r) cell is
# represented on both sides and per-cell error can be reported.
HOLDOUT_STRATIFY_COLUMNS = ["p_ratio", "eps_r"]

HOLDOUT_SEED = 0


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

CV_FOLDS = 5

# Stratify folds on the same (P_ratio, eps_r) cells. Stratifying on height as
# well would be degenerate: 28 height values against 342 geometries leaves an
# average of ~2.4 geometries per stratum, too few to populate every fold. The
# consequence is that folds are balanced in cell composition but the model is
# always tested on heights it has seen, so the CV estimate is a best case.
CV_STRATIFY_COLUMNS = ["p_ratio", "eps_r"]

CV_SEED = 0


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

# Split files are written next to the dataset, in its own directory.
HOLDOUT_FILENAME = "split_holdout.npz"
CV_FILENAME_TEMPLATE = "split_cv_k{folds}.npz"
