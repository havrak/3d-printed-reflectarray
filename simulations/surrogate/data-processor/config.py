"""
Configuration for the data processor.

Everything tunable lives here: paths, the design frequency, the clearance rule,
the eigenbasis search, and the selection of active features.

No transform function is defined in this module. The geometry-derived ones live
in `features.py` and the band-derived ones in `band_features.py`; here they are
only named and ordered.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Speed of light
# --------------------------------------------------------------------------- #

C0_MM_GHZ = 299.792458  # mm * GHz, so lambda0[mm] = C0_MM_GHZ / f0[GHz]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Directory that contains one subdirectory per CST sweep; each subdirectory
# holds the .s2p files plus a result_navigator.csv.
DEFAULT_DATA_ROOT = Path("simulations/data/pillar/later_simulations")

# Where the processed dataset is written.
DEFAULT_OUTPUT = Path("simulations/surrogate/data-processor/out/dataset.npz")


# --------------------------------------------------------------------------- #
# Design point
# --------------------------------------------------------------------------- #

F0_GHZ = 30.0
LAMBDA0_MM = C0_MM_GHZ / F0_GHZ


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

# Minimum gap between the rotated ellipse's bounding box and the cell edge, mm.
# This clearance should be way smaller then actually clearance used in the design
# The model should achieve good performance on the edges of the interval, even though
# they won't really be used in the final model
#
# as CST is modeling elipse with a polygon setting to zero, could introducer some samples
# where the geometry actually overflowed the base size
CLEARANCE_MM = 0.01

# Ellipse major axis is rotated by this angle relative to the array axes.
ELLIPSE_ROTATION_DEG = 45.0


# --------------------------------------------------------------------------- #
# Eigenbasis search and validation
# --------------------------------------------------------------------------- #

# Angular scan step for the congruence transform, degrees.
THETA_SCAN_STEP_DEG = 0.5

# A cell is only in its eigenbasis if the recovered angle sits near the
# geometric rotation. Deviations mean the diagonal symmetry is broken.
THETA_STAR_TOL_DEG = 10.0

# Reject samples whose residual (linear, summed off-diagonal magnitude)
# exceeds this. Mainly completeness; the relative residual filter below is
# the one that actually discriminates.
RESIDUAL_MAX_LINEAR = 0.05

# Samples whose aspect ratio is within this of unity are treated as circular.
# The cross-section is then rotationally symmetric, so the reflection operator
# is isotropic and the eigenbasis is degenerate: theta* is decided by
# floating-point noise and must not be used as a validity test.
CIRCULAR_TOL = 1e-3


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
#
# The functions themselves live in features.py (geometry-derived) and
# band_features.py (band-derived). Only the selection and the column names are
# declared here, so activating or deactivating a feature is a one-line change
# and no transform is defined in this module.

# Geometry-derived features: (column name, function from features.py).
# Order defines the column order of X.
FEATURE_TRANSFORMS = [
    # cell and ellipse geometry
    ("p_ratio",             "feat_p_ratio"),
    ("a_over_P",            "feat_a_over_P"),
    ("fill_fraction",       "feat_fill_fraction"),
    # electrical heights
    ("h_pillar_electrical", "feat_h_pillar_electrical"),
    ("h_base_electrical",   "feat_h_base_electrical"),
    # material and anisotropy
    ("ratio",               "feat_ratio"),
    ("eps_r",               "feat_eps_r"),
    # effective-medium derived
    ("eps_eff_major",       "feat_eps_eff_major"),
    ("eps_eff_minor",       "feat_eps_eff_minor"),
]

# Band-derived features: (column name, function from band_features.py).
# Each takes a BandContext and returns one scalar. All three are computed so
# they are available as features and as validation columns; individual ones can
# be excluded later, before training, without reprocessing.
BAND_FEATURE_TRANSFORMS = [
    ("dphi_df",             "feat_dphi_df"),
    ("band_spread",         "feat_band_spread"),
    ("resonance_in_band",   "feat_resonance_in_band"),
]


# --------------------------------------------------------------------------- #
# Sample blacklist
# --------------------------------------------------------------------------- #
#
# Exclude groups of sweep configurations from the dataset entirely, before any
# loading or validation work is done.
#
# Each entry is a dict mapping a raw parameter name to a value:
#
#     {"P_ratio": 0.5, "epsilon_r": 7.5}   # that exact combination only
#     {"P_ratio": 0.5, "epsilon_r": "*"}   # any permittivity at P_ratio = 0.5
#
# A parameter that is absent from a dict, or mapped to "*", matches anything.
# All entries in a dict must match for the sample to be excluded, so a sample
# is dropped when it satisfies at least one dict completely.
#
# Recognised keys are the raw CSV parameter names: "a", "ratio", "P_ratio",
# "epsilon_r", "H_pillar", "H_base". Matching is by exact float equality, which
# is safe because these values come straight from the CST sweep table rather
# than from arithmetic. The same rules are applied by data-statistics so the
# two tools always see the same population.

BLACKLIST: list[dict[str, float | str]] = [
    # Examples, left inactive by default:
    # {"P_ratio": 0.5, "epsilon_r": 7.5},
    {"P_ratio": 0.5, "epsilon_r": "*"},
]


# --------------------------------------------------------------------------- #
# Band descriptor parameters
# --------------------------------------------------------------------------- #

BAND_SLOPE_DF_GHZ = 0.1     # half-width of the least-squares slope window, GHz
RESONANCE_DIP_DB = -3.0     # |S| below this counts as a resonance, dB


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

OUTPUT_FORMAT = "npz"       # "npz" | "csv"
