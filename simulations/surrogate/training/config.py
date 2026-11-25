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

# Exclude the circular cross-section (ratio = 1) from the reported statistics.
#
# At ratio = 1 the two axes are physically identical, so the inter-axis offset is
# the difference of two nearly equal quantities and is ill-conditioned: a small
# error in either phase produces a large relative error in the difference, and a
# value near the 0/360 boundary flips sign under a tiny perturbation. The offset
# is identically zero there, so those cells can never satisfy the quadrature
# condition and are not candidate CP elements.
#
# They are retained in TRAINING so the model interpolates correctly down to
# ratio = 1, and excluded only from the reported metrics.
EXCLUDE_RATIO_1 = True

# Aspect ratios within this tolerance of unity count as circular.
CIRCULAR_RATIO_TOL = 1e-3


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

# Strata are defined by hand rather than derived from the columns, so the
# granularity can be chosen to suit the data. Each entry is a named stratum and
# a dict of the column values that belong to it. A sample falls in the first
# stratum whose every condition matches.
#
# Why hand-written: the small cells carry only a dozen geometries each, and
# treating them separately leaves about two geometries per fold, which makes the
# fold-to-fold spread reflect the split rather than the model. Pooling adjacent
# permittivities restores a workable group count. Grouping itself still uses the
# exact geometry tuple, so no leakage is introduced; only the granularity of the
# fold assignment changes.
#
# The trade is explicit: a pooled stratum is balanced in total size but not
# necessarily in composition, so a fold may receive more eps_r = 5 than 6. With
# a dozen geometries, balance matters more than per-cell exactness.
#
# Keep the strata exhaustive over the dataset. Any sample matching no stratum
# is reported and assigned to a fallback stratum, so nothing is silently lost.

STRATA = [
    {"name": "P0.4_er4.5",  "p_ratio": 0.4, "eps_r": 4.5},
    {"name": "P0.4_er5-6",  "p_ratio": 0.4, "eps_r": (5.0, 6.0)},
    {"name": "P0.4_er7",    "p_ratio": 0.4, "eps_r": 7.0},
    {"name": "P0.4_er7.5",  "p_ratio": 0.4, "eps_r": 7.5},
    {"name": "P0.5_er4.5",  "p_ratio": 0.5, "eps_r": 4.5},
]

# Columns used to evaluate the strata above. Listed so the split files can
# record what the strata were built from.
STRATA_COLUMNS = ["p_ratio", "eps_r"]

HOLDOUT_SEED = 0


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

CV_FOLDS = 5

# Folds use the same strata as the holdout. Stratifying on height as well would
# be degenerate: many height values against a few hundred geometries leaves too
# few groups per stratum. The consequence is that folds are balanced in cell
# composition but the model is always tested on heights it has seen, so the CV
# estimate is a best case with respect to height extrapolation. That is
# acceptable because the surrogate is queried at heights inside the simulated
# range.
CV_SEED = 0


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
#
# Z-scoring is a fit: it estimates a mean and standard deviation from data. It
# must therefore be fitted on the TRAINING rows of each split and applied to the
# validation rows, never fitted on the whole dataset. Fitting globally would
# encode the validation set's location and spread in the scaling coefficients,
# which is leakage of a subtle kind: the model never sees the validation labels,
# but every prediction it makes is shifted by knowledge of where those inputs
# lie.
#
# Because each CV fold has a different training set, each fold needs its own
# scaler. The split generator therefore writes one coefficient set per split,
# not one per dataset.
#
# The scaling itself is deliberately not applied here. The split generator
# writes the coefficients; the training code applies them, so the same
# coefficients are used for every model in a deep ensemble and the scaling is
# recorded alongside the split rather than being implicit in the pipeline.

NORMALISE_INPUTS = True
NORMALISE_TARGETS = True

# Numerical floor on the standard deviation, so a feature that happens to be
# constant within a training split (H_base, while it is fixed at 0.6 mm) does
# not produce a division by zero. Such a feature becomes a zero column after
# scaling, which is harmless: the network's bias weight covers it.
STD_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# Hyperparameter search space
# --------------------------------------------------------------------------- #
#
# Declarative description of the Optuna search space. Each entry is
#   name: {"type": "int" | "float" | "categorical", ...}
# with "log": True for quantities spanning orders of magnitude. Translating this
# into Optuna suggest calls happens in cv_tune.py, so the space can be edited
# here without touching the tuning logic.
#
# Dropout is deliberately absent: the uncertainty comes from a deep ensemble,
# which requires each member to be a well-trained deterministic function.
# Batch normalisation is absent for the same reason it is absent from the model:
# its running statistics misbehave on inputs far from the training distribution,
# which is exactly the extrapolation regime that matters here.

SEARCH_SPACE = {
    "n_layers":     {"type": "int", "low": 2, "high": 6},
    "width":        {"type": "int", "low": 64, "high": 512, "log": True},
    "activation":   {"type": "categorical", "choices": ["SiLU", "ELU", "GELU"]},
    "lr":           {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
}

# Loss weights are fixed, NOT searched.
#
# They must not be in the search space, because Optuna minimises the total loss
# and the weights are part of it. A trial that samples a small weight gets a
# small total regardless of how well it fits, so the search would drive both
# weights to their lower bounds and the "best" configuration would be the one
# that turned the penalties off. The weights are a statement about what matters,
# not a quantity to be optimised.
#
# w_slope weights the phase-slope target against the complex coefficients. The
# coefficients are the primary objective, so this is kept small: it makes the
# slope a soft regulariser rather than a competing target.
W_SLOPE = 0.1

# lambda_pass weights the passivity penalty. The data are passive by
# construction, so a well-fit model has zero penalty and this only bites on
# genuine violations. Kept small for that reason.
LAMBDA_PASS = 0.1

# The slope target is heavy-tailed: it is a derivative, so it blows up near
# resonance while the bulk of configurations sit near zero. Squared error on such
# a distribution is dominated by a handful of points. Clipping the standardised
# slope target at this value bounds the influence of those outliers without
# discarding them. Set to None to disable.
SLOPE_CLIP_SIGMA = 5.0

# Batch size is fixed rather than searched.
#
# It is tempting to tune it, but with a fixed epoch budget batch size and epoch
# count are two names for the same thing: the amount of training is
# steps_per_epoch x epochs, and steps_per_epoch is set by the batch size. Leaving
# both free means Optuna trades one against the other and the resulting
# "optimum" reflects the epoch cap rather than the problem.
#
# Fixing it also removes a source of unnecessary variance. The model is small,
# so a step on a GPU is dominated by kernel-launch overhead rather than
# arithmetic; a large batch amortises that and the gradient is smooth enough
# that the extra noise from small batches buys nothing. 1024 is large enough to
# keep the GPU busy and small enough to give a useful number of steps per epoch.
BATCH_SIZE = 1024

# Number of Optuna trials. Each trial trains on every fold, so the cost is
# trials x folds x epochs.
N_TRIALS = 100

# Pruning: abandon a trial whose fold is clearly worse than the median of
# completed trials at the same epoch. Set to 0 to disable.
PRUNER_STARTUP_TRIALS = 10
PRUNER_WARMUP_EPOCHS = 30


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
#
# With BATCH_SIZE fixed, an epoch is a well-defined unit of training again: it
# is one pass over the training portion, and the number of steps per epoch is
# the same for every trial. That is what makes an epoch budget meaningful and
# why batch size is not in the search space.
#
# The counts below assume of order 10^4 training rows, giving roughly ten steps
# per epoch at BATCH_SIZE. The dataset is small, so a fold reaches its best
# validation loss within a few hundred epochs even with a conservative learning
# rate; the budget is generous enough that early stopping, not the cap, decides
# when training ends.

MAX_EPOCHS = 400

# Stop a fold when the validation loss has not improved for this many epochs.
EARLY_STOP_PATIENCE = 40

# Minimum relative improvement counted as progress, so noise does not reset the
# patience counter.
EARLY_STOP_MIN_DELTA = 1e-5

# Fixed for reproducibility of a trial.
TRAIN_SEED = 0

# Reported in the log.
LOG_EVERY_N_EPOCHS = 25


# Tolerance on the inter-axis phase offset, in degrees. A prediction whose
# predicted offset is within this of the true offset is counted as usable.
DPHI_TOL_DEG = 5.0


# --------------------------------------------------------------------------- #
# Deep ensemble
# --------------------------------------------------------------------------- #
#
# The ensemble is trained on the full training portion using the tuned
# hyperparameters, then evaluated once on the holdout. Members differ only in
# their random initialisation and data ordering, so the spread across members is
# an epistemic uncertainty: small where the training data are dense, large where
# the model is extrapolating.

# Number of members. Ten rather than five because the spread is the point: the
# sample standard deviation from M members carries a relative error of roughly
# 1/sqrt(2(M-1)), so M = 5 gives ~35% error on sigma while M = 10 gives ~24%.
# If sigma is used to decide where to simulate next, that difference matters.
# Cost is low: members are small and train in seconds.
N_ENSEMBLE = 10

# How each member's training rows are chosen.
#
#   "fixed"      every member sees the whole training portion, differing only in
#                seed and batch order. Captures optimisation randomness. Useful
#                as a baseline: comparing it against the other modes shows how
#                much the spread grows once data variability is included.
#   "subsample"  a fraction ENSEMBLE_COVERAGE of the GEOMETRIES, drawn without
#                replacement. Each member misses a different ~20%, so the spread
#                picks up data variability while retaining most of the data.
#   "bootstrap"  geometries drawn with replacement, so ~63% appear per member and
#                some appear twice. The most diversity and the most data loss.
#
# The resampling unit is the geometry, never the individual row. Within one
# geometry the values along `a` are samples of a single smooth curve, so they
# are not independent observations: resampling rows would leave every geometry
# represented in every member, differing only in which points along each curve
# were dropped, and the members would be nearly identical. The independent unit
# for "how uncertain am I about a configuration I have not seen" is the geometry.
ENSEMBLE_SAMPLING = "subsample"     # "fixed" | "subsample" | "bootstrap"

# Fraction of geometries used by "subsample". Ignored by the other modes.
ENSEMBLE_COVERAGE = 0.8

# Base seed for member seeds. Member m uses BASE + m * STRIDE, so the number of
# members can be changed without editing a list.
ENSEMBLE_BASE_SEED = 101
ENSEMBLE_SEED_STRIDE = 101

# Scaling coefficients for the ensemble are fitted on the whole training
# portion, not per fold, because there are no folds at this stage. The holdout
# was not part of that fit, so the rule "coefficients come from training data"
# still holds.
ENSEMBLE_SCALER_NAME = "ensemble"


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

# Split files are written next to the dataset, in its own directory.
HOLDOUT_FILENAME = "split_holdout.npz"
CV_FILENAME_TEMPLATE = "split_cv_k{folds}.npz"

# Tuning outputs, also written next to the dataset.
BEST_PARAMS_FILENAME = "cv_best_params.npz"
TRIALS_CSV_FILENAME = "cv_trials.csv"
TUNE_LOG_FILENAME = "cv_tuning.log"

# Ensemble outputs.
ENSEMBLE_MODEL_FILENAME = "ensemble.pt"
ENSEMBLE_META_FILENAME = "ensemble_meta.npz"
ENSEMBLE_LOG_FILENAME = "ensemble_train.log"

# Evaluation outputs, written by evaluate.py.
EVAL_SUMMARY_FILENAME = "eval_summary.csv"
EVAL_STRATA_FILENAME = "eval_by_stratum.csv"
EVAL_RATIO_FILENAME = "eval_by_ratio.csv"
EVAL_A_FILENAME = "eval_by_a.csv"
EVAL_CALIBRATION_FILENAME = "eval_calibration.csv"
EVAL_REPORT_FILENAME = "eval_report.txt"
EVAL_REPORT_FULL_FILENAME = "eval_report_full.txt"

# Number of bins along the tuning variable for the extrapolation diagnostic.
# More bins resolve the edges better but leave fewer samples per bin.
EVAL_A_BINS = 12
