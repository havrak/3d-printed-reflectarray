# Training

Split generation for the surrogate model. Consumes a `dataset.npz` produced by
`../data-processor/` and writes split files alongside it.

## Validation

### splits.py
Filter applications looks legit, holdout is created at first and never used. splits use only training data.
Z-score normalization is also realized correctly - there isn't any leakage be it in validation or in testing



## Run

```bash
cd "simulations/surrogate/training"
python splits.py --dataset ../data-processor/out/dataset.npz --dry-run
python splits.py --dataset ../data-processor/out/dataset.npz
```

`--dry-run` prints the splits and the per-cell composition without writing.
Requires only numpy.

| flag | default | meaning |
|---|---|---|
| `--dataset` | `config.DEFAULT_DATASET` | path to `dataset.npz`; splits go in the same directory |
| `--holdout` | `0.20` | fraction of **geometries** held out |
| `--folds` | `5` | cross-validation folds |
| `--filter-resonance` | `config.FILTER_RESONANCE_IN_BAND` | drop samples whose band contains a resonance |

## Why splitting is grouped, not per-sample

Each dataset row is one `a` value belonging to one geometry, where a geometry is
a sweep along `a` at fixed `(P_ratio, eps_r, H_pillar, ratio)`. There are of
order 10^4 rows but only a few hundred geometries.

A per-sample split would place different `a` values of the same geometry on both
sides. The model would then be tested on a curve it had already trained on,
differing only along one axis, and the score would measure interpolation inside
a known configuration rather than the ability to predict an unseen one. Since
the surrogate exists to propose *new* geometries, that would test the wrong
thing.

All `a` values of a geometry therefore go to the same side. The group key is the
full fixed-parameter tuple, `GROUP_COLUMNS` in `config.py`.

## Why stratification is on (P_ratio, eps_r)

Grouping prevents leakage; stratification does not. Stratification only controls
how groups are distributed, so each fold is compositionally similar to the
others and the fold-to-fold spread reflects the model rather than the split.

The strata are written **by hand** in `config.STRATA`, one entry per cell:

```python
STRATA = [
    {"name": "P0.4_er4.5",  "p_ratio": 0.4, "eps_r": 4.5},
    {"name": "P0.4_er5-6",  "p_ratio": 0.4, "eps_r": (5.0, 6.0)},
    {"name": "P0.4_er7",    "p_ratio": 0.4, "eps_r": 7.0},
    {"name": "P0.4_er7.5",  "p_ratio": 0.4, "eps_r": 7.5},
    {"name": "P0.5_er4.5",  "p_ratio": 0.5, "eps_r": 4.5},
]
```

A value may be a scalar (exact match) or a tuple (membership). A sample belongs
to the first stratum whose every condition matches. Samples matching nothing are
reported and assigned to a fallback stratum, so nothing is silently lost.

Hand-writing them rather than deriving them from the columns exists for one
reason: the small permittivity cells carry only about a dozen geometries each,
which leaves roughly two geometries per fold and makes the fold-to-fold spread
reflect the split rather than the model. Pooling adjacent permittivities
(`eps_r = 5` with `6`) restores a workable group count. Grouping itself still
uses the exact geometry tuple, so no leakage is introduced; only the granularity
of the fold assignment changes.

The trade is explicit: a pooled stratum is balanced in total size but not
necessarily in composition, so a fold may receive more `eps_r = 5` than `6`.
With a dozen geometries, balance matters more than per-cell exactness.

Stratifying on height as well would be degenerate: many height values against a
few hundred geometries leaves too few groups per stratum. The consequence is
that folds always contain the heights the model is tested on, so the CV estimate
is a **best case with respect to height extrapolation**. That is acceptable
because the surrogate is queried at heights inside the simulated range.

If a cell is later dropped from the dataset, remove its entry from `STRATA`; the
remaining strata are matched by value, so no other change is needed.

## Cross-validated hyperparameter tuning

`cv_tune.py` searches the space declared in `config.SEARCH_SPACE` over the
folds in `split_cv_k{n}.npz`. The holdout is never touched: every fold's train
and validation rows come from the training portion only, and `splits.py`
guarantees no fold overlaps the holdout.

### Two inputs, and why

The script needs **both** the dataset and the split file, because they hold
different things:

| file | holds |
|---|---|
| `dataset.npz` | the arrays: `X`, `Y`, feature and target names |
| `split_cv_k{n}.npz` | row **indices** and scaling coefficients only |

The split file stores indices into the dataset, not the data itself, so it
cannot be used without the dataset it was generated from. This is deliberate: it
keeps the split file small, and it means the splits remain valid if you
regenerate the dataset without changing its row order.

`--folds-file` defaults to `split_cv_k{folds}.npz` in the dataset's directory,
which is where `splits.py` writes it. In the normal case only `--dataset` is
needed:

```bash
# check the plumbing without training
python cv_tune.py --dataset ../data-processor/out/dataset.npz --dry-run

# run the search
python cv_tune.py --dataset ../data-processor/out/dataset.npz --trials 100

# explicit split file, e.g. a different fold count
python cv_tune.py --dataset ../data-processor/out/dataset.npz \
                  --folds-file ../data-processor/out/split_cv_k10.npz
```

If the split file records a different number of folds than `--folds`, the file
wins and a note is printed — the file is the authority on what was actually
generated.

Requires `numpy`, `torch` and `optuna`.

| flag | default | meaning |
|---|---|---|
| `--dataset` | required | path to `dataset.npz` |
| `--folds-file` | `split_cv_k{folds}.npz` beside the dataset | split file to use |
| `--folds` | `config.CV_FOLDS` | overridden by the fold count in the file |
| `--trials` | `config.N_TRIALS` | Optuna trials |
| `--device` | auto | torch device; CUDA when available, else CPU |

### Device selection

With no `--device` the script picks CUDA when `torch.cuda.is_available()`, falls
back to MPS on Apple hardware, and otherwise uses the CPU. The chosen device is
printed at startup together with the GPU name and memory, so it is obvious from
the log which was used.

Passing `--device cuda` on a machine without a usable GPU **raises** rather than
silently running on the CPU, since the alternative is discovering after an hour
that the run was never accelerated. `--device cuda:1` selects a specific card.

For reproducibility, CUDA is seeded separately from the CPU (`manual_seed` does
not cover it) and cuDNN is put in deterministic mode with benchmarking disabled.

Note that the model is small — three layers of a few hundred units — so the
speedup from a GPU comes mainly from running many trials rather than from any
single trial being much faster. Per-trial cost at this size is dominated by
kernel-launch overhead rather than arithmetic.

### Search space

Declared in `config.SEARCH_SPACE` as plain data, so it can be edited without
touching the tuning logic:

| parameter | range |
|---|---|
| `n_layers` | 2–4 |
| `width` | 64–256 (log) |
| `activation` | SiLU, ELU, GELU |
| `lr` | 1e-4 – 1e-2 (log) |
| `weight_decay` | 1e-5 – 1e-2 (log) |

**Loss weights are fixed, not searched** (`W_SLOPE`, `LAMBDA_PASS`). They cannot
be in the search space, because Optuna minimises the total loss and the weights
are part of it: a trial that samples a small weight gets a small total regardless
of fit quality, so the search would drive both toward their lower bounds and the
"best" configuration would be the one that turned the penalties off. The weights
encode what matters, they are not a quantity to optimise.

**Batch size is fixed** at `config.BATCH_SIZE`, not searched. With a fixed epoch
budget, batch size and epoch count are two names for the same thing: the amount
of training is `steps_per_epoch × epochs`, and `steps_per_epoch` is set by the
batch size. Leaving both free lets the search trade one against the other, so
the resulting "optimum" reflects the epoch cap rather than the problem. Fixing
it also restores the epoch as a meaningful unit, which is what makes
`EARLY_STOP_PATIENCE` comparable across trials.

### The slope target is heavy-tailed

`dphi_df` is a derivative, so it blows up near resonance while the bulk of
configurations sit near zero. Squared error on that distribution is dominated by
a handful of points, which is why the slope MSE can sit at several units while
the complex targets fit well.

`SLOPE_CLIP_SIGMA` bounds the standardised slope error before squaring, which
limits the influence of the outliers without discarding them. Those are exactly
the configurations where the slope is physically interesting, so they should be
in the training set; they just should not determine the whole objective.

Set it to `None` to disable clipping.

Dropout is absent by design: the uncertainty comes from a deep ensemble, which
requires each member to be a well-trained deterministic function. Batch
normalisation is absent for the reason given in the model section.

### Objective and pruning

A trial's score is the **mean validation loss across folds**. The fold-to-fold
standard deviation is reported alongside, so a configuration whose mean looks
good but whose spread is large is visible rather than hidden.

`MedianPruner` abandons a trial whose fold is clearly worse than the median of
completed trials at the same epoch, which saves most of the cost of a bad
configuration. `PRUNER_STARTUP_TRIALS` and `PRUNER_WARMUP_EPOCHS` control how
long it waits before pruning.

### Outputs

Written next to the dataset.

| file | contents |
|---|---|
| `cv_best_params.npz` | best hyperparameters, plus the mean validation loss, trial count, fold count and the paths used |
| `cv_trials.csv` | one row per trial: hyperparameters, mean and std validation loss, mean and p95 physical phase error |
| `cv_tuning.log` | one line per trial, human-readable |

`cv_best_params.npz` is the file the deep-ensemble stage consumes, so tuning and
ensemble training use the same recipe by construction.

Note that the reported phase errors are in **physical degrees**: the predictions
are un-standardised before the phase is extracted, so a phase near the 0/360
boundary is not reported as a 360 degree error.

Z-scoring is a **fit**, so it is subject to the same leakage argument as any
other fitted step: fitting on rows that include the evaluation set encodes their
location and spread in the coefficients.

Two different rules are used, deliberately.

### Cross-validation folds — fitted on training rows only

Each fold gets **its own** scaler, fitted on that fold's training rows and
applied to its validation rows. The validation rows never contribute to the
coefficients, so the CV estimate carries no leakage.

Because each fold has a different training set, the coefficients are recomputed
per fold and are **not** shared between folds. The consequence is that a given
physical error maps to slightly different scaled errors in different folds, so
the mean CV loss is a mixture over slightly different scalings. That is accepted:
the alternative, one scaler fitted across folds, reintroduces the leakage.

### Deep ensemble — fitted on all rows

The ensemble is the final model, so a single scaling is computed from the **whole
dataset**, training and holdout together, and applied to both. This makes every
prediction and every reported error comparable on one scale.

This does mean the holdout inputs contribute their mean and spread to the
coefficients, so the final holdout score is **mildly optimistic**. The magnitude
is small: a scaler is a two-parameter fit over hundreds of points, so the
difference between the mean of the training portion and the mean of the whole set
is roughly `sigma/sqrt(n)`.

The trade is deliberate. For the ensemble a consistent scale is worth more than
the last fraction of a percent of leakage, because the ensemble is what gets
deployed and every downstream quantity — phase error in degrees, the axial-ratio
estimate, the ensemble spread — is expressed in that scale.

Every member of the ensemble uses the same coefficients, so the ensemble spread
reflects the networks rather than differences in preprocessing.

### Applying the coefficients

The scaling is not applied to the stored arrays — only the coefficients are
written. The training code applies them.

```python
d = np.load("split_holdout.npz")

# ensemble path: one scaler for everything
X_scaled = (dataset["X"] - d["ensemble_x_mean"]) / d["ensemble_x_std"]

# CV path: per-fold scaler
cv = np.load("split_cv_k5.npz")
tr, va = cv["fold0_train_idx"], cv["fold0_val_idx"]
X_tr = (dataset["X"][tr] - cv["fold0_x_mean"]) / cv["fold0_x_std"]
X_va = (dataset["X"][va] - cv["fold0_x_mean"]) / cv["fold0_x_std"]
```

A feature that is constant within a particular fit produces a zero column after
scaling, because the standard deviation is floored at `STD_FLOOR`. That is
harmless: the network's bias weight covers a constant. `h_base_mm` is the current
example, fixed at 0.6 mm throughout the dataset.

Set `NORMALISE_INPUTS` / `NORMALISE_TARGETS` in `config.py` to control which are
fitted.

## Output

Two files, written next to the dataset.

### `split_holdout.npz`

| key | meaning |
|---|---|
| `train_idx`, `test_idx` | index arrays into the (filtered) dataset rows |
| `group_ids` | geometry id per row, for auditing |
| `n_samples`, `n_groups` | sizes |
| `holdout_fraction`, `seed` | reproduction parameters |
| `filter_resonance` | whether the resonance filter was applied |
| `group_columns`, `strata_columns`, `strata_names` | keys used |
| `ensemble_x_mean`, `ensemble_x_std` | input scaling, fitted on **all** rows |
| `ensemble_y_mean`, `ensemble_y_std` | target scaling, fitted on **all** rows |
| `feature_names`, `target_names` | column labels for the coefficients |
| `dataset` | absolute path of the source dataset |

### `split_cv_k{folds}.npz`

Same metadata, plus `fold{i}_train_idx` and `fold{i}_val_idx` for each fold, and
one scaler per fold: `fold{i}_x_mean`, `fold{i}_x_std`, `fold{i}_y_mean`,
`fold{i}_y_std`.

Index arrays refer to the dataset **after** filtering. If `--filter-resonance`
was used, `X[keep][train_idx]` is the training set; without filtering the `keep`
step is the identity. The `filter_resonance` flag records which case applies, so
a consumer can reconstruct the mapping from the original dataset rows.

## Deep ensemble

`ensemble_train.py` trains `config.N_ENSEMBLE` members on the **full training
portion**, using the hyperparameters from `cv_tune.py`. The holdout is not
touched.

```bash
python ensemble_train.py --dataset ../data-processor/out/dataset.npz --dry-run
python ensemble_train.py --dataset ../data-processor/out/dataset.npz
```

| flag | default | meaning |
|---|---|---|
| `--dataset` | required | path to `dataset.npz` |
| `--params` | `cv_best_params.npz` beside the dataset | tuned hyperparameters |
| `--holdout` | `split_holdout.npz` beside the dataset | train/test indices |
| `--members` | `config.N_ENSEMBLE` | ensemble size |
| `--device` | auto | CUDA when available |

**Members differ only in random seed** — initialisation and data ordering. That
is what makes the ensemble spread interpretable as an epistemic uncertainty:
small where training data are dense, large where the model is extrapolating.
Varying the architecture between members would confound the spread with capacity
differences and the uncertainty would stop meaning anything.

`ENSEMBLE_SAMPLING` selects how each member's rows are chosen:

| mode | each member sees | spread captures |
|---|---|---|
| `fixed` | the whole training portion | optimisation randomness only |
| `bootstrap` | a bootstrap resample of **geometries** | optimisation + data variability |

**The resampling unit is the geometry, never the individual row.** Within one
geometry the values along `a` are samples of a single smooth curve, so they are
not independent observations. Resampling rows would leave every geometry
represented in every member, differing only in which points along each curve
were dropped, and the members would be nearly identical — the spread would then
measure optimisation noise only, and would be an underestimate that nothing
would reveal.

Bootstrap draws geometries with replacement, so roughly 37% are absent from each
member and some appear twice. With a few hundred geometries that is a real loss
per member, which the ensemble mean recovers; the benefit is that the spread
then reflects data variability as well.

**`N_ENSEMBLE` defaults to 10.** The sample standard deviation from `M` members
carries a relative error of about `1/sqrt(2(M-1))`: 35% at `M = 5`, 24% at
`M = 10`. If the spread is used to decide where to simulate next, that
difference matters, and members train in seconds.

**The scaler is fitted on the whole training portion**, since there are no folds
at this stage. It is applied unchanged to the holdout, which was not part of the
fit — the same rule as for a fold.

**No early stopping**, because there is no validation set. Members train for
`MAX_EPOCHS` and the training loss curve is logged, so a failure to converge is
visible after the fact.

### Outputs

| file | contents |
|---|---|
| `ensemble.pt` | hyperparameters and all member state dicts |
| `ensemble_meta.npz` | scalers, seeds, architecture, train/test indices, feature and target names |
| `ensemble_train.log` | per-member training loss curve |

`ensemble_meta.npz` carries the scalers, so evaluation does not refit anything.

## Evaluation

`evaluate.py` is the **only** consumer of the holdout. Run it once, at the end.

```bash
python evaluate.py --dataset ../data-processor/out/dataset.npz
python evaluate.py --dataset ../data-processor/out/dataset.npz \
                   --out ../data/statistics/model
```

| flag | default | meaning |
|---|---|---|
| `--dataset` | required | path to `dataset.npz` |
| `--out` | the dataset's directory | where the CSVs and report are written; created if absent |
| `--ensemble`, `--meta`, `--holdout` | beside the dataset | inputs |
| `--device` | `cpu` | torch device |
| `--tol` | `config.DPHI_TOL_DEG` | offset tolerance |

Note that `--out` affects only the **outputs**. The ensemble, metadata and holdout
split are always read from the dataset's directory unless overridden explicitly,
so redirecting the results does not break the inputs.

- **overall**
- **per stratum** — using the hand-written strata from `config.STRATA`
- **per ratio within stratum** — so a cell well fit at one elongation and badly
  fit at another is visible

### Metrics

| metric | meaning |
|---|---|
| `phase_err_mean` / `p95` | absolute error of `phi_major`, degrees |
| `dphi_err_mean` / `p95` | error of the inter-axis offset, degrees |
| `dphi_within` | fraction whose predicted offset is within `DPHI_TOL_DEG` |
| `mag_err_mean` | error of `|R|`, averaged over both axes |
| `ar_err_mean_db` | axial-ratio error implied by the predicted coefficients |
| `sigma_phi_mean` / `p95` | ensemble spread of `phi_major`, degrees |
| `sigma_dphi_mean` / `p95` | ensemble spread of the offset, degrees |
| `z_phi_mean` | calibration ratio, see below |
| `coverage_95` | fraction whose error falls inside the claimed `1.96 sigma` |

The offset is reported separately from the absolute phase because the two are
not implied by each other. The axes can be wrong in **compensating** ways, which
gives a good absolute error and a correct offset, or in **opposing** ways, which
gives the same absolute error and a drifted offset. Only the second is a problem,
and only the offset error reveals it.

### The calibration ratio

`z_phi_mean` is defined as

```
z = |error| / sigma
```

taken **per sample** and then averaged. This is the standard normalised
residual, so its scale is the familiar one: a well-calibrated Gaussian posterior
gives `E[z] ≈ 0.8`, not 1, because the mean of a half-normal is `sqrt(2/pi)`.

| `z` | reading |
|---|---|
| ≈ 0.8 | calibrated |
| > 0.8 | overconfident — sigma understates the real error, so it cannot be used as a filter |
| < 0.8 | conservative — sigma overstates it |

The per-sample ratio matters. Averaging the error and the sigma separately and
then dividing would let a few large-sigma samples cancel a few large errors,
which hides exactly the miscalibration you are looking for.

`coverage_95` is the complementary check: the fraction whose error actually falls
inside the claimed `1.96 sigma`. For a calibrated Gaussian that is 95%.

### Extrapolation diagnostic

Every grouping also reports the **mean ensemble spread** and the ratio of actual
error to claimed uncertainty (`err/sig`). Above one means the model is
overconfident in that group: the spread understates the real error, so the
uncertainty cannot be trusted as a filter.

The `eval_by_a.csv` table bins the test rows along the tuning variable. This is
where extrapolation would show: the phase slope steepens towards resonance, so
if the model knows it is less sure there, `sigma` should rise towards the ends of
the interval. A flat `sigma` alongside a rising error means it does not, and the
uncertainty is not a usable guide for where to simulate next.

### Calibration

The ensemble spread is validated by binning predictions by predicted `sigma` and
comparing the fraction falling within `±1.96 sigma` against the nominal 95%.
Systematic under-coverage means the intervals are overconfident and should be
recalibrated before the uncertainty is used to guide sampling.

The spread is taken over the derived nonlinear quantities per member — `dphi`
computed from each member separately — rather than by linear propagation of the
per-component variances.

### Outputs

| file | contents |
|---|---|
| `eval_summary.csv` | overall metrics plus mean ensemble spread |
| `eval_by_stratum.csv` | one row per stratum |
| `eval_by_ratio.csv` | one row per ratio within each stratum |
| `eval_by_a.csv` | one row per bin along the tuning variable |
| `eval_calibration.csv` | sigma bins versus observed coverage |
| `eval_report.txt` | human-readable summary |

## Plots

`plots.py` reads the evaluation CSVs and produces the figures.

```bash
python plots.py --dataset ../data-processor/out/dataset.npz
python plots.py --dataset ../data-processor/out/dataset.npz \
                --out ../data/statistics/model
```

| flag | default | meaning |
|---|---|---|
| `--dataset` | required | the evaluation CSVs are read from its directory |
| `--out` | the dataset's directory | where the PDFs are written; created if absent |
| `--tol` | `5.0` | offset tolerance for the reference line |

`--out` lets the figures go into the thesis's statistics directory while the CSVs
stay beside the dataset, or both can be redirected together by passing the same
path to `evaluate.py` and `plots.py`.

| figure | what it shows |
|---|---|
| `plot_error_vs_a.pdf` | error and sigma against the tuning variable, plus the within-tolerance fraction |
| `plot_error_vs_ratio.pdf` | error and sigma against elongation, one line per stratum |
| `plot_error_by_stratum.pdf` | grouped bars comparing error and sigma per stratum |
| `plot_calibration.pdf` | observed coverage against nominal, with the mean deviation annotated |

The `a` figure is the extrapolation diagnostic: if `sigma` rises towards the ends
of the interval alongside the error, the model knows where it is unsure. If the
error rises while `sigma` stays flat, it does not, and no uncertainty-based
filter will catch it.

## Intended workflow

The holdout is for final validation only. The full sequence is:

```bash
python ../data-processor/processor.py                        # dataset.npz
python splits.py     --dataset .../dataset.npz               # split_*.npz
python cv_tune.py    --dataset .../dataset.npz --trials 100  # cv_best_params.npz
python ensemble_train.py --dataset .../dataset.npz           # ensemble.pt
python evaluate.py       --dataset .../dataset.npz           # eval_*.csv
```

1. `splits.py` once, with the final `--filter-resonance` setting.
2. Cross-validation and hyperparameter tuning on `fold{i}_train_idx` only.
3. Deep-ensemble training on the full training portion.
4. Final evaluation on `test_idx`, **once**.

Looking at `test_idx` more than once turns it into a validation set and its score
stops being a generalisation estimate.

## The resonance filter

`resonance_in_band` is a 0/1 column in the dataset's `band` array, flagged when
the major-axis reflection dips below `RESONANCE_DIP_DB` inside the simulated
band. Such samples are valid design points but poor regression targets for the
phase slope, because the slope is not a smooth function of geometry near a
notch.

The filter lives here rather than in the data processor so the full dataset
stays intact on disk and the choice remains reversible. Activate it with
`FILTER_RESONANCE_IN_BAND = True` in `config.py` or `--filter-resonance`.
