# Training

Split generation for the surrogate model. Consumes a `dataset.npz` produced by
`../data-processor/` and writes split files alongside it.

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
others and the fold-to-fold spread reflects the model rather than the split. It
is applied on the `(P_ratio, eps_r)` cells, which are the fabrication choices
with the least coverage by design.

Stratifying on height as well would be degenerate: 28 height values against a
few hundred geometries leaves about two geometries per stratum, too few to
populate every fold.

The consequence is worth stating plainly: folds always contain the heights the
model is tested on, so the CV estimate is a **best case with respect to height
extrapolation**. That is acceptable because the surrogate is queried at heights
inside the simulated range.

The holdout uses the same stratification, so every cell is represented on both
sides and per-cell error can be reported. Note that the `P_ratio = 0.5,
eps_r = 7.5` cell contains a single geometry; it can only appear on one side, so
no meaningful per-cell error is available for it. Treat it as the spot check the
thesis already describes it as.

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
| `group_columns`, `stratify_columns` | keys used |
| `dataset` | absolute path of the source dataset |

### `split_cv_k{folds}.npz`

Same metadata, plus `fold{i}_train_idx` and `fold{i}_val_idx` for each fold.

Index arrays refer to the dataset **after** filtering. If `--filter-resonance`
was used, `X[keep][train_idx]` is the training set; without filtering the `keep`
step is the identity. The `filter_resonance` flag records which case applies, so
a consumer can reconstruct the mapping from the original dataset rows.

## Intended workflow

The holdout is for final validation only. The recommended sequence is:

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
