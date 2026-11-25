"""
Split a processed dataset into holdout and cross-validation folds.

The dataset is produced by `data-processor/`; this module only reads it and
writes split files alongside it. It never modifies the dataset.

Why grouping
------------
Each row of the dataset is one `a` value belonging to one geometry, where a
geometry is a sweep along `a` at fixed (P_ratio, eps_r, H_pillar, ratio). There
are of order 10^4 rows but only a few hundred geometries.

A sample-level split would put different `a` values of the same geometry on both
sides. The model would then be tested on a curve it had already trained on,
differing only along one axis, and the score would measure interpolation inside
a known configuration rather than the ability to predict an unseen one. Since
the surrogate exists to propose *new* geometries, that would test the wrong
thing.

Splitting is therefore done at geometry level: all `a` of a geometry go to the
same side. `GroupKFold` and grouped shuffling implement this.

Why stratification
------------------
Grouping prevents leakage; stratification does not. Stratification only controls
how groups are distributed, so that each fold is compositionally similar to the
others and the fold-to-fold spread reflects the model rather than the split. It
is applied on (P_ratio, eps_r) cells, which are the fabrication choices with the
least coverage by design.

Stratifying on height as well would be degenerate: 28 height values against a
few hundred geometries leaves about two geometries per stratum, too few to
populate every fold. The consequence is that folds always contain the heights
the model is tested on, so the CV estimate is a best case with respect to height
extrapolation. That is acceptable here, because the surrogate is queried at
heights inside the simulated range.

Usage
-----
    python splits.py --dataset path/to/dataset.npz
    python splits.py --dataset ... --holdout 0.25 --folds 10
    python splits.py --dataset ... --filter-resonance
    python splits.py --dataset ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load a dataset produced by the data processor."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def band_column(dataset: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Extract one named column from the dataset's `band` array."""
    names = [str(n) for n in dataset["band_names"]]
    if name not in names:
        raise KeyError(
            f"'{name}' not in band_names {names}; "
            f"activate it in data-processor/config.py first"
        )
    return dataset["band"][:, names.index(name)]


def apply_resonance_filter(
    dataset: dict[str, np.ndarray],
    filter_resonance: bool,
) -> np.ndarray:
    """
    Boolean mask of rows to keep.

    Only the optional resonance filter is applied here. Geometry validity,
    eigenbasis validity and target finiteness were already enforced by the data
    processor, so every row in the dataset is a valid design point; this filter
    is a modelling choice about which rows are good regression targets.
    """
    keep = np.ones(dataset["X"].shape[0], dtype=bool)

    if filter_resonance:
        flag = band_column(dataset, config.RESONANCE_FLAG_COLUMN)
        keep &= (flag == 0)

    return keep


# --------------------------------------------------------------------------- #
# Grouping and stratification
# --------------------------------------------------------------------------- #

def raw_column(dataset: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Extract one named column from the dataset's `X_raw` array."""
    names = [str(n) for n in dataset["raw_names"]]
    if name not in names:
        raise KeyError(f"'{name}' not in raw_names {names}")
    return dataset["X_raw"][:, names.index(name)]


def geometry_group_ids(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """
    One integer id per distinct geometry.

    The geometry is the full tuple of fixed parameters. Samples sharing a tuple
    belong to the same `a` sweep and must not be split across sides.
    """
    columns = [raw_column(dataset, c) for c in config.GROUP_COLUMNS]

    # Build a comparable key per row. np.unique with axis=0 handles the
    # lexicographic ordering; float equality is safe because the values come
    # from the same CSV strings, so identical configurations have identical
    # bit patterns.
    keys = np.column_stack(columns)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def stratum_ids(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """
    One integer stratum id per row, from the hand-written strata in config.

    A sample belongs to the first stratum whose every condition matches. Values
    may be a scalar (exact match) or a tuple (membership). Samples matching no
    stratum are reported and given the fallback id, so nothing is silently
    dropped.

    Strata are defined by hand rather than derived from the columns so the
    granularity can be tuned to the data: the small permittivity cells carry
    only a dozen geometries, and pooling adjacent ones restores a workable
    number of groups per fold.
    """
    n = dataset["X"].shape[0]
    ids = np.full(n, -1, dtype=np.int64)

    for sid, spec in enumerate(config.STRATA):
        mask = np.ones(n, dtype=bool)
        for column in config.STRATA_COLUMNS:
            if column not in spec:
                continue
            values = raw_column(dataset, column)
            wanted = spec[column]
            if isinstance(wanted, tuple):
                mask &= np.isin(values, wanted)
            else:
                mask &= (values == wanted)

        # First match wins; a sample already assigned is not reassigned.
        mask &= (ids < 0)
        ids[mask] = sid

    unmatched = int(np.count_nonzero(ids < 0))
    if unmatched:
        print(f"  warning: {unmatched} samples match no stratum in config.STRATA; "
              f"assigned to fallback stratum {len(config.STRATA)}")
        ids[ids < 0] = len(config.STRATA)

    return ids


def stratum_names() -> list[str]:
    """Human-readable stratum names, in id order, including the fallback."""
    return [spec["name"] for spec in config.STRATA] + ["unmatched"]


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #

def _group_stratum_table(
    groups: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce to one (group, stratum) row per group.

    Every sample of a group must carry the same stratum, which holds as long as
    the group key contains the stratify columns. This asserts that and returns
    the per-group stratum assignment, ordered by ascending group id.
    """
    uniq_groups, first_idx = np.unique(groups, return_index=True)
    group_strata = strata[first_idx]

    # Vectorised consistency check: each group must map to exactly one stratum.
    # Sorting by group then counting distinct strata per group is O(n log n)
    # and avoids a Python loop over groups.
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    s_sorted = strata[order]

    boundaries = np.flatnonzero(np.diff(g_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [g_sorted.size]))

    for a, b in zip(starts, ends):
        if np.unique(s_sorted[a:b]).size > 1:
            raise RuntimeError(
                f"group {g_sorted[a]} spans multiple strata; "
                f"GROUP_COLUMNS must contain the stratify columns"
            )

    return uniq_groups, group_strata


def make_holdout(
    groups: np.ndarray,
    strata: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grouped, stratified holdout split.

    Groups (not samples) are divided, within each stratum, so that the holdout
    keeps the cell composition of the full dataset. Returns index arrays into
    the sample axis.
    """
    rng = np.random.default_rng(seed)
    uniq_groups, group_strata = _group_stratum_table(groups, strata)

    test_groups: list[int] = []
    for stratum in np.unique(group_strata):
        pool = uniq_groups[group_strata == stratum]
        rng.shuffle(pool)

        n_test = int(round(fraction * pool.size))
        # Guarantee at least one group in the holdout for every stratum that
        # has more than one group, so per-cell error is reportable.
        if pool.size > 1:
            n_test = max(1, min(n_test, pool.size - 1))

        test_groups.extend(pool[:n_test].tolist())

    test_mask = np.isin(groups, test_groups)
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    return train_idx, test_idx


def make_folds(
    groups: np.ndarray,
    strata: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Grouped, stratified k-fold split.

    Groups are distributed across folds within each stratum, keeping both the
    fold sizes and the cell composition as similar as possible. Returns a list
    of (train_idx, val_idx) index arrays.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")

    rng = np.random.default_rng(seed)
    uniq_groups, group_strata = _group_stratum_table(groups, strata)
    n_groups = uniq_groups.size

    # Group index per sample. uniq_groups is sorted, so searchsorted is exact.
    sample_group = np.searchsorted(uniq_groups, groups)

    group_fold = np.full(n_groups, -1, dtype=np.int64)
    fold_load = np.zeros(folds, dtype=np.int64)

    # Deal groups into folds within each stratum, always picking the currently
    # least-loaded fold. This keeps fold sizes balanced even when strata have
    # very different numbers of groups, which round-robin from a fixed offset
    # does not.
    for stratum in np.unique(group_strata):
        pool = np.flatnonzero(group_strata == stratum)
        rng.shuffle(pool)

        for gi in pool:
            target = int(np.argmin(fold_load))
            group_fold[gi] = target
            fold_load[target] += 1

    if np.any(group_fold < 0):
        raise RuntimeError("some groups were never assigned to a fold")

    sample_fold = group_fold[sample_group]

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(folds):
        val_idx = np.flatnonzero(sample_fold == f)
        train_idx = np.flatnonzero(sample_fold != f)
        out.append((train_idx, val_idx))

    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def report_split(
    name: str,
    dataset: dict[str, np.ndarray],
    groups: np.ndarray,
    strata: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Print group and sample counts, plus the per-stratum composition."""
    print(f"\n--- {name} ---")
    print(f"  samples : train {train_idx.size:6d}   val {val_idx.size:6d}")
    print(f"  groups  : train {np.unique(groups[train_idx]).size:6d}   "
          f"val {np.unique(groups[val_idx]).size:6d}")

    names = stratum_names()
    print(f"  {'stratum':<16} {'train':>8} {'val':>8} {'groups val':>11}")
    for sid in range(len(names)):
        cell = (strata == sid)
        tr = int(np.count_nonzero(cell[train_idx]))
        va = int(np.count_nonzero(cell[val_idx]))
        gv = int(np.unique(groups[val_idx][cell[val_idx]]).size)
        print(f"  {names[sid]:<16} {tr:8d} {va:8d} {gv:11d}")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def fit_scaler(
    X: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a z-score on the training rows only, returning (mean, std).

    Fitting on the whole dataset would leak the validation set's location and
    spread into the scaling coefficients. The model never sees the validation
    labels, but every prediction it makes would be shifted by knowledge of
    where those inputs lie.

    The standard deviation is floored, so a feature that is constant within this
    particular training split produces a zero column rather than a division by
    zero. That is harmless: the network's bias weight covers a constant.
    """
    train = X[train_idx]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < config.STD_FLOOR, 1.0, std)
    return mean, std


def apply_scaler(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply a fitted z-score. Used by the training code, not here."""
    return (X - mean) / std


def scaler_payload(
    name: str,
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Fitted scaling coefficients for one split, ready to be stored.

    Keys are prefixed with the split name so a single file can hold several
    scalers, e.g. `holdout_x_mean`, `fold0_y_std`.
    """
    payload: dict[str, np.ndarray] = {}

    if config.NORMALISE_INPUTS:
        x_mean, x_std = fit_scaler(X, train_idx)
        payload[f"{name}_x_mean"] = x_mean
        payload[f"{name}_x_std"] = x_std

    if config.NORMALISE_TARGETS:
        y_mean, y_std = fit_scaler(Y, train_idx)
        payload[f"{name}_y_mean"] = y_mean
        payload[f"{name}_y_std"] = y_std

    return payload


def report_scaler(
    name: str,
    feature_names: list[str],
    target_names: list[str],
    payload: dict[str, np.ndarray],
) -> None:
    """Print the fitted means and standard deviations for inspection."""
    for kind, names in (("x", feature_names), ("y", target_names)):
        mean = payload.get(f"{name}_{kind}_mean")
        std = payload.get(f"{name}_{kind}_std")
        if mean is None:
            continue
        print(f"  {kind} scaling ({name}):")
        for n, m, s in zip(names, mean, std):
            print(f"    {n:<22} mean {m:12.5g}   std {s:12.5g}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="directory holding dataset.npz; split files are written to the "
             "same directory. Default: config.DEFAULT_DATASET's directory",
    )
    parser.add_argument(
        "--holdout", type=float, default=config.HOLDOUT_FRACTION,
        help="fraction of geometries held out for final evaluation",
    )
    parser.add_argument(
        "--folds", type=int, default=config.CV_FOLDS,
        help="number of cross-validation folds",
    )
    parser.add_argument(
        "--filter-resonance", action="store_true",
        default=config.FILTER_RESONANCE_IN_BAND,
        help="exclude samples whose band contains a resonance",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the splits without writing them",
    )
    args = parser.parse_args(argv)

    in_dir: Path = (args.dir.resolve() if args.dir
                    else (Path(__file__).resolve().parents[3]
                          / config.DEFAULT_DATASET).parent)
    dataset_path = in_dir / "dataset.npz"

    dataset = load_dataset(dataset_path)
    n_total = dataset["X"].shape[0]

    keep = apply_resonance_filter(dataset, args.filter_resonance)

    if args.filter_resonance:
        print(f"resonance filter: keeping {np.count_nonzero(keep)} "
              f"of {n_total} samples")
        if not np.any(keep):
            print("error: the filter removed every sample")
            return 1
    else:
        print(f"no filtering: using all {n_total} samples")

    # Restrict every array to the kept rows before splitting, so indices in the
    # saved files refer to the filtered dataset consistently.
    sub = {k: (v[keep] if v.ndim >= 1 and v.shape[0] == n_total else v)
           for k, v in dataset.items()}

    # The kept row indices are written to the split files. Without them a
    # consumer cannot map the stored indices back onto the dataset, because the
    # indices refer to the filtered subset while the dataset on disk is
    # unfiltered. Storing the mask keeps the dataset intact and the mapping
    # explicit.
    keep_idx = np.flatnonzero(keep).astype(np.int64)

    groups = geometry_group_ids(sub)
    strata = stratum_ids(sub)

    print(f"geometries: {np.unique(groups).size}")
    print(f"strata ({config.STRATA_COLUMNS}): {np.unique(strata).size}")

    # ---- holdout ----
    train_idx, test_idx = make_holdout(
        groups, strata, args.holdout, config.HOLDOUT_SEED,
    )
    report_split("holdout", sub, groups, strata, train_idx, test_idx)

    # ---- cross-validation ----
    # Folds are computed over the TRAINING portion only. Computing them over the
    # full dataset would put holdout samples into some folds' training sets, so
    # hyperparameters would be chosen with knowledge of the holdout and its
    # score would stop being a generalisation estimate.
    train_folds = make_folds(
        groups[train_idx], strata[train_idx], args.folds, config.CV_SEED,
    )
    folds = [
        (train_idx[tr], train_idx[va])
        for tr, va in train_folds
    ]
    for f, (tr, va) in enumerate(folds):
        report_split(f"fold {f}", sub, groups, strata, tr, va)

    # Guard: no fold may touch a holdout sample, or the holdout stops being an
    # independent estimate and hyperparameters would be chosen against it.
    test_set = set(test_idx.tolist())
    for f, (tr, va) in enumerate(folds):
        overlap = test_set.intersection(tr.tolist()) | test_set.intersection(va.tolist())
        if overlap:
            raise RuntimeError(
                f"fold {f} overlaps the holdout by {len(overlap)} samples; "
                f"folds must be built from the training portion only"
            )

    # ---- normalisation ----
    # Two different rules, deliberately:
    #
    #   fold scalers     fitted on that fold's training rows only, so the
    #                    validation rows never contribute to the coefficients.
    #                    This is what keeps the CV estimate free of leakage.
    #
    #   ensemble scaler  fitted on ALL rows, training and holdout together, and
    #                    then applied to both. The ensemble is the final model,
    #                    and a single scaling computed over the whole dataset
    #                    makes every prediction and every reported error
    #                    comparable on one scale.
    #
    # NOTE: fitting the ensemble scaler on all rows means the holdout inputs
    # contribute their mean and spread to the coefficients, so the final
    # holdout score is mildly optimistic. See the README and the thesis for the
    # reasoning and the magnitude (a two-parameter fit over hundreds of points).
    X = sub["X"]
    Y = sub["Y"]
    feature_names = [str(n) for n in sub["feature_names"]]
    target_names = [str(n) for n in sub["target_names"]]

    all_idx = np.arange(X.shape[0], dtype=np.int64)
    ensemble_scaler = scaler_payload("ensemble", X, Y, all_idx)
    if ensemble_scaler:
        print("\n--- normalisation ---")
        report_scaler("ensemble", feature_names, target_names, ensemble_scaler)

    fold_scalers: dict[str, np.ndarray] = {}
    for f, (tr, _) in enumerate(folds):
        fold_scalers.update(scaler_payload(f"fold{f}", X, Y, tr))

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    # ---- write ----
    out_dir = dataset_path.parent

    holdout_path = out_dir / config.HOLDOUT_FILENAME
    np.savez_compressed(
        holdout_path,
        train_idx=train_idx,
        test_idx=test_idx,
        group_ids=groups,
        n_samples=np.int64(groups.size),
        n_dataset=np.int64(n_total),
        keep_idx=keep_idx,
        n_groups=np.int64(np.unique(groups).size),
        holdout_fraction=np.float64(args.holdout),
        seed=np.int64(config.HOLDOUT_SEED),
        filter_resonance=np.array(args.filter_resonance),
        group_columns=np.array(config.GROUP_COLUMNS),
        strata_columns=np.array(config.STRATA_COLUMNS),
        strata_names=np.array(stratum_names()),
        dataset=np.array(str(dataset_path)),
        feature_names=np.array(feature_names),
        target_names=np.array(target_names),
        **ensemble_scaler,
    )
    print(f"\nwrote {holdout_path}")

    cv_path = out_dir / config.CV_FILENAME_TEMPLATE.format(folds=args.folds)
    payload = {
        "group_ids": groups,
        "n_samples": np.int64(groups.size),
        "n_dataset": np.int64(n_total),
        "keep_idx": keep_idx,
        "n_groups": np.int64(np.unique(groups).size),
        "folds": np.int64(args.folds),
        "seed": np.int64(config.CV_SEED),
        "filter_resonance": np.array(args.filter_resonance),
        "group_columns": np.array(config.GROUP_COLUMNS),
        "strata_columns": np.array(config.STRATA_COLUMNS),
        "strata_names": np.array(stratum_names()),
        "dataset": np.array(str(dataset_path)),
        "feature_names": np.array(feature_names),
        "target_names": np.array(target_names),
    }
    for f, (tr, va) in enumerate(folds):
        payload[f"fold{f}_train_idx"] = tr
        payload[f"fold{f}_val_idx"] = va
    payload.update(fold_scalers)

    np.savez_compressed(cv_path, **payload)
    print(f"wrote {cv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
