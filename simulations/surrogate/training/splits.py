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


def apply_filters(
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


def stratum_ids(dataset: dict[str, np.ndarray], columns: list[str]) -> np.ndarray:
    """One integer stratum id per row, from the given X_raw columns."""
    keys = np.column_stack([raw_column(dataset, c) for c in columns])
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


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
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Print group and sample counts, plus the per-cell composition."""
    print(f"\n--- {name} ---")
    print(f"  samples : train {train_idx.size:6d}   val {val_idx.size:6d}")
    print(f"  groups  : train {np.unique(groups[train_idx]).size:6d}   "
          f"val {np.unique(groups[val_idx]).size:6d}")

    p_ratio = raw_column(dataset, "p_ratio")
    eps_r = raw_column(dataset, "eps_r")

    print(f"  {'P_ratio':>8} {'eps_r':>7} | {'train':>7} {'val':>7} {'groups val':>11}")
    cells = sorted({(float(a), float(b)) for a, b in zip(p_ratio, eps_r)})
    for p, e in cells:
        cell = (p_ratio == p) & (eps_r == e)
        tr = int(np.count_nonzero(cell[train_idx]))
        va = int(np.count_nonzero(cell[val_idx]))
        gv = int(np.unique(groups[val_idx][cell[val_idx]]).size)
        print(f"  {p:8.2f} {e:7.2f} | {tr:7d} {va:7d} {gv:11d}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="path to dataset.npz; split files are written to the same directory",
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

    dataset_path: Path = args.dataset or (
        Path(__file__).resolve().parents[3] / config.DEFAULT_DATASET
    )
    dataset_path = dataset_path.resolve()

    dataset = load_dataset(dataset_path)
    n_total = dataset["X"].shape[0]

    keep = apply_filters(dataset, args.filter_resonance)

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

    groups = geometry_group_ids(sub)
    holdout_strata = stratum_ids(sub, config.HOLDOUT_STRATIFY_COLUMNS)
    cv_strata = stratum_ids(sub, config.CV_STRATIFY_COLUMNS)

    print(f"geometries: {np.unique(groups).size}")
    print(f"holdout strata ({config.HOLDOUT_STRATIFY_COLUMNS}): "
          f"{np.unique(holdout_strata).size}")
    print(f"cv strata ({config.CV_STRATIFY_COLUMNS}): "
          f"{np.unique(cv_strata).size}")

    # ---- holdout ----
    train_idx, test_idx = make_holdout(
        groups, holdout_strata, args.holdout, config.HOLDOUT_SEED,
    )
    report_split("holdout", sub, groups, train_idx, test_idx)

    # ---- cross-validation ----
    folds = make_folds(groups, cv_strata, args.folds, config.CV_SEED)
    for f, (tr, va) in enumerate(folds):
        report_split(f"fold {f}", sub, groups, tr, va)

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
        n_groups=np.int64(np.unique(groups).size),
        holdout_fraction=np.float64(args.holdout),
        seed=np.int64(config.HOLDOUT_SEED),
        filter_resonance=np.array(args.filter_resonance),
        group_columns=np.array(config.GROUP_COLUMNS),
        stratify_columns=np.array(config.HOLDOUT_STRATIFY_COLUMNS),
        dataset=np.array(str(dataset_path)),
    )
    print(f"\nwrote {holdout_path}")

    cv_path = out_dir / config.CV_FILENAME_TEMPLATE.format(folds=args.folds)
    payload = {
        "group_ids": groups,
        "n_samples": np.int64(groups.size),
        "n_groups": np.int64(np.unique(groups).size),
        "folds": np.int64(args.folds),
        "seed": np.int64(config.CV_SEED),
        "filter_resonance": np.array(args.filter_resonance),
        "group_columns": np.array(config.GROUP_COLUMNS),
        "stratify_columns": np.array(config.CV_STRATIFY_COLUMNS),
        "dataset": np.array(str(dataset_path)),
    }
    for f, (tr, va) in enumerate(folds):
        payload[f"fold{f}_train_idx"] = tr
        payload[f"fold{f}_val_idx"] = va

    np.savez_compressed(cv_path, **payload)
    print(f"wrote {cv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
