"""
Train the deep ensemble on the full training portion.

Consumes the tuned hyperparameters from cv_tune.py and the holdout split from
splits.py, trains `config.N_ENSEMBLE` members on the training portion only, and
writes the ensemble plus its metadata.

The holdout is not touched here. It is reserved for evaluate.py, which is run
once at the end.

Usage
-----
    python ensemble_train.py --dataset path/to/dataset.npz
    python ensemble_train.py --dataset ... --members 5 --device cuda
    python ensemble_train.py --dataset ... --dry-run

Why members differ only in randomness
-------------------------------------
The ensemble spread is used as an epistemic uncertainty: small where the
training data are dense, large where the model is extrapolating. That
interpretation only holds if the members are samples from the same
distribution, differing in initialisation and data ordering. Varying the
architecture or hyperparameters between members would confound the spread with
capacity differences, and the resulting uncertainty would not mean anything.

Why the scaler is fitted on the whole training portion
------------------------------------------------------
There are no folds at this stage, so there is one training set. The coefficients
come from that set and are applied unchanged to the holdout, which was not part
of the fit. This is the same rule as for a fold: coefficients come from training
data, never from evaluation data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import model as net
import splits as split_lib


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_npz(path: Path, what: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def tuned_params(path: Path) -> dict:
    """Read the hyperparameters produced by cv_tune.py."""
    raw = load_npz(path, "tuned hyperparameters")
    out = {}
    for key in ("n_layers", "width", "activation", "lr", "weight_decay"):
        if key not in raw:
            raise KeyError(f"'{key}' missing from {path}")
        value = raw[key]
        out[key] = str(value) if key == "activation" else value.item()
    return out


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

def resample_geometries(
    train_idx: np.ndarray,
    groups: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
    coverage: float = config.ENSEMBLE_COVERAGE,
) -> np.ndarray:
    """
    Choose the rows one ensemble member trains on.

    The resampling unit is the geometry, never the individual row. Within one
    geometry the values along `a` are samples of a single smooth curve, so they
    are not independent observations. Resampling rows would leave every geometry
    represented in every member, differing only in which points along each curve
    were dropped, and the members would be nearly identical — the spread would
    then measure optimisation noise only.

    Modes
    -----
    "fixed"      the whole training portion, so members differ only in seed and
                 batch order. The spread captures optimisation randomness.
    "subsample"  a fraction `coverage` of the geometries, drawn WITHOUT
                 replacement.
    "bootstrap"  geometries drawn WITH replacement, so ~63% appear and some
                 appear twice. The most diversity, the most data loss.

    Stratification
    --------------
    When `strata` is given, the draw is made WITHIN each stratum rather than
    across the whole set. With only a few hundred geometries split unevenly
    between strata, an unstratified draw can give one member 8 geometries of a
    small stratum and another 18. The resulting spread would then reflect that
    composition noise rather than genuine uncertainty, and it would be inflated
    precisely in the small strata — which are the ones already hardest to fit.

    Parameters
    ----------
    train_idx : row indices of the training portion
    groups    : geometry id per row, aligned with the full dataset
    mode      : "fixed", "subsample" or "bootstrap"
    rng       : generator, so the choice is reproducible per member
    strata    : optional stratum id per row, aligned with the full dataset
    coverage  : fraction of geometries used by "subsample"

    Returns
    -------
    Row indices to train on. For "fixed" this is `train_idx` unchanged.
    """
    if mode == "fixed":
        return train_idx

    train_groups = groups[train_idx]

    # One (geometry, stratum) row per geometry. Every row of a geometry carries
    # the same stratum, since the stratum is derived from columns that are part
    # of the group key.
    uniq_groups, first = np.unique(train_groups, return_index=True)
    if strata is not None:
        group_strata = strata[train_idx][first]
    else:
        group_strata = np.zeros(uniq_groups.size, dtype=np.int64)

    if mode == "bootstrap":
        drawn = []
        for s in np.unique(group_strata):
            pool = uniq_groups[group_strata == s]
            drawn.append(rng.choice(pool, size=pool.size, replace=True))
        drawn = np.concatenate(drawn)
    elif mode == "subsample":
        drawn = []
        for s in np.unique(group_strata):
            pool = uniq_groups[group_strata == s]
            n_keep = max(1, int(round(coverage * pool.size)))
            drawn.append(rng.choice(pool, size=n_keep, replace=False))
        drawn = np.concatenate(drawn)
    else:
        raise ValueError(f"unknown ENSEMBLE_SAMPLING {mode!r}")

    # Expand each drawn geometry to its rows. Building a lookup once is cheaper
    # than searching per geometry.
    rows_by_group: dict[int, np.ndarray] = {}
    for g in uniq_groups:
        rows_by_group[int(g)] = train_idx[train_groups == g]

    pieces = [rows_by_group[int(g)] for g in drawn]
    return np.concatenate(pieces)


def member_seed(m: int) -> int:
    """Seed for member m, so the ensemble size can change without editing a list."""
    return config.ENSEMBLE_BASE_SEED + m * config.ENSEMBLE_SEED_STRIDE


# --------------------------------------------------------------------------- #
# Training one member
# --------------------------------------------------------------------------- #

def train_member(
    params: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    seed: int,
    device: str,
    log,
) -> tuple[dict, list[tuple[int, float]]]:
    """
    Train one ensemble member on the full training portion.

    There is no validation set at this stage, so early stopping cannot be used.
    Instead the member trains for a fixed number of epochs, and the training
    loss curve is logged so a failure to converge is visible after the fact.

    Returns the state dict and the loss history.
    """
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = net.build_mlp(
        n_inputs=X_train.shape[1],
        n_outputs=Y_train.shape[1],
        n_layers=params["n_layers"],
        width=params["width"],
        activation=params["activation"],
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    batch = config.BATCH_SIZE
    history: list[tuple[int, float]] = []

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0

        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            optimiser.zero_grad()

            pred = model(Xt[idx])
            loss, _ = net.mse_loss(
                pred, Yt[idx],
                w_slope=config.W_SLOPE,
                lambda_pass=config.LAMBDA_PASS,
                y_mean=y_mean_t,
                y_std=y_std_t,
                slope_clip=config.SLOPE_CLIP_SIGMA,
            )
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * idx.shape[0]

        epoch_loss = total / n
        history.append((epoch, epoch_loss))

        if epoch % config.LOG_EVERY_N_EPOCHS == 0 or epoch == config.MAX_EPOCHS - 1:
            log(f"    epoch {epoch:4d}  train_loss {epoch_loss:.5f}")

    return {k: v.detach().clone() for k, v in model.state_dict().items()}, history


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="directory holding dataset.npz, the holdout split "
                             "and cv_best_params.npz")
    parser.add_argument("--params", type=Path, default=None,
                        help="cv_best_params.npz; default in --dir")
    parser.add_argument("--holdout", type=Path, default=None,
                        help="split_holdout.npz; default in --dir")
    parser.add_argument("--members", type=int, default=config.N_ENSEMBLE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="load and check the inputs, then exit")
    args = parser.parse_args(argv)

    import cv_tune  # reuse the device resolution and benchmark helpers

    device = cv_tune.resolve_device(args.device)

    in_dir = args.dir.resolve()
    dataset_path = in_dir / "dataset.npz"
    dataset = load_dataset(dataset_path)
    out_dir = in_dir

    params_path = (args.params or out_dir / config.BEST_PARAMS_FILENAME).resolve()
    holdout_path = (args.holdout or out_dir / config.HOLDOUT_FILENAME).resolve()

    params = tuned_params(params_path)
    holdout = load_npz(holdout_path, "holdout split")

    # The split file stores indices into the FILTERED subset, so map them back
    # onto the dataset on disk. Without this the indices would address the wrong
    # rows whenever a filter was active during split generation.
    n_dataset = dataset["X"].shape[0]
    if "n_dataset" in holdout and int(holdout["n_dataset"]) != n_dataset:
        raise RuntimeError(
            f"split file was generated for a dataset of "
            f"{int(holdout['n_dataset'])} samples but this one has {n_dataset}; "
            f"regenerate the splits"
        )

    keep_idx = holdout["keep_idx"] if "keep_idx" in holdout else np.arange(n_dataset)
    train_idx = keep_idx[holdout["train_idx"]]
    test_idx = keep_idx[holdout["test_idx"]]

    print(f"dataset  : {dataset_path}")
    print(f"device   : {cv_tune.describe_device(device)}")
    print(f"params   : {params_path}")
    print(f"holdout  : {holdout_path}")
    print(f"X        : {dataset['X'].shape}")
    print(f"train    : {train_idx.size} samples")
    print(f"test     : {test_idx.size} samples (not used here)")
    print(f"members  : {args.members}")
    print(f"speed    : {cv_tune.benchmark_steps(device, dataset['X'].shape[1], dataset['Y'].shape[1])}")

    print("\nhyperparameters:")
    for k, v in params.items():
        print(f"  {k:<16} {v}")

    if args.dry_run:
        print("\ndry run: nothing trained")
        return 0

    # ---- scaling, fitted on the whole training portion ----
    X = dataset["X"]
    Y = dataset["Y"]

    x_mean, x_std = split_lib.fit_scaler(X, train_idx)
    y_mean, y_std = split_lib.fit_scaler(Y, train_idx)

    X_train = (X[train_idx] - x_mean) / x_std
    Y_train = (Y[train_idx] - y_mean) / y_std

    # ---- train the members ----
    log_path = out_dir / config.ENSEMBLE_LOG_FILENAME
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"ensemble training, {args.members} members")
    log(f"params: {params}")
    log(f"sampling: {config.ENSEMBLE_SAMPLING}")

    # Geometry ids, needed only when resampling. Computed once, over the full
    # dataset, so the ids are stable across members.
    groups = None
    strata = None
    if config.ENSEMBLE_SAMPLING != "fixed":
        groups = split_lib.geometry_group_ids(dataset)
        strata = split_lib.stratum_ids(dataset)
        log(f"geometries in training portion: "
            f"{np.unique(groups[train_idx]).size}")
        log(f"strata in training portion: "
            f"{np.unique(strata[train_idx]).size}")

    states: list = []
    histories: list[list[tuple[int, float]]] = []
    t_start = time.time()

    for m in range(args.members):
        seed = member_seed(m)
        rng = np.random.default_rng(seed)

        if groups is None:
            rows = np.arange(X_train.shape[0])
            n_geom = -1   # not meaningful for the fixed mode
        else:
            member_rows = resample_geometries(
                train_idx, groups, config.ENSEMBLE_SAMPLING, rng,
                strata=strata,
            )
            # Map full-dataset rows back to positions inside X_train.
            lookup = {int(r): i for i, r in enumerate(train_idx)}
            rows = np.array([lookup[int(r)] for r in member_rows])
            n_geom = np.unique(groups[member_rows]).size

        log(f"\n  member {m}  (seed {seed})  "
            f"rows {rows.size}  distinct geometries {n_geom}")

        state, history = train_member(
            params, X_train[rows], Y_train[rows], y_mean, y_std,
            seed=seed, device=device, log=log,
        )
        states.append(state)
        histories.append(history)

        final = history[-1][1]
        log(f"  member {m} done  final train_loss {final:.5f}  "
            f"[{time.time() - t_start:.0f} s]")

    # ---- write ----
    import torch

    model_path = out_dir / config.ENSEMBLE_MODEL_FILENAME
    torch.save(
        {
            "params": params,
            "states": states,
            "n_inputs": int(X.shape[1]),
            "n_outputs": int(Y.shape[1]),
        },
        model_path,
    )
    log(f"\nwrote {model_path}")

    meta_path = out_dir / config.ENSEMBLE_META_FILENAME
    np.savez_compressed(
        meta_path,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        members=np.int64(args.members),
        seeds=np.array([member_seed(m) for m in range(args.members)], dtype=np.int64),
        sampling=np.array(config.ENSEMBLE_SAMPLING),
        n_layers=np.int64(params["n_layers"]),
        width=np.int64(params["width"]),
        activation=np.array(params["activation"]),
        lr=np.float64(params["lr"]),
        weight_decay=np.float64(params["weight_decay"]),
        max_epochs=np.int64(config.MAX_EPOCHS),
        feature_names=dataset["feature_names"],
        target_names=dataset["target_names"],
        dataset=np.array(str(dataset_path)),
        train_idx=train_idx,
        test_idx=test_idx,
    )
    log(f"wrote {meta_path}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"wrote {log_path}")

    finals = [h[-1][1] for h in histories]
    print(f"\nfinal train losses: "
          f"min {min(finals):.5f}  max {max(finals):.5f}  "
          f"spread {max(finals) - min(finals):.5f}")
    print(f"total {time.time() - t_start:.0f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
