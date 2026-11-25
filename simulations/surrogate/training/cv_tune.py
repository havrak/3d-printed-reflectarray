"""
Cross-validated hyperparameter tuning with Optuna.

Consumes a dataset and the split file produced by splits.py, tunes the network
over the folds, and writes the best hyperparameters plus a trial log next to the
dataset.

The holdout is never touched here. Every fold's training and validation rows come
from the training portion only, and splits.py already guarantees that no fold
overlaps the holdout.

Usage
-----
    python cv_tune.py --dataset path/to/dataset.npz
    python cv_tune.py --dataset ... --trials 20 --folds-file split_cv_k5.npz
    python cv_tune.py --dataset ... --dry-run

Requires numpy and torch, and optuna for the search itself.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import model as net


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_splits(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"split file not found: {path}\n"
            f"run splits.py first to generate it"
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fold_indices(splits: dict[str, np.ndarray], folds: int):
    """
    Yield (fold, train_idx, val_idx) from a split file.

    Indices in the split file address the FILTERED subset used when the splits
    were generated, so they are mapped back onto the dataset on disk via
    `keep_idx`. Without this they would address the wrong rows whenever a
    resonance filter was active.
    """
    keep_idx = splits["keep_idx"] if "keep_idx" in splits else None

    for f in range(folds):
        train_idx = splits[f"fold{f}_train_idx"]
        val_idx = splits[f"fold{f}_val_idx"]

        if keep_idx is not None:
            train_idx = keep_idx[train_idx]
            val_idx = keep_idx[val_idx]

        yield f, train_idx, val_idx


def scaled_fold(
    dataset: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
):
    """
    Return standardised (X_train, Y_train, X_val, Y_val) for one fold.

    The scaling coefficients stored by splits.py were fitted on that fold's
    training rows only, so applying them to the validation rows introduces no
    leakage. They are read rather than recomputed so tuning and the later
    ensemble share exactly the same preprocessing.
    """
    X = dataset["X"]
    Y = dataset["Y"]

    x_mean = splits[f"fold{fold}_x_mean"]
    x_std = splits[f"fold{fold}_x_std"]
    y_mean = splits[f"fold{fold}_y_mean"]
    y_std = splits[f"fold{fold}_y_std"]

    X_train = (X[train_idx] - x_mean) / x_std
    Y_train = (Y[train_idx] - y_mean) / y_std
    X_val = (X[val_idx] - x_mean) / x_std
    Y_val = (Y[val_idx] - y_mean) / y_std

    return X_train, Y_train, X_val, Y_val, y_mean, y_std


# --------------------------------------------------------------------------- #
# Training one fold
# --------------------------------------------------------------------------- #

def resolve_device(requested: str | None) -> str:
    """
    Pick a torch device.

    With no explicit request, use CUDA when a GPU is available and fall back to
    CPU otherwise. An explicit request is honoured but still checked, so asking
    for `cuda` on a machine without one fails loudly instead of silently running
    on the CPU for hours.
    """
    import torch

    if requested:
        req = requested.lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {requested} requested but torch.cuda.is_available() "
                f"is False (torch {torch.__version__}, "
                f"built with CUDA {torch.version.cuda})"
            )
        if req.startswith("cuda"):
            index = 0
            if ":" in req:
                index = int(req.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"--device {requested} requested but only "
                    f"{torch.cuda.device_count()} device(s) present"
                )
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def describe_device(device: str) -> str:
    """One-line description for the log."""
    import torch

    if device.startswith("cuda"):
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"{device}  ({name}, {total:.1f} GiB)"
    return device


def benchmark_steps(device: str, n_inputs: int, n_outputs: int) -> str:
    """
    Time a few training steps so the expected speed is visible up front.

    A small model on a large batch is dominated by kernel-launch and Python
    overhead rather than arithmetic, so `nvidia-smi` shows almost no utilisation
    even when the GPU is being used. This measures the actual step rate, which
    is the number that predicts how long the study will take.
    """
    import time as _time

    import torch

    model = net.build_mlp(
        n_inputs=n_inputs, n_outputs=n_outputs,
        n_layers=3, width=128, activation="SiLU",
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(config.BATCH_SIZE, n_inputs, device=device)
    y = torch.randn(config.BATCH_SIZE, n_outputs, device=device)
    y_mean = torch.zeros(n_outputs, device=device)
    y_std = torch.ones(n_outputs, device=device)

    # Warm up, so the first CUDA kernel compilation is not counted.
    for _ in range(5):
        optimiser.zero_grad()
        loss, _ = net.mse_loss(model(x), y, 0.1, 1.0, y_mean, y_std)
        loss.backward()
        optimiser.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    n = 20
    t0 = _time.perf_counter()
    for _ in range(n):
        optimiser.zero_grad()
        loss, _ = net.mse_loss(model(x), y, 0.1, 1.0, y_mean, y_std)
        loss.backward()
        optimiser.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (_time.perf_counter() - t0) / n

    steps_per_epoch = max(1, 15000 // config.BATCH_SIZE)
    return (
        f"{dt * 1e3:.2f} ms/step at batch {config.BATCH_SIZE} "
        f"(~{1 / dt:.0f} steps/s, ~{steps_per_epoch * dt:.2f} s/epoch "
        f"at {steps_per_epoch} steps/epoch)"
    )


def train_fold(
    params: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: str,
    report_epoch=None,
) -> dict[str, float]:
    """
    Train one fold and return its final validation metrics.

    Early stopping monitors the validation loss. Each fold is independent, so
    the epoch at which it stops varies; that is expected and adds diversity when
    the same recipe is later used for ensemble members.

    `report_epoch(epoch, val_loss)` is called periodically so an Optuna pruner
    can abandon a clearly bad trial early.
    """
    import torch

    torch.manual_seed(config.TRAIN_SEED)
    if device.startswith("cuda"):
        # Seeding CUDA separately is required: torch.manual_seed only covers the
        # CPU generator, and cuDNN kernel selection is a further source of
        # run-to-run variation.
        torch.cuda.manual_seed_all(config.TRAIN_SEED)
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

    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    Yv = torch.as_tensor(Y_val, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    batch = config.BATCH_SIZE

    best_loss = float("inf")
    best_state = None
    since_improved = 0

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)

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

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss, val_parts = net.mse_loss(
                val_pred, Yv,
                w_slope=config.W_SLOPE,
                lambda_pass=config.LAMBDA_PASS,
                y_mean=y_mean_t,
                y_std=y_std_t,
                slope_clip=config.SLOPE_CLIP_SIGMA,
            )
        val_loss = float(val_loss)

        if report_epoch is not None:
            report_epoch(epoch, val_loss)

        if val_loss < best_loss - config.EARLY_STOP_MIN_DELTA:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= config.EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv)
        _, parts = net.mse_loss(
            val_pred, Yv,
            w_slope=config.W_SLOPE,
            lambda_pass=config.LAMBDA_PASS,
            y_mean=y_mean_t,
            y_std=y_std_t,
            slope_clip=config.SLOPE_CLIP_SIGMA,
        )
        physical = val_pred[:, :4] * y_std_t[:4] + y_mean_t[:4]
        true_physical = Yv[:, :4] * y_std_t[:4] + y_mean_t[:4]

    # Physical phase errors in degrees. The difference is wrapped to (-180, 180]
    # so a phase near the 0/360 boundary is not reported as a 360 degree error.
    #
    # Columns of Y: 0 = Re(major), 1 = Im(major), 2 = Re(minor), 3 = Im(minor).
    def _phase(block):
        return torch.rad2deg(torch.atan2(block[:, 1], block[:, 0]))

    def _wrapped_err(pred_deg, true_deg):
        return torch.abs(torch.remainder(pred_deg - true_deg + 180.0, 360.0) - 180.0)

    true_phi_major = _phase(true_physical)
    pred_phi_major = _phase(physical)
    phase_err = _wrapped_err(pred_phi_major, true_phi_major)

    # The inter-axis offset is the quantity the whole device turns on: it is what
    # has to hold near 90 degrees for circular polarisation. Its error is
    # therefore more informative than the absolute phase error, and it is not
    # implied by it, because the two axes can be wrong in compensating ways.
    true_phi_minor = torch.rad2deg(
        torch.atan2(true_physical[:, 3], true_physical[:, 2])
    )
    pred_phi_minor = torch.rad2deg(
        torch.atan2(physical[:, 3], physical[:, 2])
    )
    true_dphi = torch.remainder(true_phi_minor - true_phi_major, 360.0)
    pred_dphi = torch.remainder(pred_phi_minor - pred_phi_major, 360.0)
    dphi_err = _wrapped_err(pred_dphi, true_dphi)

    # Fraction of predictions whose predicted offset lands within tolerance of
    # the true offset. Reported alongside the mean so a configuration that is
    # usually right but occasionally badly wrong is distinguishable from one
    # that is uniformly mediocre.
    dphi_ok = (dphi_err <= config.DPHI_TOL_DEG).float().mean()

    return {
        "val_loss": best_loss,
        "complex_mse": parts["complex_mse"],
        "slope_mse": parts["slope_mse"],
        "passivity": parts["passivity"],
        "phase_err_mean_deg": float(phase_err.mean()),
        "phase_err_p95_deg": float(torch.quantile(phase_err, 0.95)),
        "dphi_err_mean_deg": float(dphi_err.mean()),
        "dphi_err_p95_deg": float(torch.quantile(dphi_err, 0.95)),
        "dphi_within_tol": float(dphi_ok),
    }


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #

def suggest(trial, name: str, spec: dict):
    """Translate one SEARCH_SPACE entry into an Optuna suggestion."""
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    raise ValueError(f"unknown search-space type {kind!r} for {name!r}")


def run_tuning(
    dataset: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    folds: int,
    n_trials: int,
    device: str,
    log_path: Path,
) -> tuple[dict, list[dict]]:
    """Run the Optuna study and return (best_params, trial records)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.PRUNER_STARTUP_TRIALS,
        n_warmup_steps=config.PRUNER_WARMUP_EPOCHS,
    )

    records: list[dict] = []
    t_start = time.time()

    def objective(trial):
        params = {name: suggest(trial, name, spec)
                  for name, spec in config.SEARCH_SPACE.items()}

        fold_metrics: list[dict] = []

        for f, train_idx, val_idx in fold_indices(splits, folds):
            (X_train, Y_train, X_val, Y_val,
             y_mean, y_std) = scaled_fold(dataset, splits, f, train_idx, val_idx)

            def report_epoch(epoch, loss, _f=f):
                trial.report(loss, step=_f * config.MAX_EPOCHS + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            metrics = train_fold(
                params, X_train, Y_train, X_val, Y_val,
                y_mean, y_std, device, report_epoch=report_epoch,
            )
            fold_metrics.append(metrics)

            # Per-fold line so progress is visible while a trial is running.
            # Without this the script is silent for the whole study.
            # The loss components are printed separately because the total is
            # dominated by whichever term is largest, and a total alone cannot
            # distinguish a bad fit from a passivity penalty that is firing.
            # The offset error is reported next to the absolute phase error
            # because it is the quantity the CP condition depends on.
            print(
                f"  trial {trial.number:4d}  fold {f}  "
                f"loss {metrics['val_loss']:.4f} "
                f"[cx {metrics['complex_mse']:.4f} "
                f"sl {metrics['slope_mse']:.4f} "
                f"ps {metrics['passivity']:.4f}]  "
                f"phi_err {metrics['phase_err_mean_deg']:.2f} deg  "
                f"dphi_err {metrics['dphi_err_mean_deg']:.2f} deg "
                f"({100 * metrics['dphi_within_tol']:.0f}% within "
                f"{config.DPHI_TOL_DEG:.0f})  "
                f"[{time.time() - t_start:6.1f} s]",
                flush=True,
            )

        # The trial score is the mean validation loss across folds; the spread
        # is reported alongside so a high-variance configuration is visible even
        # when its mean looks good.
        losses = np.array([m["val_loss"] for m in fold_metrics])
        record = {
            "trial": trial.number,
            **params,
            "val_loss_mean": float(losses.mean()),
            "val_loss_std": float(losses.std()),
            "phase_err_mean_deg": float(np.mean([m["phase_err_mean_deg"] for m in fold_metrics])),
            "phase_err_p95_deg": float(np.mean([m["phase_err_p95_deg"] for m in fold_metrics])),
            "dphi_err_mean_deg": float(np.mean([m["dphi_err_mean_deg"] for m in fold_metrics])),
            "dphi_err_p95_deg": float(np.mean([m["dphi_err_p95_deg"] for m in fold_metrics])),
            "dphi_within_tol": float(np.mean([m["dphi_within_tol"] for m in fold_metrics])),
        }
        records.append(record)

        print(
            f"trial {trial.number:4d} DONE  "
            f"loss {record['val_loss_mean']:.5f} +/- {record['val_loss_std']:.5f}  "
            f"phi_err {record['phase_err_mean_deg']:.2f} deg  "
            f"dphi_err {record['dphi_err_mean_deg']:.2f} deg  "
            f"({100 * record['dphi_within_tol']:.0f}% within {config.DPHI_TOL_DEG:.0f})  "
            f"layers {params['n_layers']} width {params['width']} "
            f"act {params['activation']} lr {params['lr']:.2e}",
            flush=True,
        )

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"trial {trial.number:4d}  "
                f"loss {record['val_loss_mean']:.5f} +/- {record['val_loss_std']:.5f}  "
                f"phi_err {record['phase_err_mean_deg']:.2f} deg  "
                f"dphi_err {record['dphi_err_mean_deg']:.2f} deg  "
                f"dphi_ok {100 * record['dphi_within_tol']:.0f}%  "
                f"layers {params['n_layers']} width {params['width']} "
                f"act {params['activation']} lr {params['lr']:.2e}\n"
            )

        return record["val_loss_mean"]

    def on_trial_complete(study, trial):
        """Report the running best after each trial."""
        done = len([t for t in study.trials if t.value is not None])
        print(
            f"  --- {done} trials complete, "
            f"best so far {study.best_value:.5f} ---",
            flush=True,
        )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        objective, n_trials=n_trials, callbacks=[on_trial_complete],
    )

    return study.best_params, records


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two inputs are required and they are different things:\n"
            "  --dataset     holds the actual arrays (X, Y, feature names)\n"
            "  --folds-file  holds only row indices and scaling coefficients\n"
            "The split file refers to dataset rows by index, so it cannot be\n"
            "used without the dataset it was generated from."
        ),
    )
    parser.add_argument(
        "--dir", type=Path, required=True,
        help="directory holding dataset.npz and the CV split file",
    )
    parser.add_argument(
        "--folds-file", type=Path, default=None,
        help="split_cv_k{n}.npz from splits.py; provides fold indices and "
             "scalers. Default: split_cv_k{folds}.npz in --dir",
    )
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS,
                        help="overridden by the fold count recorded in the split file")
    parser.add_argument("--trials", type=int, default=config.N_TRIALS)
    parser.add_argument("--device", default=None,
                        help="torch device; default is cuda when available, else cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="load and check the data, then exit")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)

    in_dir = args.dir.resolve()
    dataset_path = in_dir / "dataset.npz"
    dataset = load_dataset(dataset_path)

    folds_path = args.folds_file or (
        in_dir / config.CV_FILENAME_TEMPLATE.format(folds=args.folds)
    )
    splits = load_splits(folds_path.resolve())

    n_folds_in_file = int(splits["folds"])
    if n_folds_in_file != args.folds:
        print(f"note: split file contains {n_folds_in_file} folds, "
              f"using that instead of --folds {args.folds}")
        args.folds = n_folds_in_file

    print(f"dataset   : {dataset_path}")
    print(f"device    : {describe_device(device)}")
    print(f"X         : {dataset['X'].shape}")
    print(f"Y         : {dataset['Y'].shape}")
    print(f"splits    : {folds_path}  ({args.folds} folds)")

    for f, tr, va in fold_indices(splits, args.folds):
        print(f"  fold {f}: train {tr.size:6d}  val {va.size:6d}")

    print(f"speed     : {benchmark_steps(device, dataset['X'].shape[1], dataset['Y'].shape[1])}")

    if args.dry_run:
        print("\ndry run: nothing trained")
        return 0

    log_path = dataset_path.parent / config.TUNE_LOG_FILENAME
    log_path.write_text("", encoding="utf-8")

    t0 = time.time()
    best, records = run_tuning(
        dataset, splits, args.folds, args.trials, device, log_path,
    )
    elapsed = time.time() - t0

    # ---- summary ----
    losses = np.array([r["val_loss_mean"] for r in records])
    phase = np.array([r["phase_err_mean_deg"] for r in records])
    dphi = np.array([r["dphi_err_mean_deg"] for r in records])
    dphi_ok = np.array([r["dphi_within_tol"] for r in records])

    print(f"\ntuned {len(records)} trials in {elapsed / 60:.1f} min")
    print(f"  val loss  : best {losses.min():.5f}   median {np.median(losses):.5f}")
    print(f"  phi error : best {phase.min():.2f} deg")
    print(f"  dphi error: best {dphi.min():.2f} deg")
    print(f"  dphi ok   : best {100 * dphi_ok.max():.0f}% within "
          f"{config.DPHI_TOL_DEG:.0f} deg")
    print("\nbest hyperparameters:")
    for k, v in best.items():
        print(f"  {k:<16} {v}")

    # ---- write ----
    out_dir = dataset_path.parent

    params_path = out_dir / config.BEST_PARAMS_FILENAME
    np.savez_compressed(
        params_path,
        **{k: np.array(v) for k, v in best.items()},
        val_loss_mean=np.float64(losses.min()),
        n_trials=np.int64(len(records)),
        folds=np.int64(args.folds),
        dataset=np.array(str(dataset_path)),
        splits=np.array(str(folds_path)),
    )
    print(f"\nwrote {params_path}")

    trials_path = out_dir / config.TRIALS_CSV_FILENAME
    if records:
        keys = list(records[0].keys())
        with open(trials_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {trials_path}")

    print(f"wrote {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
