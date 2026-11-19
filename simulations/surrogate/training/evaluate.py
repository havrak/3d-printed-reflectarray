"""
Evaluate the trained ensemble on the holdout, with per-stratum detail.

Loads the ensemble written by ensemble_train.py and the holdout split written by
splits.py, predicts the test rows with every member, and reports the metrics
that matter for the design task rather than the training loss.

This script is the only consumer of the holdout. Run it once, at the end.

Usage
-----
    python evaluate.py --dataset path/to/dataset.npz
    python evaluate.py --dataset ... --strata-by-ratio
    python evaluate.py --dataset ... --dry-run

Reported metrics
----------------
For every grouping (overall, per stratum, per ratio within stratum):

    phase_err      absolute error of phi_major, degrees
    dphi_err       error of the inter-axis offset, degrees
    dphi_within    fraction whose predicted offset is within DPHI_TOL_DEG
    mag_err        error of |R| for both axes
    ar_err         axial-ratio error implied by the predicted coefficients

The offset is reported separately from the absolute phase because the two are
not implied by each other: the axes can be wrong in compensating ways, which
gives a good absolute error and a correct offset, or in opposing ways, which
gives the same absolute error and a drifted offset. The offset is what the
circular-polarisation condition depends on.

Uncertainty
-----------
The ensemble spread is reported alongside the error, and a calibration table
bins predictions by predicted sigma and compares the fraction falling within
+-1.96 sigma against the nominal 95%. Systematic under-coverage means the
intervals are overconfident and should be recalibrated before being used to
guide sampling.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_npz(path: Path, what: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_ensemble(path: Path):
    """Load the ensemble checkpoint and return (params, states, n_in, n_out)."""
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"ensemble not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return (
        ckpt["params"],
        ckpt["states"],
        int(ckpt["n_inputs"]),
        int(ckpt["n_outputs"]),
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict_members(
    states: list,
    params: dict,
    X: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Run every ensemble member over X.

    Returns an array of shape (n_members, n_samples, n_outputs), in standardised
    space. The members share the architecture but differ in initialisation and
    data ordering, so their spread is the epistemic uncertainty.
    """
    import torch

    import model as net

    out = []
    for state in states:
        model = net.build_mlp(
            n_inputs=X.shape[1],
            n_outputs=5,
            n_layers=params["n_layers"],
            width=params["width"],
            activation=params["activation"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=device)
            out.append(model(xt).cpu().numpy())

    return np.stack(out, axis=0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _wrap(err_deg: np.ndarray) -> np.ndarray:
    """Wrap an angular difference to (-180, 180]."""
    return np.abs(np.remainder(err_deg + 180.0, 360.0) - 180.0)


def derive_quantities(block: np.ndarray, y_mean, y_std):
    """
    Turn a (..., 5) block of standardised targets into physical quantities.

    Columns are Re(major), Im(major), Re(minor), Im(minor), slope.

    Returns a dict of arrays: phi_major, phi_minor, dphi, mag_major, mag_minor,
    axial_ratio_db.
    """
    phys = block[..., :4] * y_std[:4] + y_mean[:4]

    re_maj, im_maj = phys[..., 0], phys[..., 1]
    re_min, im_min = phys[..., 2], phys[..., 3]

    phi_major = np.rad2deg(np.arctan2(im_maj, re_maj))
    phi_minor = np.rad2deg(np.arctan2(im_min, re_min))

    mag_major = np.hypot(re_maj, im_maj)
    mag_minor = np.hypot(re_min, im_min)

    # Axial ratio from the two orthogonal components. For a CP wave the ratio of
    # the two circular components sets the AR; here the two LP components with
    # their phase offset are combined directly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mag_minor / mag_major
        ar_db = 20.0 * np.log10(np.maximum(ratio, 1e-12))

    return {
        "phi_major": phi_major,
        "phi_minor": phi_minor,
        "dphi": np.remainder(phi_minor - phi_major, 360.0),
        "mag_major": mag_major,
        "mag_minor": mag_minor,
        "ar_db": ar_db,
    }


def group_metrics(pred_q: dict, true_q: dict, tol_deg: float) -> dict:
    """Metrics for one group of samples."""
    phase_err = _wrap(pred_q["phi_major"] - true_q["phi_major"])
    dphi_err = _wrap(pred_q["dphi"] - true_q["dphi"])

    mag_err = 0.5 * (
        np.abs(pred_q["mag_major"] - true_q["mag_major"])
        + np.abs(pred_q["mag_minor"] - true_q["mag_minor"])
    )
    ar_err = np.abs(pred_q["ar_db"] - true_q["ar_db"])

    n = phase_err.size
    return {
        "n": int(n),
        "phase_err_mean": float(phase_err.mean()),
        "phase_err_p95": float(np.quantile(phase_err, 0.95)),
        "dphi_err_mean": float(dphi_err.mean()),
        "dphi_err_p95": float(np.quantile(dphi_err, 0.95)),
        "dphi_within": float((dphi_err <= tol_deg).mean()),
        "mag_err_mean": float(mag_err.mean()),
        "ar_err_mean_db": float(ar_err.mean()),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def fmt_row(label: str, m: dict, tol: float) -> str:
    return (
        f"  {label:<24} n={m['n']:<6d} "
        f"phi {m['phase_err_mean']:6.2f} "
        f"dphi {m['dphi_err_mean']:6.2f} "
        f"({100 * m['dphi_within']:5.1f}%) "
        f"mag {m['mag_err_mean']:.4f} "
        f"ar {m['ar_err_mean_db']:5.2f} dB"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, default=None,
                        help="ensemble.pt; default beside the dataset")
    parser.add_argument("--meta", type=Path, default=None,
                        help="ensemble_meta.npz; default beside the dataset")
    parser.add_argument("--holdout", type=Path, default=None,
                        help="split_holdout.npz; default beside the dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tol", type=float, default=config.DPHI_TOL_DEG,
                        help="tolerance on the offset, degrees")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = args.dataset.resolve()
    out_dir = dataset_path.parent

    dataset = load_npz(dataset_path, "dataset")
    meta = load_npz((args.meta or out_dir / config.ENSEMBLE_META_FILENAME).resolve(),
                    "ensemble metadata")
    holdout = load_npz((args.holdout or out_dir / config.HOLDOUT_FILENAME).resolve(),
                       "holdout split")

    params, states, n_in, n_out = load_ensemble(
        (args.ensemble or out_dir / config.ENSEMBLE_MODEL_FILENAME).resolve()
    )

    test_idx = holdout["test_idx"]
    if int(holdout["n_samples"]) != dataset["X"].shape[0]:
        raise RuntimeError(
            f"holdout refers to {int(holdout['n_samples'])} samples but the "
            f"dataset has {dataset['X'].shape[0]}; regenerate the splits"
        )

    print(f"dataset  : {dataset_path}")
    print(f"ensemble : {len(states)} members, "
          f"{params['n_layers']}x{params['width']} {params['activation']}")
    print(f"test     : {test_idx.size} samples")
    print(f"tol      : {args.tol} deg")

    if args.dry_run:
        print("\ndry run: nothing evaluated")
        return 0

    # ---- predict ----
    x_mean, x_std = meta["x_mean"], meta["x_std"]
    y_mean, y_std = meta["y_mean"], meta["y_std"]

    X_test_raw = dataset["X"][test_idx]
    Y_test_raw = dataset["Y"][test_idx]

    X_test = (X_test_raw - x_mean) / x_std
    Y_test_std = (Y_test_raw - y_mean) / y_std

    member_pred = predict_members(states, params, X_test, args.device)
    # (n_members, n_test, 5)

    pred_std = member_pred.mean(axis=0)
    pred_sigma = member_pred.std(axis=0)

    pred_q = derive_quantities(pred_std, y_mean, y_std)
    true_q = derive_quantities(Y_test_std, y_mean, y_std)

    # Per-member quantities, so the spread can be taken over the derived
    # nonlinear quantities rather than propagated linearly.
    member_q = [derive_quantities(member_pred[m], y_mean, y_std)
                for m in range(member_pred.shape[0])]
    dphi_sigma = np.std(np.stack([q["dphi"] for q in member_q], axis=0), axis=0)
    phi_sigma = np.std(np.stack([q["phi_major"] for q in member_q], axis=0), axis=0)

    # ---- overall ----
    overall = group_metrics(pred_q, true_q, args.tol)

    print("\n=== overall ===")
    print(fmt_row("all", overall, args.tol))

    # ---- by stratum ----
    raw_names = [str(n) for n in dataset["raw_names"]]
    p_ratio = dataset["X_raw"][test_idx, raw_names.index("p_ratio")]
    eps_r = dataset["X_raw"][test_idx, raw_names.index("eps_r")]
    ratio = dataset["X_raw"][test_idx, raw_names.index("ratio")]

    strata_rows: list[dict] = []
    print("\n=== by stratum ===")
    for spec in config.STRATA:
        eps_wanted = spec.get("eps_r")
        p_wanted = spec.get("p_ratio")

        mask = np.ones(test_idx.size, dtype=bool)
        if p_wanted is not None:
            mask &= np.isclose(p_ratio, p_wanted)
        if eps_wanted is not None:
            if isinstance(eps_wanted, tuple):
                mask &= np.isin(eps_r, eps_wanted)
            else:
                mask &= np.isclose(eps_r, eps_wanted)

        if not np.any(mask):
            print(f"  {spec['name']:<24} (no test samples)")
            continue

        sub_pred = {k: v[mask] for k, v in pred_q.items()}
        sub_true = {k: v[mask] for k, v in true_q.items()}
        m = group_metrics(sub_pred, sub_true, args.tol)
        m["stratum"] = spec["name"]
        strata_rows.append(m)
        print(fmt_row(spec["name"], m, args.tol))

    # ---- by ratio within stratum ----
    ratio_rows: list[dict] = []
    print("\n=== by ratio within stratum ===")
    for spec in config.STRATA:
        eps_wanted = spec.get("eps_r")
        p_wanted = spec.get("p_ratio")

        cell = np.ones(test_idx.size, dtype=bool)
        if p_wanted is not None:
            cell &= np.isclose(p_ratio, p_wanted)
        if eps_wanted is not None:
            if isinstance(eps_wanted, tuple):
                cell &= np.isin(eps_r, eps_wanted)
            else:
                cell &= np.isclose(eps_r, eps_wanted)

        if not np.any(cell):
            continue

        print(f"  -- {spec['name']} --")
        for r in np.unique(ratio[cell]):
            mask = cell & np.isclose(ratio, r)
            if not np.any(mask):
                continue
            sub_pred = {k: v[mask] for k, v in pred_q.items()}
            sub_true = {k: v[mask] for k, v in true_q.items()}
            m = group_metrics(sub_pred, sub_true, args.tol)
            m["stratum"] = spec["name"]
            m["ratio"] = float(r)
            ratio_rows.append(m)
            print(fmt_row(f"ratio {r:.2f}", m, args.tol))

    # ---- calibration ----
    phase_err = _wrap(pred_q["phi_major"] - true_q["phi_major"])
    calib_rows: list[dict] = []

    print("\n=== calibration of the ensemble spread ===")
    print(f"  {'sigma bin':<18} {'n':>7} {'inside 1.96s':>13} {'expected':>9}")
    if np.all(phi_sigma < 1e-9):
        print("  (all members identical, no spread to calibrate)")
    else:
        edges = np.quantile(phi_sigma, np.linspace(0, 1, 6))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (phi_sigma >= lo) & (phi_sigma <= hi)
            if not np.any(mask):
                continue
            inside = float((phase_err[mask] <= 1.96 * phi_sigma[mask]).mean())
            calib_rows.append({
                "sigma_lo": float(lo), "sigma_hi": float(hi),
                "n": int(mask.sum()), "inside_frac": inside,
                "expected": 0.95,
            })
            print(f"  {lo:7.3f} - {hi:7.3f}  {int(mask.sum()):7d} "
                  f"{100 * inside:12.1f}% {95.0:8.1f}%")

    # ---- write ----
    summary_row = {
        "group": "all",
        **overall,
        "n_members": len(states),
        "tol_deg": args.tol,
        "mean_sigma_phi": float(phi_sigma.mean()),
        "mean_sigma_dphi": float(dphi_sigma.mean()),
    }
    write_csv(out_dir / config.EVAL_SUMMARY_FILENAME, [summary_row])
    write_csv(out_dir / config.EVAL_STRATA_FILENAME, strata_rows)
    write_csv(out_dir / config.EVAL_RATIO_FILENAME, ratio_rows)
    write_csv(out_dir / config.EVAL_CALIBRATION_FILENAME, calib_rows)

    report_path = out_dir / config.EVAL_REPORT_FILENAME
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("ensemble evaluation\n")
        fh.write(f"dataset  : {dataset_path}\n")
        fh.write(f"members  : {len(states)}\n")
        fh.write(f"params   : {params}\n")
        fh.write(f"test     : {test_idx.size} samples\n")
        fh.write(f"tol      : {args.tol} deg\n\n")
        fh.write(fmt_row("all", overall, args.tol) + "\n\n")
        for row in strata_rows:
            fh.write(fmt_row(row["stratum"], row, args.tol) + "\n")
    print(f"\nwrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
