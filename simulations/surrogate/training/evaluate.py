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


def circular_std_deg(angles_deg: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Circular standard deviation of wrapped angles, in degrees.

    A linear standard deviation is only meaningful for angles that share a
    branch. Two members predicting +179 and -179 degrees differ by 2 degrees in
    reality, but np.std reports 358. Because the phase sweeps the full circle as
    `a` varies, some fraction of any group sits near the wrap point, and those
    samples dominate an RMS aggregate: five percent of samples at 180 degrees
    among the rest at 2 degrees already gives about 40 degrees of apparent
    spread.

    The circular standard deviation is branch-independent. It is derived from the
    mean resultant length R of the unit vectors,

        sigma = sqrt(-2 ln R),   R = |mean(exp(i theta))|

    and reduces to the ordinary standard deviation for small spreads, which is
    the regime expected here.

    Parameters
    ----------
    angles_deg : array with the members along `axis`
    axis       : axis over which the members are stacked

    Returns
    -------
    Array of the same shape with `axis` removed.
    """
    a = np.deg2rad(angles_deg)
    c = np.cos(a).mean(axis=axis)
    s = np.sin(a).mean(axis=axis)
    r = np.hypot(c, s)
    # Clip away from zero: R -> 0 means the members are spread uniformly over
    # the circle, which corresponds to an unbounded standard deviation. Reporting
    # a large finite value keeps downstream arithmetic well defined.
    r = np.clip(r, 1e-12, 1.0)
    return np.rad2deg(np.sqrt(-2.0 * np.log(r)))


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


def group_metrics(
    pred_q: dict,
    true_q: dict,
    tol_deg: float,
    phi_sigma: np.ndarray | None = None,
    dphi_sigma: np.ndarray | None = None,
) -> dict:
    """
    Metrics for one group of samples.

    `phi_sigma` and `dphi_sigma` are the ensemble spreads per sample. Their means
    over the group are reported alongside the errors, so a group where the model
    is both wrong and confident can be told apart from one where it is wrong but
    at least knows it. That distinction is what makes the uncertainty usable for
    deciding where to simulate next.
    """
    phase_err = _wrap(pred_q["phi_major"] - true_q["phi_major"])
    dphi_err = _wrap(pred_q["dphi"] - true_q["dphi"])

    mag_err = 0.5 * (
        np.abs(pred_q["mag_major"] - true_q["mag_major"])
        + np.abs(pred_q["mag_minor"] - true_q["mag_minor"])
    )
    ar_err = np.abs(pred_q["ar_db"] - true_q["ar_db"])

    n = phase_err.size
    out = {
        "n": int(n),
        # Both error conventions are reported. MAE is the intuitive "typical
        # element is off by this much"; RMS weights large errors quadratically,
        # which is what matters for a design decision, since one badly
        # mispredicted element can send the search to the wrong geometry.
        "phase_err_mae": float(phase_err.mean()),
        "phase_err_rms": float(np.sqrt((phase_err ** 2).mean())),
        "phase_err_p95": float(np.quantile(phase_err, 0.95)),
        "dphi_err_mae": float(dphi_err.mean()),
        "dphi_err_rms": float(np.sqrt((dphi_err ** 2).mean())),
        "dphi_err_p95": float(np.quantile(dphi_err, 0.95)),
        "dphi_within": float((dphi_err <= tol_deg).mean()),
        "mag_err_mean": float(mag_err.mean()),
        "ar_err_mean_db": float(ar_err.mean()),
    }

    if phi_sigma is not None:
        # sigma is the spread of a SINGLE member at a fixed input, and is itself
        # a spread, so it is aggregated over a group as a root mean square rather
        # than an arithmetic mean: mean(sigma) <= rms(sigma) by Jensen.
        out["sigma_phi_mean"] = float(phi_sigma.mean())
        out["sigma_phi_rms"] = float(np.sqrt((phi_sigma ** 2).mean()))
        out["sigma_phi_p95"] = float(np.quantile(phi_sigma, 0.95))

        # The error above is that of the ENSEMBLE MEAN, while sigma is that of a
        # single member. Averaging M roughly independent predictions reduces the
        # scatter about the truth by sqrt(M), so the mean should sit about
        # sigma/sqrt(M) from the truth, not sigma.
        #
        # Both sides of the calibration check are therefore second-moment
        # quantities: RMS error against the RMS of the per-sample sigma divided
        # by sqrt(M). Their ratio has an expected value of 1 for a calibrated
        # ensemble, with no conversion constant.
        n_members = float(config.N_ENSEMBLE)
        sigma_of_mean = phi_sigma / np.sqrt(n_members)

        out["sigma_mean_phi"] = float(sigma_of_mean.mean())
        out["sigma_mean_phi_rms"] = float(np.sqrt((sigma_of_mean ** 2).mean()))

        rms_err = float(np.sqrt((phase_err ** 2).mean()))
        rms_sigma_mean = out["sigma_mean_phi_rms"]
        out["z_phi"] = (rms_err / rms_sigma_mean
                        if rms_sigma_mean > 1e-12 else float("nan"))

        # Coverage is computed per sample against the per-sample interval, so it
        # is unaffected by the choice of group aggregate.
        out["coverage_95"] = float(
            (phase_err <= 1.96 * sigma_of_mean).mean()
        )
        # Member-level figures, kept for reference and for the plots.
        out["coverage_95_member"] = float(
            (phase_err <= 1.96 * phi_sigma).mean()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            z_member = np.where(phi_sigma > 1e-12, phase_err / phi_sigma, np.nan)
        out["z_phi_member"] = (float(np.nanmean(z_member))
                               if np.any(np.isfinite(z_member)) else float("nan"))

    if dphi_sigma is not None:
        out["sigma_dphi_mean"] = float(dphi_sigma.mean())
        out["sigma_dphi_rms"] = float(np.sqrt((dphi_sigma ** 2).mean()))
        out["sigma_dphi_p95"] = float(np.quantile(dphi_sigma, 0.95))

        sigma_of_mean = dphi_sigma / np.sqrt(float(config.N_ENSEMBLE))
        out["sigma_mean_dphi_rms"] = float(np.sqrt((sigma_of_mean ** 2).mean()))

        rms_err = float(np.sqrt((dphi_err ** 2).mean()))
        rms_sigma_mean = out["sigma_mean_dphi_rms"]
        out["z_dphi"] = (rms_err / rms_sigma_mean
                         if rms_sigma_mean > 1e-12 else float("nan"))
        out["coverage_95_dphi"] = float(
            (dphi_err <= 1.96 * sigma_of_mean).mean()
        )

    return out


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
    """
    One line of the report.

    MAE and RMS are both reported for each of the two phase quantities. MAE is
    the intuitive "typical element is off by this much"; RMS weights large
    errors quadratically, which is what matters for a design decision, since one
    badly mispredicted element can send the search to the wrong geometry.

    The spread columns are the per-sample ensemble spread aggregated over the
    group as a root mean square, since a spread is itself a second-moment
    quantity and averaging it arithmetically would understate it.
    """
    base = (
        f"  {label:<22} {m['n']:>6d}"
        f"  {m['phase_err_mae']:6.2f} {m['phase_err_rms']:6.2f}"
        f"  {m['dphi_err_mae']:6.2f} {m['dphi_err_rms']:6.2f}"
        f"  {m['mag_err_mean']:7.4f}"
    )
    if "sigma_phi_rms" in m:
        base += (
            f"  {m['sigma_phi_rms']:6.2f} {m['sigma_dphi_rms']:6.2f}"
            f"  {m['z_phi']:5.2f}"
        )
    if "coverage_95" in m:
        base += f"  {100 * m['coverage_95']:4.0f}%"
    return base


def print_table_header(header: str) -> None:
    """Column header matching fmt_row."""
    print(f"  {header:<22} {'n':>6}"
          f"  {'MAE':>6} {'RMS':>6}"
          f"  {'MAE':>6} {'RMS':>6}"
          f"  {'mag':>7}"
          f"  {'s_phi':>6} {'s_dphi':>6}"
          f"  {'z':>5}  {'cov':>5}")
    print(f"  {'':<22} {'':>6}"
          f"  {'phi':>6} {'phi':>6}"
          f"  {'dphi':>6} {'dphi':>6}"
          f"  {'':>7}"
          f"  {'':>6} {'':>6}"
          f"  {'':>5}  {'':>5}")
    print("  " + "-" * 92)


def print_legend() -> None:
    """Column key, printed once before the tables."""
    print(REPORT_KEY)


# The column key, shared between the console output and the two report files so
# the three cannot drift apart.
REPORT_KEY = """
column key
  n         number of test samples in this group
  phi MAE   mean |error| of phi_major, degrees
  phi RMS   root mean square error of phi_major, degrees
  dphi MAE  mean |error| of the inter-axis offset, degrees
  dphi RMS  root mean square error of the inter-axis offset, degrees
  mag       mean |error| of |R|, averaged over both axes (linear, 0..1)
  s_phi     ensemble spread of phi_major, degrees. A CIRCULAR standard
            deviation across the members at a fixed input, so it is branch
            independent. This is the uncertainty of ONE member, not of the
            ensemble mean
  s_dphi    same, for the inter-axis offset
  z         phi RMS error divided by s_phi/sqrt(M); 1 means calibrated
  cov       fraction whose phi error falls inside the claimed 95% interval

Why the spread is circular
  phi_major and dphi are wrapped angles. A linear standard deviation is only
  meaningful for angles sharing a branch: two members predicting +179 and -179
  degrees differ by 2 degrees, but np.std reports 358. Because the phase sweeps
  the full circle as `a` varies, some fraction of any group sits near the wrap
  point, and those samples dominate an RMS aggregate -- five percent at 180
  degrees among the rest at 2 degrees already gives about 40 degrees of apparent
  spread.

  The circular standard deviation, sqrt(-2 ln R) with R the mean resultant
  length, is branch independent and reduces to the ordinary standard deviation
  for small spreads.

Why z and cov are computed this way
  The error is that of the ENSEMBLE MEAN. The spread s_phi is that of a SINGLE
  member at a fixed input. Averaging M roughly independent predictions reduces
  the scatter about the truth by sqrt(M), so the mean should sit about
  s_phi/sqrt(M) from the truth, not s_phi.

  Both sides of z are second-moment quantities, RMS error against RMS spread, so
  a calibrated ensemble gives z ~ 1 with no conversion constant. Comparing MAE
  against a spread instead would require the factor sqrt(2/pi) ~ 0.8, which is a
  property of the Gaussian, not of the ensemble.

  cov is built per sample from the same s_phi/sqrt(M) interval. 95% is
  calibrated; well below means the spread understates the error of the mean,
  which happens where the members are strongly correlated.

Why both MAE and RMS are reported
  MAE is the intuitive typical error. RMS weights large errors quadratically,
  which is what matters for a design decision: one badly mispredicted element
  can send the search to the wrong geometry even when the typical element is
  accurate.
"""


def print_legend() -> None:
    """
    Column key, printed once before the tables.

    The tables are dense and the abbreviations are not self-explanatory, so a
    key costs a few lines and saves a reader guessing.
    """
    print("""
column key
  n         number of test samples in this group
  phi       mean |error| of phi_major, degrees (the absolute phase)
  dphi      mean |error| of the inter-axis offset phi_minor - phi_major, degrees
  (%)       fraction of samples whose dphi error is within tolerance
  mag       mean |error| of |R|, averaged over both axes (linear, 0..1)
  ar        mean |error| of the axial ratio implied by the predicted
            coefficients, dB
  s_phi     ensemble spread of phi_major, degrees: the standard deviation
            across the members at a fixed input. This is the uncertainty of
            ONE member, not of the ensemble mean
  s_dphi    same, for the inter-axis offset
  z         mean of |error| / sigma taken per sample. Because the error is
            that of the ensemble mean and sigma is that of a single member,
            z is expected to be near 1/sqrt(M) ~ 0.32 for M = 10 roughly
            independent members. Reading it against 0.8 assumes the two
            measure the same thing, which they do not
  cov       fraction whose error falls inside the claimed 1.96 sigma.
            Near 100% for the same reason: sigma describes a member, the
            error describes the mean of ten of them
""")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="directory holding dataset.npz, the splits, the "
                             "ensemble and the evaluation CSVs")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for the CSVs and report; created if "
                             "absent. Default: the same as --dir")
    parser.add_argument("--ensemble", type=Path, default=None,
                        help="ensemble.pt; default in --dir")
    parser.add_argument("--meta", type=Path, default=None,
                        help="ensemble_meta.npz; default in --dir")
    parser.add_argument("--holdout", type=Path, default=None,
                        help="split_holdout.npz; default in --dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tol", type=float, default=config.DPHI_TOL_DEG,
                        help="tolerance on the offset, degrees")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    in_dir = args.dir.resolve()
    out_dir = args.out.resolve() if args.out else in_dir

    dataset_path = in_dir / "dataset.npz"
    dataset = load_npz(dataset_path, "dataset")
    meta = load_npz((args.meta or in_dir / config.ENSEMBLE_META_FILENAME).resolve(),
                    "ensemble metadata")
    holdout = load_npz((args.holdout or in_dir / config.HOLDOUT_FILENAME).resolve(),
                       "holdout split")

    params, states, n_in, n_out = load_ensemble(
        (args.ensemble or in_dir / config.ENSEMBLE_MODEL_FILENAME).resolve()
    )

    n_dataset = dataset["X"].shape[0]
    if "n_dataset" in holdout and int(holdout["n_dataset"]) != n_dataset:
        raise RuntimeError(
            f"split file was generated for a dataset of "
            f"{int(holdout['n_dataset'])} samples but this one has {n_dataset}; "
            f"regenerate the splits"
        )

    # Split indices address the FILTERED subset, so map them onto the dataset on
    # disk before using them.
    keep_idx = holdout["keep_idx"] if "keep_idx" in holdout else np.arange(n_dataset)
    test_idx = keep_idx[holdout["test_idx"]]

    print(f"dataset  : {dataset_path}")
    print(f"ensemble : {len(states)} members, "
          f"{params['n_layers']}x{params['width']} {params['activation']}")
    print(f"test     : {test_idx.size} samples")
    print(f"tol      : {args.tol} deg")
    print(f"writing  : {out_dir}")

    if args.dry_run:
        print("\ndry run: nothing evaluated")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    print_legend()

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
    #
    # A CIRCULAR standard deviation is used, not a linear one: both phi_major and
    # dphi are wrapped angles, and members landing on opposite sides of the wrap
    # point would otherwise register as disagreeing by nearly the full circle.
    member_q = [derive_quantities(member_pred[m], y_mean, y_std)
                for m in range(member_pred.shape[0])]
    dphi_sigma = circular_std_deg(
        np.stack([q["dphi"] for q in member_q], axis=0), axis=0
    )
    phi_sigma = circular_std_deg(
        np.stack([q["phi_major"] for q in member_q], axis=0), axis=0
    )

    # ---- optional exclusion of the circular cross-section ----
    # Applied to the reported metrics only, after prediction. The circular cells
    # are kept in training so the model interpolates correctly down to
    # ratio = 1; here they are excluded because the offset is ill-conditioned and
    # identically zero for them, so they are not candidate CP elements.
    raw_names = [str(n) for n in dataset["raw_names"]]
    p_ratio = dataset["X_raw"][test_idx, raw_names.index("p_ratio")]
    eps_r = dataset["X_raw"][test_idx, raw_names.index("eps_r")]
    ratio = dataset["X_raw"][test_idx, raw_names.index("ratio")]
    a_mm = dataset["X_raw"][test_idx, raw_names.index("a_mm")]

    keep = np.ones(test_idx.size, dtype=bool)
    n_kept = int(test_idx.size)
    if config.EXCLUDE_RATIO_1:
        circular = np.abs(ratio - 1.0) <= config.CIRCULAR_RATIO_TOL
        keep &= ~circular
        n_excluded = int(np.count_nonzero(circular))
        print(f"\nexcluding {n_excluded} circular samples (ratio = 1) from the "
              f"reported metrics; {int(keep.sum())} remain")
        if not np.any(keep):
            print("error: the exclusion removed every test sample")
            return 1

        pred_q = {k: v[keep] for k, v in pred_q.items()}
        true_q = {k: v[keep] for k, v in true_q.items()}
        phi_sigma = phi_sigma[keep]
        dphi_sigma = dphi_sigma[keep]
        p_ratio = p_ratio[keep]
        eps_r = eps_r[keep]
        ratio = ratio[keep]
        a_mm = a_mm[keep]
        n_kept = int(keep.sum())

    # ---- overall ----
    overall = group_metrics(pred_q, true_q, args.tol, phi_sigma, dphi_sigma)

    print("\n=== overall ===")
    print_table_header("all")
    print(fmt_row("all", overall, args.tol))

    # ---- by stratum ----
    # n_kept, not test_idx.size: the arrays were already reduced by the circular
    # exclusion, so the masks must match their current length.
    n_kept = pred_q["phi_major"].size
    strata_rows: list[dict] = []
    print("\n=== by stratum ===")
    print_table_header("stratum")
    for spec in config.STRATA:
        eps_wanted = spec.get("eps_r")
        p_wanted = spec.get("p_ratio")

        mask = np.ones(n_kept, dtype=bool)
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
        m = group_metrics(sub_pred, sub_true, args.tol,
                          phi_sigma[mask], dphi_sigma[mask])
        m["stratum"] = spec["name"]
        strata_rows.append(m)
        print(fmt_row(spec["name"], m, args.tol))

    # ---- by ratio within stratum ----
    ratio_rows: list[dict] = []
    print("\n=== by ratio within stratum ===")
    for spec in config.STRATA:
        eps_wanted = spec.get("eps_r")
        p_wanted = spec.get("p_ratio")

        cell = np.ones(n_kept, dtype=bool)
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
        print_table_header("ratio")
        for r in np.unique(ratio[cell]):
            mask = cell & np.isclose(ratio, r)
            if not np.any(mask):
                continue
            sub_pred = {k: v[mask] for k, v in pred_q.items()}
            sub_true = {k: v[mask] for k, v in true_q.items()}
            m = group_metrics(sub_pred, sub_true, args.tol,
                              phi_sigma[mask], dphi_sigma[mask])
            m["stratum"] = spec["name"]
            m["ratio"] = float(r)
            ratio_rows.append(m)
            print(fmt_row(f"ratio {r:.2f}", m, args.tol))

    # ---- by tuning variable a ----
    # This is the extrapolation diagnostic. The tuning variable is the axis the
    # model sweeps, and the phase slope steepens towards resonance, so the
    # ensemble spread should grow towards the ends of the interval if the model
    # knows it is less sure there. A flat sigma with a rising error means it does
    # not, and the uncertainty cannot be used to filter predictions.
    a_rows: list[dict] = []

    print("\n=== by tuning variable a (extrapolation diagnostic) ===")
    print(f"  {'a bin (mm)':<22} {'n':>6}"
          f"  {'MAE':>6} {'RMS':>6}"
          f"  {'MAE':>6} {'RMS':>6}"
          f"  {'mag':>7}"
          f"  {'s_phi':>6} {'s_dphi':>6}"
          f"  {'z':>5}  {'cov':>5}")
    print(f"  {'':<22} {'':>6}"
          f"  {'phi':>6} {'phi':>6}"
          f"  {'dphi':>6} {'dphi':>6}"
          f"  {'':>7}"
          f"  {'':>6} {'':>6}"
          f"  {'':>5}  {'':>5}")
    print("  " + "-" * 92)

    edges = np.linspace(a_mm.min(), a_mm.max(), config.EVAL_A_BINS + 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # Include the right edge in the last bin so the maximum is not lost.
        mask = (a_mm >= lo) & (a_mm < hi) if i < len(edges) - 2 else (a_mm >= lo)
        if not np.any(mask):
            continue

        sub_pred = {k: v[mask] for k, v in pred_q.items()}
        sub_true = {k: v[mask] for k, v in true_q.items()}
        m = group_metrics(sub_pred, sub_true, args.tol,
                          phi_sigma[mask], dphi_sigma[mask])
        m["a_lo"] = float(lo)
        m["a_hi"] = float(hi)
        a_rows.append(m)

        print(fmt_row(f"{lo:.3f}-{hi:.3f}", m, args.tol))

    # ---- calibration ----
    phase_err = _wrap(pred_q["phi_major"] - true_q["phi_major"])
    # The interval is built from the spread expected of the MEAN, not of a
    # single member, for the reason given in the column key.
    phi_sigma_of_mean = phi_sigma / np.sqrt(float(config.N_ENSEMBLE))
    calib_rows: list[dict] = []

    print("\n=== calibration of the ensemble spread ===")
    print(f"  {'sigma bin':<18} {'n':>7} {'inside 1.96s':>13} {'expected':>9}")
    if np.all(phi_sigma_of_mean < 1e-9):
        print("  (all members identical, no spread to calibrate)")
    else:
        edges = np.quantile(phi_sigma_of_mean, np.linspace(0, 1, 6))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (phi_sigma_of_mean >= lo) & (phi_sigma_of_mean <= hi)
            if not np.any(mask):
                continue
            inside = float(
                (phase_err[mask] <= 1.96 * phi_sigma_of_mean[mask]).mean()
            )
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
    write_csv(out_dir / config.EVAL_A_FILENAME, a_rows)
    write_csv(out_dir / config.EVAL_CALIBRATION_FILENAME, calib_rows)

    report_path = out_dir / config.EVAL_REPORT_FILENAME
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("ensemble evaluation\n")
        fh.write(f"dataset  : {dataset_path}\n")
        fh.write(f"members  : {len(states)} "
                 f"({params['n_layers']}x{params['width']} {params['activation']})\n")
        fh.write(f"test     : {test_idx.size} samples"
                 + (f" ({n_kept} after excluding ratio = 1)"
                    if config.EXCLUDE_RATIO_1 else "") + "\n")
        fh.write(f"tol      : {args.tol} deg\n")
        fh.write(REPORT_KEY)

        header = (
            f"  {'':<22} {'n':>6}"
            f"  {'MAE':>6} {'RMS':>6}"
            f"  {'MAE':>6} {'RMS':>6}"
            f"  {'mag':>7}"
            f"  {'s_phi':>6} {'s_dphi':>6}"
            f"  {'z':>5}  {'cov':>5}\n"
        )
        sub = (
            f"  {'':<22} {'':>6}"
            f"  {'phi':>6} {'phi':>6}"
            f"  {'dphi':>6} {'dphi':>6}"
            f"  {'':>7}"
            f"  {'':>6} {'':>6}"
            f"  {'':>5}  {'':>5}\n"
        )

        fh.write("=== overall ===\n")
        fh.write(header)
        fh.write(sub)
        fh.write(fmt_row("all", overall, args.tol) + "\n")

        fh.write("\n=== by stratum ===\n")
        fh.write(header.replace(f"{'':<22}", f"{'stratum':<22}", 1))
        fh.write(sub)
        for row in strata_rows:
            fh.write(fmt_row(row["stratum"], row, args.tol) + "\n")

        fh.write("\n=== by tuning variable a ===\n")
        fh.write(header.replace(f"{'':<22}", f"{'a bin (mm)':<22}", 1))
        fh.write(sub)
        for row in a_rows:
            label = f"{row['a_lo']:.3f}-{row['a_hi']:.3f}"
            fh.write(fmt_row(label, row, args.tol) + "\n")

    print(f"\nwrote {report_path}")

    # ---- full report, including the per-ratio breakdown ----
    # Kept separate from the short report so the latter stays readable when
    # pasted into a document, while the per-ratio detail is still on disk.
    full_path = out_dir / config.EVAL_REPORT_FULL_FILENAME
    with open(full_path, "w", encoding="utf-8") as fh:
        fh.write("ensemble evaluation, full detail\n")
        fh.write(f"dataset  : {dataset_path}\n")
        fh.write(f"members  : {len(states)} "
                 f"({params['n_layers']}x{params['width']} {params['activation']})\n")
        fh.write(f"sampling : {meta['sampling'] if 'sampling' in meta else 'unknown'}\n")
        fh.write(f"test     : {test_idx.size} samples"
                 + (f" ({n_kept} after excluding ratio = 1)"
                    if config.EXCLUDE_RATIO_1 else "") + "\n")
        fh.write(f"tol      : {args.tol} deg\n")
        fh.write(REPORT_KEY)

        fh.write("=== overall ===\n")
        fh.write(header)
        fh.write(sub)
        fh.write(fmt_row("all", overall, args.tol) + "\n")

        fh.write("\n=== by stratum ===\n")
        fh.write(header.replace(f"{'':<22}", f"{'stratum':<22}", 1))
        fh.write(sub)
        for row in strata_rows:
            fh.write(fmt_row(row["stratum"], row, args.tol) + "\n")

        fh.write("\n=== by ratio within stratum ===\n")
        current = None
        for row in ratio_rows:
            if row["stratum"] != current:
                current = row["stratum"]
                fh.write(f"\n-- {current} --\n")
                fh.write(header.replace(f"{'':<22}", f"{'ratio':<22}", 1))
                fh.write(sub)
            fh.write(fmt_row(f"{row['ratio']:.2f}", row, args.tol) + "\n")

        fh.write("\n=== by tuning variable a ===\n")
        fh.write(header.replace(f"{'':<22}", f"{'a bin (mm)':<22}", 1))
        fh.write(sub)
        for row in a_rows:
            label = f"{row['a_lo']:.3f}-{row['a_hi']:.3f}"
            fh.write(fmt_row(label, row, args.tol) + "\n")

        fh.write("\n=== calibration of the ensemble spread ===\n")
        fh.write(f"  {'sigma lo':>9} {'sigma hi':>9} {'n':>7} "
                 f"{'inside':>8} {'expected':>9}\n")
        for row in calib_rows:
            fh.write(f"  {row['sigma_lo']:9.3f} {row['sigma_hi']:9.3f} "
                     f"{row['n']:7d} {100 * row['inside_frac']:7.1f}% "
                     f"{100 * row['expected']:8.1f}%\n")

    print(f"wrote {full_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
