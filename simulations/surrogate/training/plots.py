"""
Diagnostic plots for the ensemble evaluation.

Reads the CSVs written by evaluate.py and produces the figures that answer what
the tables cannot answer at a glance: does the ensemble spread track the actual
error, and does either grow towards the edges of the domain.

Run after evaluate.py:

    python plots.py --dataset path/to/dataset.npz

Requires numpy and matplotlib. Saves PDFs next to the dataset, so they can be
included in the thesis directly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict]:
    """Read a CSV written by evaluate.py, converting numeric fields."""
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))

    rows = []
    for row in raw:
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = value
        rows.append(parsed)
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_error_vs_a(rows: list[dict], out_path: Path, tol: float) -> None:
    """
    Error and both ensemble spreads against the tuning variable.

    Three panels sharing the x axis: the absolute phase error with its spread,
    the offset error with its spread, and the within-tolerance fraction. The two
    spreads are shown separately because they measure different things and the
    offset is the one the CP condition depends on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_mid = [(r["a_lo"] + r["a_hi"]) / 2.0 for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    ax1.plot(a_mid, [r["phase_err_mae"] for r in rows], "o-",
             label="MAE", color="C0")
    ax1.plot(a_mid, [r["phase_err_rms"] for r in rows], "^--",
             label="RMS", color="C0", alpha=0.7)
    ax1.plot(a_mid, [r["sigma_phi_rms"] for r in rows], "s:",
             label="spread", color="C3", alpha=0.7)
    ax1.set_ylabel("phi_major (deg)")
    ax1.set_title("Absolute phase: error versus ensemble spread")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(a_mid, [r["dphi_err_mae"] for r in rows], "o-",
             label="MAE", color="C1")
    ax2.plot(a_mid, [r["dphi_err_rms"] for r in rows], "^--",
             label="RMS", color="C1", alpha=0.7)
    ax2.plot(a_mid, [r["sigma_dphi_rms"] for r in rows], "s:",
             label="spread", color="C3", alpha=0.7)
    ax2.set_ylabel("delta phi (deg)")
    ax2.set_title("Inter-axis offset: error versus ensemble spread")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3.plot(a_mid, [100.0 * r["dphi_within"] for r in rows], "^-", color="C2")
    ax3.axhline(100-tol, linestyle=":", color="grey",
                label=f"{tol:g} deg tolerance")
    ax3.set_xlabel("a (mm)")
    ax3.set_ylabel("within tolerance (%)")
    ax3.set_ylim(0, 102)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_error_vs_ratio(rows: list[dict], out_path: Path) -> None:
    """
    Error and both spreads against elongation, one line per stratum.

    Ratio controls the anisotropy and therefore the achievable offset, so a
    degradation at the extremes is physically expected. Whether the spread
    follows it is the question, and the two spreads are shown separately because
    the offset is what the CP condition depends on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = sorted({r["stratum"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (ax1, ax2), (ax3, ax4) = axes

    for i, name in enumerate(strata):
        sub = sorted([r for r in rows if r["stratum"] == name],
                     key=lambda r: r["ratio"])
        x = [r["ratio"] for r in sub]
        color = f"C{i}"

        ax1.plot(x, [r["phase_err_mae"] for r in sub], "o-",
                 color=color, label=name)
        ax2.plot(x, [r["sigma_phi_rms"] for r in sub], "o-",
                 color=color, label=name)
        ax3.plot(x, [r["dphi_err_mae"] for r in sub], "o-",
                 color=color, label=name)
        ax4.plot(x, [r["sigma_dphi_rms"] for r in sub], "o-",
                 color=color, label=name)

    for ax, title, ylabel in (
        (ax1, "phi_major error", "error (deg)"),
        (ax2, "phi_major spread", "sigma (deg)"),
        (ax3, "delta phi error", "error (deg)"),
        (ax4, "delta phi spread", "sigma (deg)"),
    ):
        ax.set_xlabel("ratio a/b")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_spread_vs_error(rows: list[dict], out_path: Path) -> None:
    """
    Scatter of ensemble spread against actual error, one point per group.

    The diagonal marks error = spread. Points below it mean the spread
    overstates the error, which is what a correlated ensemble produces: members
    trained on overlapping data disagree by more than the ensemble mean is
    wrong, so the spread is an upper bound rather than an estimate.

    Because the spread and the error can differ by an order of magnitude, the
    axes are logarithmic. On linear axes every point collapses against the left
    edge and the plot says nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, err_key, sig_key, title in (
        (ax1, "phase_err_rms", "sigma_phi_rms", "phi_major"),
        (ax2, "dphi_err_rms", "sigma_dphi_rms", "delta phi"),
    ):
        err = np.array([r[err_key] for r in rows])
        sig = np.array([r[sig_key] for r in rows])

        # Drop non-positive values, which cannot appear on a log axis.
        keep = (err > 0) & (sig > 0)
        err, sig = err[keep], sig[keep]

        ax.scatter(sig, err, s=34, alpha=0.75, zorder=3)
        for r, e, s in zip([x for x, k in zip(rows, keep) if k], err, sig):
            if "ratio" in r:
                ax.annotate(f"{r['ratio']:.1f}", (s, e),
                            fontsize=6, alpha=0.6,
                            xytext=(3, 3), textcoords="offset points")

        lo = min(err.min(), sig.min()) * 0.5
        hi = max(err.max(), sig.max()) * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="error = spread")
        # Calibrated expectation for a Gaussian: E|e| = 0.8 sigma.
        ax.plot([lo, hi], [0.8 * lo, 0.8 * hi], ":", color="grey", lw=1,
                label="calibrated (0.8 sigma)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("ensemble spread (deg)")
        ax.set_ylabel("actual error (deg)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")

        # The ratio of the two, summarised as one number per panel.
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.median(err / sig)
        ax.text(0.97, 0.05, f"median error/spread = {frac:.2f}",
                transform=ax.transAxes, fontsize=9, ha="right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_calibration(rows: list[dict], out_path: Path) -> None:
    """
    Observed coverage against nominal, per sigma bin.

    A calibrated ensemble sits on the diagonal. Points below it mean the model
    is overconfident: the real error is larger than the spread claims, so sigma
    cannot be used as a filter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = [r["inside_frac"] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="calibrated")
    ax.plot(obs, obs, "o", label="bins", alpha=0.0)   # keeps the legend tidy
    ax.scatter(obs, obs, s=40, color="C0", zorder=3)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("observed coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Ensemble calibration")
    ax.grid(True, alpha=0.3)

    # Annotate the mean deviation from the diagonal, which is the single number
    # worth quoting in the text.
    dev = sum(abs(o - 0.95) for o in obs) / max(len(obs), 1)
    ax.text(0.03, 0.95, f"mean |obs - 0.95| = {dev:.3f}",
            transform=ax.transAxes, fontsize=9, va="top")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_error_histogram(rows: list[dict], out_path: Path) -> None:
    """
    Per-stratum comparison of error and both spreads.

    Grouped bars are the clearest way to see which stratum the model struggles
    with, and whether its spread admits it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["stratum"] for r in rows]
    err = [r["phase_err_mae"] for r in rows]
    s_phi = [r["sigma_phi_rms"] for r in rows]
    s_dphi = [r["sigma_dphi_rms"] for r in rows]

    x = range(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - width for i in x], err, width, label="error", color="C0")
    ax.bar(list(x), s_phi, width, label="spread phi", color="C1")
    ax.bar([i + width for i in x], s_dphi, width, label="spread dphi",
           color="C2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("degrees")
    ax.set_title("Phase error and both ensemble spreads by stratum")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="directory holding the evaluation CSVs written by "
                             "evaluate.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for the figures; created if absent. "
                             "Default: the same as --dir")
    parser.add_argument("--tol", type=float, default=5.0,
                        help="offset tolerance for the reference line, degrees")
    args = parser.parse_args(argv)

    in_dir = args.dir.resolve()
    out_dir = args.out.resolve() if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    a_rows = read_csv(in_dir / "eval_by_a.csv")
    ratio_rows = read_csv(in_dir / "eval_by_ratio.csv")
    strata_rows = read_csv(in_dir / "eval_by_stratum.csv")
    calib_rows = read_csv(in_dir / "eval_calibration.csv")

    if not (a_rows or ratio_rows or strata_rows or calib_rows):
        print(f"no evaluation CSVs found in {in_dir}; run evaluate.py first")
        return 1

    print(f"reading  : {in_dir}")
    print(f"writing  : {out_dir}")

    if a_rows:
        plot_error_vs_a(a_rows, out_dir / "plot_error_vs_a.pdf", args.tol)
    if ratio_rows:
        plot_error_vs_ratio(ratio_rows, out_dir / "plot_error_vs_ratio.pdf")
    if strata_rows:
        plot_error_histogram(strata_rows, out_dir / "plot_error_by_stratum.pdf")
    if calib_rows:
        plot_calibration(calib_rows, out_dir / "plot_calibration.pdf")
    if ratio_rows:
        plot_spread_vs_error(ratio_rows,
                             out_dir / "plot_spread_vs_error.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
