"""
Coverage statistics for the simulated dataset.

Answers three questions about how usable the data is for surrogate training:

  1. how many valid samples exist for every (P_ratio, epsilon_r) combination
  2. for each combination, how the `a` values are distributed (all ratios pooled)
  3. for each combination, how the `ratio` values are distributed

Only samples passing the validity clearance in config.VALIDITY_CLEARANCE_MM are
counted, so the numbers describe geometry that could actually be built.

Usage
-----
    python coverage_stats.py                       # default data root
    python coverage_stats.py --root <dir>
    python coverage_stats.py --no-titles           # figures without a title
    python coverage_stats.py --out plots/
    python coverage_stats.py --format png --dpi 150

Each (P_ratio, epsilon_r) combination produces up to two figures, written as
separate files so they can be placed independently in a document.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow `python coverage_stats.py` from inside this directory as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import common


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(
    samples: list[common.Sample],
) -> dict[tuple[float, float], dict]:
    """
    Group valid samples by (P_ratio, epsilon_r).

    Returns, per combination, the list of `a` values and the list of `ratio`
    values, with the ratios kept separate from `a` so that each combination can
    be histogrammed on either axis independently.
    """
    agg: dict[tuple[float, float], dict] = defaultdict(
        lambda: {"a": [], "ratio": [], "n": 0}
    )

    for s in samples:
        if not s.is_valid:
            continue
        key = (
            round(s.p_ratio / config.VALUE_TOL) * config.VALUE_TOL,
            round(s.epsilon_r / config.VALUE_TOL) * config.VALUE_TOL,
        )
        agg[key]["a"].append(s.a)
        agg[key]["ratio"].append(s.ratio)
        agg[key]["n"] += 1

    return dict(sorted(agg.items(), key=lambda kv: kv[0]))


def count_table(agg: dict[tuple[float, float], dict],
                total_samples: int) -> str:
    """Console table of sample counts per (P_ratio, epsilon_r)."""
    if not agg:
        return "no valid samples"

    lines: list[str] = []
    valid = sum(v["n"] for v in agg.values())

    lines.append(f"samples read              : {total_samples}")
    lines.append(f"samples passing clearance : {valid}"
                 f"  ({100.0 * valid / total_samples:.1f}%)")
    lines.append(f"validity clearance        : "
                 f"{config.VALIDITY_CLEARANCE_MM} mm")
    lines.append("")
    lines.append(f"{'P_ratio':>8} {'epsilon_r':>10} {'n':>6} "
                 f"{'a_min':>8} {'a_max':>8} {'n_ratio':>8}")
    lines.append("-" * 54)

    for (p_ratio, eps), v in agg.items():
        lines.append(
            f"{common.fmt(p_ratio):>8} {common.fmt(eps):>10} {v['n']:>6} "
            f"{min(v['a']):>8.3f} {max(v['a']):>8.3f} "
            f"{len(set(v['ratio'])):>8}"
        )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _setup_matplotlib():
    """Import matplotlib and apply the shared style, or return None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    matplotlib.rcParams.update({
        "axes.labelsize": config.PLOT_AXIS_LABEL_SIZE,
        "xtick.labelsize": config.PLOT_TICK_LABEL_SIZE,
        "ytick.labelsize": config.PLOT_TICK_LABEL_SIZE,
        "axes.titlesize": config.PLOT_TITLE_SIZE,
        "legend.fontsize": config.PLOT_LEGEND_SIZE,
        "figure.dpi": config.PLOT_DPI,
        "savefig.dpi": config.PLOT_DPI,
        # keep glyphs as text in vector output so labels stay selectable
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    return plt


def _decorate(ax, xlabel: str, ylabel: str, title: str, show_title: bool,
              plt) -> None:
    """Apply the shared axis cosmetics to one panel."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title)
    if config.PLOT_GRID:
        ax.grid(True, alpha=config.PLOT_GRID_ALPHA,
                linestyle=config.PLOT_GRID_LINESTYLE)


def plot_a_histograms(
    agg: dict[tuple[float, float], dict],
    out_dir: Path,
    show_title: bool,
) -> list[Path]:
    """
    One figure per (P_ratio, epsilon_r): histogram of sampled `a`, all ratios
    pooled.

    Pooling is deliberate: the question is how densely the tuning variable is
    covered for a given cell and permittivity, regardless of which elongation
    produced each point.
    """
    plt = _setup_matplotlib()
    if plt is None:
        return []

    written: list[Path] = []

    for (p_ratio, eps), v in agg.items():
        a_values = v["a"]
        if not a_values:
            continue

        fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
        ax.hist(
            a_values,
            bins=config.A_HIST_BINS,
            color=config.PLOT_BAR_COLORS[0],
            alpha=config.PLOT_BAR_ALPHA,
            edgecolor=config.PLOT_BAR_EDGE,
            linewidth=config.PLOT_BAR_EDGE_WIDTH,
        )
        _decorate(
            ax,
            "a (mm)",
            "samples",
            f"a coverage:  P_ratio={common.fmt(p_ratio)}, "
            f"epsilon_r={common.fmt(eps)}  (n={v['n']})",
            show_title, plt,
        )
        fig.tight_layout()

        path = out_dir / f"a_hist_P{common.fmt(p_ratio)}_er{common.fmt(eps)}" \
                         f".{config.PLOT_FORMAT}"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written


def plot_ratio_histograms(
    agg: dict[tuple[float, float], dict],
    out_dir: Path,
    show_title: bool,
) -> list[Path]:
    """
    One figure per (P_ratio, epsilon_r): histogram of sampled `ratio`.

    `ratio` takes few distinct values, so the bars are placed on the discrete
    values themselves rather than binned continuously. That makes the count at
    each elongation readable directly, which is what tells you whether a
    combination was swept across the full anisotropy range or only at one end.
    """
    plt = _setup_matplotlib()
    if plt is None:
        return []

    written: list[Path] = []

    for (p_ratio, eps), v in agg.items():
        ratios = v["ratio"]
        if not ratios:
            continue

        counts = Counter(round(r / config.VALUE_TOL) * config.VALUE_TOL
                         for r in ratios)
        xs = sorted(counts)
        ys = [counts[x] for x in xs]

        fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
        ax.bar(
            [str(common.fmt(x)) for x in xs],
            ys,
            color=config.PLOT_BAR_COLORS[1],
            alpha=config.PLOT_BAR_ALPHA,
            edgecolor=config.PLOT_BAR_EDGE,
            linewidth=config.PLOT_BAR_EDGE_WIDTH,
        )
        _decorate(
            ax,
            "ratio (a/b)",
            "samples",
            f"ratio coverage:  P_ratio={common.fmt(p_ratio)}, "
            f"epsilon_r={common.fmt(eps)}  (n={v['n']})",
            show_title, plt,
        )
        fig.tight_layout()

        path = out_dir / f"ratio_hist_P{common.fmt(p_ratio)}_er{common.fmt(eps)}" \
                         f".{config.PLOT_FORMAT}"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written


def plot_counts_summary(
    agg: dict[tuple[float, float], dict],
    out_dir: Path,
    show_title: bool,
) -> list[Path]:
    """
    One figure summarising sample counts across all combinations.

    Bars are grouped by epsilon_r so the effect of permittivity on how much data
    exists is visible directly, which is the single most useful overview when
    deciding where to add simulations.
    """
    plt = _setup_matplotlib()
    if plt is None or not agg:
        return []

    eps_values = sorted({k[1] for k in agg})
    p_values = sorted({k[0] for k in agg})

    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)

    n_series = len(eps_values)
    bar_w = 0.8 / max(n_series, 1)

    for i, eps in enumerate(eps_values):
        heights = [agg.get((p, eps), {"n": 0})["n"] for p in p_values]
        offsets = [j + (i - (n_series - 1) / 2) * bar_w
                   for j in range(len(p_values))]
        ax.bar(
            offsets, heights, width=bar_w,
            color=config.PLOT_BAR_COLORS[i % len(config.PLOT_BAR_COLORS)],
            alpha=config.PLOT_BAR_ALPHA,
            edgecolor=config.PLOT_BAR_EDGE,
            linewidth=config.PLOT_BAR_EDGE_WIDTH,
            label=f"epsilon_r = {common.fmt(eps)}",
        )

    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([common.fmt(p) for p in p_values])
    _decorate(
        ax,
        "P_ratio  (cell size / lambda_0)",
        "valid samples",
        "Valid samples per (P_ratio, epsilon_r)",
        show_title, plt,
    )
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    path = out_dir / f"counts_summary.{config.PLOT_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help=f"data root (default: {config.DEFAULT_DATA_ROOT})")
    parser.add_argument("--out", type=Path, default=Path("coverage_plots"),
                        help="directory for the figures")
    parser.add_argument("--no-titles", action="store_true",
                        help="omit plot titles, for figures that get a caption")
    parser.add_argument("--format", default=config.PLOT_FORMAT,
                        choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=config.PLOT_DPI)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="name every navigator CSV visited")
    parser.add_argument("--no-plots", action="store_true",
                        help="print the count table only")
    parser.add_argument("--table-out", type=Path, default=None,
                        help="also write the count table to this file, so it "
                             "can be included in a document verbatim")
    args = parser.parse_args(argv)

    # Command-line cosmetics override the config defaults for this run only
    config.PLOT_FORMAT = args.format
    config.PLOT_DPI = args.dpi
    show_title = not args.no_titles and config.PLOT_SHOW_TITLE

    root = common.resolve_root(args.root)
    if not root.is_dir():
        print(f"error: data root not found: {root}")
        return 1

    samples, warnings = common.collect_samples(root, verbose=args.verbose)
    for message in warnings:
        print(f"warning: {message}")
    if not samples:
        print("error: no samples loaded")
        return 1

    agg = aggregate(samples)

    table = count_table(agg, len(samples))
    print()
    print(table)
    print()

    if args.table_out is not None:
        table_path = (args.table_out if args.table_out.is_absolute()
                      else Path.cwd() / args.table_out)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {table_path.resolve()}")

    if args.no_plots:
        return 0

    out_dir = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written += plot_a_histograms(agg, out_dir, show_title)
    written += plot_ratio_histograms(agg, out_dir, show_title)
    written += plot_counts_summary(agg, out_dir, show_title)

    if not written:
        print("note: matplotlib unavailable, no figures written")
        return 0

    print(f"wrote {len(written)} figures to {out_dir}")
    for path in written:
        print(f"  {path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
