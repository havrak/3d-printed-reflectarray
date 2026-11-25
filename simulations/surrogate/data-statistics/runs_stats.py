"""
Overview of the CST sweep coverage in a directory tree.

Scans for `result_navigator.csv` files only, reads the swept parameters from
each, and renders one page showing what configurations exist and how densely
the tuning variable was sampled.

Hierarchy, outermost first:

    P_ratio -> epsilon_r -> H_pillar -> ratio -> a

Usage
-----
    python stats.py                          # default data root, outputs to cwd
    python stats.py <directory>              # positional root
    python stats.py --root <directory>       # same, named form
    python stats.py --root <dir> --out plots/
    python stats.py <dir> --csv out.csv --png out.svg --top 60
    python stats.py <dir> --no-png           # table only

Outputs
-------
    console   the tree table and the summary block
    CSV       one row per leaf, for grepping
    image     the same table plus a summary panel, one page

Only the navigator CSV is read; the .s2p files are not touched, so this is
fast even on a large tree.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python stats.py` from inside this directory as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import common

# Loading, CSV parsing and the geometry rule now live in common.py so that
# coverage_stats.py reads the tree identically. These aliases keep the rest of
# this file unchanged.
NAVIGATOR_NAME = config.NAVIGATOR_NAME
LEVELS = config.LEVELS
TUNING = config.TUNING
VALUE_TOL = config.VALUE_TOL
SUMMARY_WRAP_COLS = config.SUMMARY_WRAP_COLS
LEV_SEP = config.LEV_SEP

LAMBDA0_MM = config.LAMBDA0_MM
GEOMETRIC_CLEARANCE_MM = 0.0
EDGE_MARGIN_MM = config.EDGE_MARGIN_MM
TARGET_STEP_MM = config.TARGET_STEP_MM

max_major_axis = common.max_major_axis
extend_sweep = common.extend_sweep
Leaf = common.Leaf
read_navigator = common.read_navigator
collect_leaves = common.collect_leaves
_fmt = common.fmt


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #

def render_tree(leaves: list[Leaf], top: int | None) -> str:
    """
    Indented tree table.

    A parent value is printed only on the first row of its block, so the
    grouping is visible as blank space and the eye can follow one branch.
    """
    headers = LEVELS + ["n", "a_min", "a_max", "a_lim",
                        "new_from", "new_to", "n_new", "dir"]
    widths = [max(len(h), 7) for h in headers]

    out: list[str] = []
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out.append(header_line)
    out.append("-" * len(header_line))

    shown = leaves if top is None else leaves[:top]
    prev: tuple[float, ...] | None = None

    for leaf in shown:
        cells: list[str] = []
        for idx, value in enumerate(leaf.levels):
            if prev is not None and all(
                abs(leaf.levels[k] - prev[k]) < VALUE_TOL for k in range(idx + 1)
            ):
                cells.append("")          # unchanged prefix: leave blank
            else:
                cells.append(_fmt(value))

        limit = leaf.a_limit
        ext = leaf.extension
        if ext is None:
            ext_from = ext_to = ext_n = "-"
        else:
            start, end, n_new = ext
            ext_from = f"{start:.2f}"
            ext_to = f"{end:.2f}"
            ext_n = str(n_new)

        cells += [
            str(leaf.n_samples),
            _fmt(leaf.a_min),
            _fmt(leaf.a_max),
            f"{limit:.2f}" if limit == limit else "nan",
            ext_from,
            ext_to,
            ext_n,
            leaf.source,
        ]
        out.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        prev = leaf.levels

    if top is not None and len(leaves) > top:
        out.append(f"... {len(leaves) - top} more rows (use --csv for all)")

    return "\n".join(out)


def _wrap_items(prefix: str, items: list[str], indent: str,
                width: int = SUMMARY_WRAP_COLS) -> list[str]:
    """
    Join comma-separated items onto as many lines as needed.

    The first line starts with `prefix`, continuation lines with `indent`, so a
    long list stays aligned and nothing is dropped. Wrapping is preferred over
    an ellipsis because the omitted values are exactly what the reader needs in
    order to judge whether the dataset covers the parameter space.
    """
    if not items:
        return [prefix.rstrip()]

    lines: list[str] = []
    current = prefix
    current_has_item = False

    for item in items:
        addition = item if not current_has_item else ", " + item
        if current_has_item and len(current) + len(addition) > width:
            lines.append(current)
            current = indent + item
        else:
            current += addition
        current_has_item = True

    lines.append(current)
    return lines


def render_summary(leaves: list[Leaf]) -> str:
    """Totals and per-level coverage, wrapped rather than truncated."""
    if not leaves:
        return "no data"

    out: list[str] = []
    total = sum(lf.n_samples for lf in leaves)

    out.append(f"leaves (ratio-level groups) : {len(leaves)}")
    out.append(f"total samples              : {total}")
    out.append(f"a range overall            : "
               f"{_fmt(min(lf.a_min for lf in leaves))} .. "
               f"{_fmt(max(lf.a_max for lf in leaves))}")
    out.append("")
    out.append("distinct values per level:")

    for idx, level in enumerate(LEVELS):
        values = sorted({lf.levels[idx] for lf in leaves})
        items = [_fmt(v) for v in values]
        prefix = f"  {level:<10} {len(values):>3} : "
        indent = " " * len(prefix)
        out.extend(_wrap_items(prefix, items, indent))

    # a-sampling density, which is what matters for surrogate quality
    counts = [lf.n_samples for lf in leaves]
    out.append("")
    out.append("a samples per leaf:")
    out.append(f"  min {min(counts)}   median {sorted(counts)[len(counts)//2]}"
               f"   max {max(counts)}")

    # ---- geometric edge coverage --------------------------------------
    # A sweep that stops short of the cell boundary leaves the most extreme
    # geometry unsampled, which is exactly the region the surrogate must
    # interpolate at the edge of its domain.
    needs = [lf for lf in leaves if lf.needs_extension]
    extra = sum(lf.extension[2] for lf in needs)

    out.append("")
    out.append("geometric edge coverage:")
    out.append(f"  sweep ends short of the edge : {len(needs)} of {len(leaves)}")
    out.append(f"  additional runs required     : {extra}")
    if needs:
        # widest gaps first, since those distort the surrogate most; every
        # group is listed so nothing is hidden behind a count
        worst = sorted(
            needs,
            key=lambda lf: (lf.a_limit - lf.a_max) / max(lf.a_limit, 1e-9),
            reverse=True,
        )
        
        # out.append("  largest relative gaps:")
        # for lf in worst:
        #    gap = lf.a_limit - lf.a_max
        #    pct = 100.0 * gap / lf.a_limit if lf.a_limit else float("nan")
        #    # one parameter per line keeps each row short enough to fit the
        #    # panel without wrapping mid-value
        #    out.append(f"    {LEV_SEP}")
        #    for name, value in zip(LEVELS, lf.levels):
        #        out.append(f"      {name:<10} = {_fmt(value)}")
        #    out.append(f"      a_max      = {lf.a_max:.2f}")
        #    out.append(f"      a_lim      = {lf.a_limit:.2f}"
        #               f"   ({pct:.0f}% short)")
        #    out.append(f"      {LEV_SEP}")
        

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# PNG rendering
# --------------------------------------------------------------------------- #

def render_image(leaves: list[Leaf], summary: str, out_path: Path) -> bool:
    """
    Draw the tree table and the summary on one page, all rows, at a fixed
    physical row height.

    The row height is constant in inches rather than a fraction of the figure,
    so the text stays the same size no matter how many rows there are; the page
    simply grows. Combined with a vector format this stays readable and
    zoomable at any size.

    Returns False when matplotlib is unavailable, rather than failing the run:
    the console output is the primary product.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        # Keep text as real <text> elements in SVG output so it stays
        # selectable and copyable. The default ('path') converts glyphs to
        # outlines, which looks identical but cannot be copied.
        matplotlib.rcParams["svg.fonttype"] = "none"
    except ImportError:
        return False

    # ---- layout, in inches -------------------------------------------
    row_in = 0.155           # physical height of one table row
    font_pt = 6.5            # row font size, constant per row
    head_pt = 7.0
    # Left margin only needs room for the branch tick, which sits 0.008 axes
    # units left of x = 0, plus a little breathing space.
    left_margin = 0.30
    top_margin = 0.45
    bottom_margin = 0.25
    # The summary is wrapped by render_summary to SUMMARY_WRAP_COLS, so the
    # panel is sized for that width rather than for the longest raw line. That
    # keeps the figure bounded no matter how many values a level has.
    summary_font_pt = 6.5
    summary_chars = min(
        max((len(ln) for ln in summary.splitlines()), default=40),
        SUMMARY_WRAP_COLS,
    )
    summary_char_in = 0.60 * (summary_font_pt / 72.0)
    summary_width = summary_chars * summary_char_in + 0.12
    table_width = 7.4        # inches for the table columns (12 columns)
    right_margin = 0.12

    n_rows = len(leaves)
    fig_w = left_margin + table_width + 0.25 + summary_width + right_margin
    # one row for the header, one per data row
    fig_h = top_margin + bottom_margin + row_in * (n_rows + 2)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Axes sized in inches, converted to figure fractions
    ax_table = fig.add_axes([
        left_margin / fig_w,
        bottom_margin / fig_h,
        table_width / fig_w,
        (fig_h - top_margin - bottom_margin) / fig_h,
    ])
    ax_summary = fig.add_axes([
        (left_margin + table_width + 0.25) / fig_w,
        bottom_margin / fig_h,
        summary_width / fig_w,
        (fig_h - top_margin - bottom_margin) / fig_h,
    ])
    for ax in (ax_table, ax_summary):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        # y is measured in rows, counting down from the top. The header
        # occupies its own row at y = 0 and data starts at y = 1, so the
        # first row cannot collide with the header text.
        ax.set_ylim(n_rows + 2, 0)

    # ---- colours -----------------------------------------------------
    # One colour per branch level. Deeper levels get progressively more
    # saturated colours, so the eye can follow a branch by colour alone
    # without reading the numbers.
    level_colors = ["#1f4e79", "#2e7d32", "#b8860b", "#8b2c8b"]
    bar_color = "#4a90b8"

    # Very pale tints for the epsilon_r bands. Each permittivity gets its own
    # hue rather than alternating grey/white, so the background identifies
    # *which* permittivity you are in, not merely that the group changed.
    # Saturation is kept low so the coloured foreground text stays legible.
    eps_fills = [
        "#eef3f9",   # pale blue
        "#fdf6e3",   # pale amber
        "#eef7ee",   # pale green
        "#f7eef7",   # pale violet
        "#fdf0f0",   # pale rose
        "#f0f7f7",   # pale teal
    ]

    # ---- header ------------------------------------------------------
    # Column positions are derived from the character width of each column's
    # widest content rather than hand-tuned, so a column can never be given
    # less room than its text needs.
    #
    # Layout (12 columns):
    #   P_ratio | epsilon_r | H_pillar | ratio | n | a_min | a_max | gap
    #   | a_lim | from | to | n_new         then: samples bar, edge bar
    headers = LEVELS + [
        "n", "a_min", "a_max", "gap", "a_lim", "from", "to", "n_new",
    ]

    # Widest text each column must hold, in characters. Decimals are formatted
    # to three places, so a value like 3.163 is 5 characters.
    widths_chars = [
        8,    # P_ratio   (header)
        10,    # epsilon_r (header, the longest name)
        9,    # H_pillar  (header)
        7,    # ratio     (header)
        6,    # n         (includes thousands, e.g. 1250)
        6,    # a_min     0.400
        6,    # a_max     3.163
        4,    # gap       header only; the cell holds a marker
        6,    # a_lim     3.163
        6,    # from      2.510
        6,    # to        3.150
        6,    # n_new     128
    ]
    assert len(widths_chars) == len(headers)

    # A monospace glyph at 6.5 pt occupies roughly this fraction of the axes
    # width. Measured: 6.5 pt ~ 0.090 in per em, monospace advance ~0.60 em.
    char_frac = 0.60 * (6.5 / 72.0) / table_width

    pad_chars = 1.0     # blank space between columns, in characters

    col_x: list[float] = []
    x_cursor = 0.0
    for w in widths_chars:
        col_x.append(x_cursor)
        x_cursor += (w + pad_chars) * char_frac

    # Reserve room on the right for the two bars before deciding the layout
    # fits, so the last columns cannot collide with them.
    samples_w = 0.075
    bars_need = 2 * samples_w + 0.020

    if x_cursor + bars_need > 0.99:
        raise ValueError(
            f"columns need {x_cursor:.2f} plus {bars_need:.2f} for the bars; "
            f"widen table_width or drop a column"
        )

    # x where the red cross is drawn, centred in the gap column
    gap_x = col_x[headers.index("gap")] + 0.5 * char_frac

    samples_x0 = 0.99 - bars_need          # right-align the bar pair
    cover_x0 = samples_x0 + samples_w + 0.020
    cover_w = samples_w

    for x, name in zip(col_x, headers):
        ax_table.text(x, 0, name, fontsize=head_pt, fontweight="bold",
                      family="monospace", va="center", zorder=6)
    ax_table.text(samples_x0, 0, "samples", fontsize=head_pt,
                  fontweight="bold", family="monospace", va="center", zorder=6)
    ax_table.text(cover_x0, 0, "edge", fontsize=head_pt,
                  fontweight="bold", family="monospace", va="center", zorder=6)

    # header rule, drawn under the header text
    ax_table.plot([0, 1], [0.5, 0.5], color="0.3", linewidth=0.8, zorder=6)

    n_max = max((lf.n_samples for lf in leaves), default=1)

    # ---- background bands keyed on epsilon_r -------------------------
    # epsilon_r is the parameter with the most distinct values, and it is the
    # one that most affects whether a configuration is usable at all (it sets
    # the phase range and the resonator Q). Tinting by epsilon_r therefore
    # separates the groups a reader actually wants to distinguish; P_ratio has
    # too few categories to be a useful key.
    #
    # A band is a maximal run of consecutive rows sharing one epsilon_r, so
    # the tint reads as "this stretch of the table is one permittivity" even
    # when the rows span several P_ratio blocks.
    eps_idx = LEVELS.index("epsilon_r")
    block_index: list[int] = []
    block = -1
    for r, leaf in enumerate(leaves):
        prev_eps = leaves[r - 1].levels[eps_idx] if r else None
        if r == 0 or abs(leaf.levels[eps_idx] - prev_eps) >= VALUE_TOL:
            block += 1
        block_index.append(block)

    # ---- rows --------------------------------------------------------
    prev: tuple[float, ...] | None = None
    prev_p_ratio: float | None = None

    for r, leaf in enumerate(leaves):
        y = r + 1                      # data starts one row below the header
        blk = block_index[r]
        starts_eps_band = r == 0 or blk != block_index[r - 1]

        # background tint for the whole epsilon_r band
        ax_table.add_patch(Rectangle(
            (0, y - 0.5), 1.0, 1.0,
            facecolor=eps_fills[blk % len(eps_fills)],
            edgecolor="none", zorder=0,
        ))

        # a heavier rule where P_ratio changes, since that is now a separate
        # grouping from the background tint
        new_p_ratio = (prev_p_ratio is None
                       or abs(leaf.levels[0] - prev_p_ratio) >= VALUE_TOL)
        if new_p_ratio and r > 0:
            ax_table.plot([0, 1], [y - 0.5, y - 0.5],
                          color="0.30", linewidth=1.0, zorder=1)

        # ---- cells ----
        for idx, value in enumerate(leaf.levels):
            # a value is printed only when it differs from the row above,
            # which is what makes the nesting readable
            unchanged = prev is not None and all(
                abs(leaf.levels[k] - prev[k]) < VALUE_TOL for k in range(idx + 1)
            )
            if unchanged:
                continue
            ax_table.text(
                col_x[idx], y, _fmt(value),
                fontsize=font_pt, family="monospace", va="center", zorder=3,
                color=level_colors[idx % len(level_colors)],
                fontweight="bold" if idx == 0 else "normal",
            )
            # a short tick marking a new branch at this level
            ax_table.plot(
                [col_x[idx] - 0.008, col_x[idx] - 0.003],
                [y, y], color=level_colors[idx % len(level_colors)],
                linewidth=1.2, zorder=3,
            )

        # ---- numeric columns ----
        # col_x[4:] has eight entries, matching the eight cells from n onwards.
        # The gap column is a placeholder: its marker is drawn separately.
        limit = leaf.a_limit
        coverage = leaf.a_coverage
        ext = leaf.extension
        if ext is None:
            ext_from = ext_to = ext_n = "-"
        else:
            ext_from = f"{ext[0]:.3f}"
            ext_to = f"{ext[1]:.3f}"
            ext_n = str(ext[2])

        numeric = [
            str(leaf.n_samples),                  # n
            f"{leaf.a_min:.3f}",                  # a_min
            f"{leaf.a_max:.3f}",                  # a_max
            "",                                   # gap: marker, not text
            f"{limit:.3f}" if limit == limit else "nan",   # a_lim
            ext_from,                             # from
            ext_to,                               # to
            ext_n,                                # n_new
        ]
        assert len(col_x[4:]) == len(numeric), (
            f"numeric cells ({len(numeric)}) do not match columns "
            f"({len(col_x[4:])})"
        )
        for x, cell in zip(col_x[4:], numeric):
            if not cell:
                continue
            ax_table.text(x, y, cell, fontsize=font_pt,
                          family="monospace", va="center", zorder=3)

        # ---- sample-count bar ----
        # Relative to the densest leaf, so a sparsely sampled group shows as a
        # short bar. Drawn behind the text.
        s_w = samples_w * (leaf.n_samples / n_max) if n_max else 0.0
        ax_table.add_patch(Rectangle(
            (samples_x0, y - 0.28), samples_w, 0.56,
            facecolor="white", edgecolor="0.80", linewidth=0.4, zorder=1,
        ))
        if s_w > 0:
            ax_table.add_patch(Rectangle(
                (samples_x0, y - 0.28), max(s_w, 0.0015), 0.56,
                facecolor=bar_color, edgecolor="none", zorder=1.5,
            ))

        # ---- edge-coverage bar ----
        # Fraction of the usable a interval that was actually sampled; the
        # unfilled remainder is what the surrogate must extrapolate over.
        c_w = cover_w * (coverage if coverage == coverage else 0.0)
        c_w = max(0.0, min(c_w, cover_w))
        ax_table.add_patch(Rectangle(
            (cover_x0, y - 0.28), cover_w, 0.56,
            facecolor="white", edgecolor="0.80", linewidth=0.4, zorder=1,
        ))
        if c_w > 0:
            ax_table.add_patch(Rectangle(
                (cover_x0, y - 0.28), c_w, 0.56,
                facecolor="#e08a3c", edgecolor="none", zorder=1.5,
            ))

        # ---- red cross marking a sweep that stops short of the edge ----
        if leaf.needs_extension:
            ax_table.plot(gap_x, y, marker="X", markersize=5.0,
                          color="#c62828", markeredgewidth=0.0, zorder=6)

        prev = leaf.levels
        prev_p_ratio = leaf.levels[0]

    # ---- summary panel ----------------------------------------------
    # The panel holds free-form text whose lines can be long. It is drawn with
    # clip_on=False so nothing is cut off at the axes edge, and its own y-limit
    # is set from the line count so the text is not squeezed vertically either.
    n_summary_lines = max(1, summary.count("\n") + 1)
    ax_summary.set_ylim(n_summary_lines + 1, 0)
    ax_summary.text(
        0.0, 0, summary,
        fontsize=summary_font_pt, family="monospace",
        va="top", ha="left", linespacing=1.45, clip_on=False,
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return True


def write_csv(leaves: list[Leaf], out_csv: Path) -> None:
    """
    One row per leaf, including the geometry limit and the extra sweep needed.

    `extend_from` / `extend_to` / `extend_n` are empty when the sweep already
    reaches the cell edge.
    """
    header = LEVELS + [
        "n_samples", "a_min", "a_max", "a_span",
        "a_limit", "coverage", "extend_from", "extend_to", "extend_n",
        "source_dir",
    ]
    lines = ["\t".join(header)]

    for leaf in leaves:
        limit = leaf.a_limit
        coverage = leaf.a_coverage
        ext = leaf.extension

        cells = [_fmt(v) for v in leaf.levels] + [
            str(leaf.n_samples),
            _fmt(leaf.a_min),
            _fmt(leaf.a_max),
            f"{leaf.a_span:.3f}",
            f"{limit:.4f}" if limit == limit else "",
            f"{coverage:.4f}" if coverage == coverage else "",
            f"{ext[0]:.3f}" if ext else "",
            f"{ext[1]:.3f}" if ext else "",
            str(ext[2]) if ext else "",
            leaf.source,
        ]
        lines.append("\t".join(cells))

    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=None,
                        help="root to scan recursively "
                             f"(default: {config.DEFAULT_DATA_ROOT})")
    parser.add_argument("--root", type=Path, default=None,
                        help="data root; same as the positional argument, "
                             "provided for consistency with coverage_stats.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for coverage.csv and the image "
                             "(default: the current directory)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the full table here (overrides --out)")
    parser.add_argument("--png", type=Path, default=None,
                        help="write the overview image here (overrides --out)")
    parser.add_argument("--no-png", action="store_true",
                        help="skip the image even if --png is not given")
    parser.add_argument("--top", type=int, default=0,
                        help="rows to show on the console (default 0 = all); "
                             "the image always contains every row")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="name every navigator CSV visited, with its "
                             "header and whether it was usable")
    parser.add_argument("--format", default="svg",
                        choices=["svg", "pdf", "png"],
                        help="image format (default svg, vector and zoomable)")
    args = parser.parse_args(argv)

    # The root can arrive positionally or via --root. Giving both is a mistake
    # worth reporting rather than silently preferring one.
    if args.root is not None and args.directory is not None:
        print("error: give the data root either positionally or with --root, "
              "not both")
        return 1
    root = common.resolve_root(args.root or args.directory)

    if not root.is_dir():
        print(f"error: data root not found: {root}")
        return 1

    leaves, warnings = collect_leaves(root, verbose=args.verbose)
    for message in warnings:
        print(f"warning: {message}")
    if not leaves:
        print("error: nothing to report")
        return 1

    top = None if args.top == 0 else args.top

    print()
    print(render_tree(leaves, top))
    print()
    print("=" * 60)
    print(render_summary(leaves))
    print("=" * 60)

    # Outputs default to the directory the script was run from, not the data
    # directory, which is where the source data lives and should stay clean.
    # --out redirects both, while --csv and --png override individually.
    out_dir = (args.out if args.out is not None else Path.cwd())
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    csv_path = args.csv if args.csv is not None else out_dir / "coverage.csv"
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(leaves, csv_path)
    print(f"\nwrote {csv_path.resolve()}  ({len(leaves)} rows)")

    if not args.no_png:
        img_path = (args.png if args.png is not None
                    else out_dir / f"data_statistics.{args.format}")
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        if render_image(leaves, render_summary(leaves), img_path):
            print(f"wrote {img_path.resolve()}  (all {len(leaves)} rows)")
        else:
            print("note: matplotlib not available, no image written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
