# Data statistics

Tools for judging whether the simulated dataset is good enough to train a
surrogate on. Three scripts share one configuration module.

| File | Purpose |
|---|---|
| `config.py` | **every tunable constant**: data root, clearance, plot cosmetics, binning |
| `common.py` | CSV parsing, geometry rule, loading and grouping — no drawing |
| `stats.py` | one-page table of every configuration plus edge-gap analysis |
| `coverage_stats.py` | coverage statistics and per-combination histograms |

Reads **only** `result_navigator.csv` files — the `.s2p` data is never opened,
so all three run fast even over tens of thousands of runs.

Requires the standard library; the plotting scripts additionally need
matplotlib. If it is absent they still print their tables and skip the figures.

## Run

```bash
cd "feasibility study/simulations/surrogate/data statistics"

python stats.py "../../data"                    # table + one-page overview
python coverage_stats.py                        # statistics + histograms

python coverage_stats.py --no-titles            # for figures with a caption
python coverage_stats.py --format png --dpi 150
python coverage_stats.py --no-plots             # counts table only
python coverage_stats.py --root ../../data --out plots/
```

## `stats.py` — configuration table

One page showing what exists and how densely `a` was sampled:

```
P_ratio -> epsilon_r -> H_pillar -> ratio -> a
```

One row per `ratio`-level group. Columns: the four levels, `n`, `a_min`,
`a_max`, `gap`, `a_lim`, `from`, `to`, `n_new`, plus two bars (sample count,
edge coverage). A red **✗** marks a sweep stopping short of the cell edge,
and `from`/`to`/`n_new` say exactly what additional sweep would close the gap.

Parent values print only on the first row of their block, so nesting reads as
indentation-by-blankness. Background bands are tinted per `epsilon_r`; heavy
rules mark `P_ratio` boundaries.

### Outputs

- console: the table plus a summary block
- `coverage.csv`: one row per leaf, all rows regardless of `--top`
- `data_statistics.svg`: the same table, vector and text-selectable

## `coverage_stats.py` — dataset quality

Counts only samples passing `config.VALIDITY_CLEARANCE_MM`, so the numbers
describe geometry that could actually be built.

Per `(P_ratio, epsilon_r)` combination:

1. **count table** on the console — `n`, `a` range, number of distinct ratios
2. **`a` histogram** — one figure per combination, all ratios pooled
3. **ratio histogram** — one figure per combination, one bar per distinct value

Plus **`counts_summary`**: a single figure of sample counts across all
combinations, grouped by `epsilon_r`. This is the overview that shows where the
dataset is thin.

Figures are written as separate files so each can be placed independently.

## Design notes

**Values are compared with a tolerance.** `2.2` and `2.2000001` arrive from
different export paths; grouping them on exact equality would split a group into
two rows and overstate the leaf count.

**Column lookup is by name, not position.** CST's navigator column order varies
between export scripts, and column names are matched case-insensitively.

**`stats.py` filters nothing.** It reports the raw state of the tree, including
incomplete groups. If a `ratio` value was swept at only three `a` points, that
is exactly what you want to see there. `coverage_stats.py` is the script that
applies the clearance filter.

## Geometry rule

The ellipse major axis is rotated 45°, so its axis-aligned bounding box
half-extent is `sqrt((a² + b²)/2)` with `b = a/ratio`. Fitting inside the cell
with a clearance gives:

```
a_max = (P/2 - clearance) * sqrt(2) / sqrt(1 + 1/ratio²)
```

A larger `ratio` permits a longer major axis, which is why a sweep sharing one
upper limit across several ratios never reaches the edge for the elongated ones.

`VALIDITY_CLEARANCE_MM = 0.01` is the gap allowed between the ellipse bounding
box and the cell wall. It is **not** the manufacturability clearance: 0.2 mm in
the MATLAB explorer reflects what an FDM nozzle can print. For deciding which
simulated points are physically valid, a small margin suffices.

## Plot cosmetics

All in `config.py`:

| Constant | Default |
|---|---|
| `PLOT_FIGSIZE` | `(6.5, 4.2)` in |
| `PLOT_DPI` | 300 |
| `PLOT_FORMAT` | `pdf` (vector) |
| `PLOT_AXIS_LABEL_SIZE` | 13 |
| `PLOT_TICK_LABEL_SIZE` | 12 |
| `PLOT_TITLE_SIZE` | 13 |
| `PLOT_LEGEND_SIZE` | 11 |
| `PLOT_GRID` | `True`, alpha 0.30, dotted |
| `PLOT_SHOW_TITLE` | `True` — toggle per run with `--no-titles` |
| `A_HIST_BINS` | 40 |

`--format` and `--dpi` override the config for a single run only.

PDF text is kept as text (`pdf.fonttype = 42`), so labels stay selectable and
copyable from the finished document.
