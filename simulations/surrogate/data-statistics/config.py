"""
Shared configuration for every script in the data-statistics folder.

All tunable constants live here: the sweep location, the plot cosmetics, the
geometry clearance and the sampling parameters. Each script imports from this
module rather than defining its own, so a change of style or of the clearance
rule affects every output consistently.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Data location
# --------------------------------------------------------------------------- #

# Root scanned recursively for result_navigator.csv files. Relative to the
# repository root; scripts resolve it against the repo, not the cwd.
DEFAULT_DATA_ROOT = Path("feasibility study/simulations/data")

NAVIGATOR_NAME = "result_navigator.csv"

# Nesting order of the swept parameters, outermost first. `a` is the tuning
# variable and is treated separately wherever it matters.
LEVELS = ["P_ratio", "epsilon_r", "H_pillar", "ratio"]
TUNING = "a"

# Values closer than this are treated as equal, so that 2.2 and 2.2000001 from
# different export paths do not split a group.
VALUE_TOL = 1e-6


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
#
# The ellipse major axis is rotated 45 degrees, so its axis-aligned bounding box
# half-extent is sqrt((a^2 + b^2)/2) with b = a/ratio. Requiring that to stay
# inside the cell gives the largest usable semi-major axis:
#
#     sqrt((a^2 + (a/ratio)^2)/2) <= P/2 - clearance
#     a <= (P/2 - clearance) * sqrt(2) / sqrt(1 + 1/ratio^2)
#
# A larger ratio therefore permits a longer major axis, which is why a sweep
# that shares one upper limit across several ratios never reaches the edge for
# the more elongated ones.

C0_MM_GHZ = 299.792458           # mm * GHz
F0_GHZ = 30.0                    # design frequency
LAMBDA0_MM = C0_MM_GHZ / F0_GHZ

# Allowed gap between the ellipse bounding box and the cell wall. This is NOT
# the manufacturability clearance used in the MATLAB explorer: 0.2 mm there
# reflects what an FDM nozzle can print. For deciding which simulated points
# are physically valid, a small margin is enough. Statistics scripts use
# VALIDITY_CLEARANCE_MM to decide inclusion.
VALIDITY_CLEARANCE_MM = 0.01

# Where a sweep counts as reaching the edge. Slightly larger than the validity
# clearance so that the recommended final sample is comfortably inside.
EDGE_MARGIN_MM = 0.01

# Largest acceptable spacing for a recommended additional sweep.
TARGET_STEP_MM = 0.02


# --------------------------------------------------------------------------- #
# Plot cosmetics
# --------------------------------------------------------------------------- #
#
# One place to change how every figure looks. Sizes are inches; the figure size
# suits a single-column thesis page.

PLOT_FIGSIZE = (6.5, 4.2)
PLOT_DPI = 300
PLOT_FORMAT = "pdf"              # pdf is vector; png/svg also work

PLOT_AXIS_LABEL_SIZE = 13
PLOT_TICK_LABEL_SIZE = 12
PLOT_TITLE_SIZE = 13
PLOT_LEGEND_SIZE = 11
PLOT_GRID = True
PLOT_GRID_ALPHA = 0.30
PLOT_GRID_LINESTYLE = ":"

# Default for whether figures carry a title. Scripts expose it as --titles/--no-titles
# so the same plot can go into a document with a caption instead.
PLOT_SHOW_TITLE = True

# Colour cycle for histogram bars, one per P_ratio within a panel.
PLOT_BAR_COLORS = [
    "#1f4e79",   # dark blue
    "#2e7d32",   # green
    "#b8860b",   # ochre
    "#8b2c8b",   # purple
    "#c62828",   # red
    "#00695c",   # teal
]

PLOT_BAR_ALPHA = 0.85
PLOT_BAR_EDGE = "white"
PLOT_BAR_EDGE_WIDTH = 0.5


# --------------------------------------------------------------------------- #
# Histogram binning
# --------------------------------------------------------------------------- #

# Number of bins for the `a` coverage histograms.
A_HIST_BINS = 40

# `ratio` takes few discrete values, so it is binned one bar per distinct
# value rather than by a numeric bin count.
RATIO_AS_CATEGORIES = True


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

# Maximum characters per line in printed summary blocks, kept consistent with
# the table renderer.
SUMMARY_WRAP_COLS = 78
LEV_SEP = "-" * 30
