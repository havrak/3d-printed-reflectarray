"""
Configuration for the data processor.

Everything tunable lives here: paths, the design frequency, the clearance
rule, the eigenbasis search, the feature transforms and the filters.

Nothing in this module imports the rest of the package, so it can be edited
freely without touching the processing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Speed of light
# --------------------------------------------------------------------------- #

C0_MM_GHZ = 299.792458  # mm * GHz, so lambda0[mm] = C0_MM_GHZ / f0[GHz]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Directory that contains one subdirectory per CST sweep; each subdirectory
# holds the .s2p files plus a result_navigator.csv.
DEFAULT_DATA_ROOT = Path("simulations/data")

# Where the processed dataset is written.
DEFAULT_OUTPUT = Path("simulations/surrogate/data-processor/out/dataset.npz")


# --------------------------------------------------------------------------- #
# Design point
# --------------------------------------------------------------------------- #

F0_GHZ = 30.0
LAMBDA0_MM = C0_MM_GHZ / F0_GHZ


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

# Minimum gap between the rotated ellipse's bounding box and the cell edge, mm.
# This clearance should be way smaller then actually clearance used in the design
# The model should achieve good performance on the edges of the interval, even though
# they won't really be used in the final model
#
# as CST is modeling elipse with a polygon setting to zero, could introducer some samples
# where the geometry actually overflowed the base size
CLEARANCE_MM = 0.01

# Ellipse major axis is rotated by this angle relative to the array axes.
ELLIPSE_ROTATION_DEG = 45.0


# --------------------------------------------------------------------------- #
# Eigenbasis search and validation
# --------------------------------------------------------------------------- #

# Angular scan step for the congruence transform, degrees.
THETA_SCAN_STEP_DEG = 0.5

# A cell is only in its eigenbasis if the recovered angle sits near the
# geometric rotation. Deviations mean the diagonal symmetry is broken.
THETA_STAR_TOL_DEG = 10.0

# Reject samples whose residual (linear, summed off-diagonal magnitude)
# exceeds this. Mainly completeness; the relative residual filter below is
# the one that actually discriminates.
RESIDUAL_MAX_LINEAR = 0.05


# --------------------------------------------------------------------------- #
# Feature transforms
# --------------------------------------------------------------------------- #
#
# Each transform takes the raw per-sample geometry (in mm where absolute) and
# returns one feature column. They are applied in the order listed, so adding
# or replacing a feature is a one-line change here.


@dataclass
class RawGeometry:
    """Raw per-sample geometry, before any transform. Length = n_samples."""

    a_mm: np.ndarray          # ellipse semi-major axis, mm
    ratio: np.ndarray         # a / b, always > 1
    p_ratio: np.ndarray       # cell size as a fraction of lambda0 (0.5 = lambda0/2)
    eps_r: np.ndarray         # relative permittivity
    h_pillar_mm: np.ndarray   # pillar height, absolute mm
    h_base_mm: np.ndarray     # base layer thickness, absolute mm

    def __len__(self) -> int:
        return int(self.a_mm.shape[0])


def _lambda_g_factor(eps_r: np.ndarray) -> np.ndarray:
    """sqrt(eps_r): converts physical length to electrical length in the medium."""
    return np.sqrt(eps_r)


def feat_fill_fraction(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Ellipse area divided by cell area.

    This is the dielectric fill fraction, the quantity that actually sets the
    effective permittivity of the cell. Much closer to physics than the raw
    semi-axis length.
    """
    b_mm = g.a_mm / g.ratio
    cell_mm = g.p_ratio * lam0
    return np.pi * g.a_mm * b_mm / (cell_mm ** 2)


def feat_electrical_height_pillar(g: RawGeometry, lam0: float) -> np.ndarray:
    """Pillar height in electrical wavelengths inside the dielectric."""
    return (g.h_pillar_mm / lam0) * _lambda_g_factor(g.eps_r)


def feat_electrical_height_base(g: RawGeometry, lam0: float) -> np.ndarray:
    """Base thickness in electrical wavelengths inside the dielectric."""
    return (g.h_base_mm / lam0) * _lambda_g_factor(g.eps_r)


def feat_ratio(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Ellipse aspect ratio a / b, the direct geometric anisotropy parameter.

    Kept linear rather than logarithmic. The physical chain from geometry to
    phase difference is

        a/b  ->  shape  ->  N_a, N_b  ->  eps_eff  ->  delta_phi

    and every step is algebraic or rational, none exponential or logarithmic,
    so there is no mechanism that would make the response logarithmic in
    elongation. A log would also degrade relative precision exactly where the
    sampling is densest (ratio near 1, where ln -> 0).

    The depolarisation factors are deliberately not carried as separate
    features: they are deterministic functions of the ratio alone, and their
    effect enters through the effective permittivities below.
    """
    return g.ratio


def feat_eps_r(g: RawGeometry, lam0: float) -> np.ndarray:
    """Relative permittivity itself (low Q vs high Q)."""
    return g.eps_r


def feat_p_ratio(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Cell footprint as a fraction of the design wavelength.

    This is the electrical size of the cell, and it is not recoverable from the
    fill fraction: two cells can share a fill fraction while having different
    electrical size. It matters because it sets the mutual coupling between
    adjacent elements, the grating-lobe threshold, and the lateral cutoff of
    the modes trapped in the cell, all of which shift the phase response.
    """
    return g.p_ratio


def feat_a_over_P(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Semi-major axis relative to the cell size, a / P.

    The ellipse is described relative to the cell rather than in millimetres
    because the lateral boundary condition and the coupling to neighbours are
    governed by how much of the cell the pillar occupies, not by its absolute
    physical size. This also makes the feature invariant under the 30 -> 15 GHz
    frequency scaling, which a raw millimetre value would not be.

    Retained directly despite being closely related to the fill fraction: the
    phase response is not monotonic in 'a' (it rises through resonance and then
    falls), so the network needs this coordinate itself to locate the
    resonance.
    """
    cell_mm = g.p_ratio * lam0
    with np.errstate(divide="ignore", invalid="ignore"):
        return g.a_mm / cell_mm


# ---- effective-medium features -------------------------------------------
#
# These follow Sihvola's anisotropic Maxwell-Garnett rule, with the
# depolarisation factors of an elliptical cylinder taken from Zhang & Wu.
# See effective_medium.py for the derivation.


def _fill_and_eps(g: RawGeometry, lam0: float):
    """Shared intermediate: fill fraction and effective permittivities."""
    import effective_medium as em

    fill = feat_fill_fraction(g, lam0)
    b_mm = g.a_mm / g.ratio
    eps_major, eps_minor = em.effective_permittivities(
        g.a_mm, b_mm, g.eps_r, fill
    )
    return fill, eps_major, eps_minor


def feat_eps_eff_major(g: RawGeometry, lam0: float) -> np.ndarray:
    """Effective permittivity along the ellipse major axis (Maxwell-Garnett)."""
    _, eps_major, _ = _fill_and_eps(g, lam0)
    return eps_major


def feat_eps_eff_minor(g: RawGeometry, lam0: float) -> np.ndarray:
    """Effective permittivity along the ellipse minor axis (Maxwell-Garnett)."""
    _, _, eps_minor = _fill_and_eps(g, lam0)
    return eps_minor


# The feature set. Order defines the column order of X.
FEATURE_TRANSFORMS: list[tuple[str, Callable[[RawGeometry, float], np.ndarray]]] = [
    # geometric / material
    ("p_ratio",             feat_p_ratio),
    ("a_over_P",            feat_a_over_P),
    ("fill_fraction",       feat_fill_fraction),
    ("h_pillar_electrical", feat_electrical_height_pillar),
    ("h_base_electrical",   feat_electrical_height_base),
    ("ratio",               feat_ratio),
    ("eps_r",               feat_eps_r),
    # effective-medium derived
    ("eps_eff_major",       feat_eps_eff_major),
    ("eps_eff_minor",       feat_eps_eff_minor),
]


# --------------------------------------------------------------------------- #
# Band descriptors
# --------------------------------------------------------------------------- #
#
# The .s2p export covers a narrow band around f0. Instead of collapsing it to
# a single number (which breaks when a resonance does/does not fall inside),
# every sample carries a fixed-length triple that is always well defined.

BAND_SLOPE_DF_GHZ = 0.1     # half-width of the window for the least-squares slope fit, GHz
RESONANCE_DIP_DB = -3.0     # |S| below this counts as a resonance
BAND_FEATURE_NAMES = ["dphi_df", "band_spread", "resonance_in_band"]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

OUTPUT_FORMAT = "npz"       # "npz" | "csv"
