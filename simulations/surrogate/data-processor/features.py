"""
Geometry-derived input features.

Everything here maps a per-sample geometry to one feature column. Each transform
takes a `RawGeometry` (arrays over samples) and the design wavelength, and
returns one array of the same length.

The feature set actually used is selected in `config.py`; this module only
provides the functions. To add a feature, define it here and register it there.

All features are normalised to the design frequency, so that the 30 -> 15 GHz
scaling leaves them invariant. Raw millimetre values are deliberately not used
as features; the un-transformed geometry is kept separately in `X_raw` for
provenance.

Band-derived features live in `band_features.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RawGeometry:
    """Raw per-sample geometry, before any transform. Length = n_samples."""

    a_mm: np.ndarray          # ellipse semi-major axis, mm
    ratio: np.ndarray         # a / b, >= 1
    p_ratio: np.ndarray       # cell size as a fraction of lambda0 (0.5 = lambda0/2)
    eps_r: np.ndarray         # relative permittivity
    h_pillar_mm: np.ndarray   # pillar height, absolute mm
    h_base_mm: np.ndarray     # base layer thickness, absolute mm

    def __len__(self) -> int:
        return int(self.a_mm.shape[0])

    @property
    def b_mm(self) -> np.ndarray:
        """Semi-minor axis, from a and the aspect ratio."""
        return self.a_mm / self.ratio


def cell_mm_of(g: RawGeometry, lam0: float) -> np.ndarray:
    """Cell side length in mm, from the normalised footprint."""
    return g.p_ratio * lam0


def _lambda_g_factor(eps_r: np.ndarray) -> np.ndarray:
    """sqrt(eps_r): converts physical length to electrical length in the medium."""
    return np.sqrt(eps_r)


# --------------------------------------------------------------------------- #
# Cell and ellipse geometry
# --------------------------------------------------------------------------- #

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
    physical size. This also makes the feature invariant under the frequency
    scaling, which a raw millimetre value would not be.

    Retained directly despite being closely related to the fill fraction: the
    phase response is not monotonic in `a` (it rises through resonance and then
    falls), so the network needs this coordinate itself to locate the
    resonance.
    """
    cell_mm = cell_mm_of(g, lam0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return g.a_mm / cell_mm


def feat_fill_fraction(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Ellipse area divided by cell area, pi a b / P^2.

    This is the dielectric loading of the cell and the quantity that sets its
    effective permittivity, so it is much closer to physics than the semi-axis
    length alone. Dimensionless and frequency-invariant, since a, b and P all
    scale together.
    """
    cell_mm = cell_mm_of(g, lam0)
    return np.pi * g.a_mm * g.b_mm / (cell_mm ** 2)


# --------------------------------------------------------------------------- #
# Electrical heights
# --------------------------------------------------------------------------- #

def feat_h_pillar_electrical(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Pillar height in electrical wavelengths inside the dielectric.

    The reflection phase of a grounded dielectric slab is governed by its
    electrical height rather than its physical height, so normalising by
    lambda0 and by the refractive index sqrt(eps_r) makes the feature
    scale-free and collapses two independent variables into the one that
    matters.
    """
    return (g.h_pillar_mm / lam0) * _lambda_g_factor(g.eps_r)


def feat_h_base_electrical(g: RawGeometry, lam0: float) -> np.ndarray:
    """
    Base thickness in electrical wavelengths inside the dielectric.

    Carried separately from the pillar because the base is a structural
    element whose thickness is not required to scale with the cell. If it is
    held constant while the frequency halves, the stack is no longer a scaled
    copy of itself, and this feature has to expose that.
    """
    return (g.h_base_mm / lam0) * _lambda_g_factor(g.eps_r)


# --------------------------------------------------------------------------- #
# Material and anisotropy
# --------------------------------------------------------------------------- #

def feat_eps_r(g: RawGeometry, lam0: float) -> np.ndarray:
    """Relative permittivity itself, which sets the Q of the resonator."""
    return g.eps_r


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
    effect enters through the effective permittivities.
    """
    return g.ratio


# --------------------------------------------------------------------------- #
# Effective-medium derived
# --------------------------------------------------------------------------- #
#
# These follow Sihvola's anisotropic Maxwell-Garnett rule, with the
# depolarisation factors of an elliptical cylinder taken from Zhang & Wu.
# See effective_medium.py for the derivation.


def _effective_permittivities(g: RawGeometry, lam0: float):
    """Shared intermediate: fill fraction and both effective permittivities."""
    import effective_medium as em

    fill = feat_fill_fraction(g, lam0)
    return em.effective_permittivities(g.a_mm, g.b_mm, g.eps_r, fill)


def feat_eps_eff_major(g: RawGeometry, lam0: float) -> np.ndarray:
    """Effective permittivity along the ellipse major axis (Maxwell-Garnett)."""
    eps_major, _ = _effective_permittivities(g, lam0)
    return eps_major


def feat_eps_eff_minor(g: RawGeometry, lam0: float) -> np.ndarray:
    """Effective permittivity along the ellipse minor axis (Maxwell-Garnett)."""
    _, eps_minor = _effective_permittivities(g, lam0)
    return eps_minor
