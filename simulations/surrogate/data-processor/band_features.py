"""
Band-derived features, computed from a multi-frequency Touchstone export.

The .s2p covers a narrow band around the design frequency (1 GHz at 30 GHz in
the current sweeps). Collapsing that band to a single scalar is fragile: a
resonance may or may not fall inside it, so the scalar is not comparable across
samples.

Instead every band feature is a function of the whole band, and the feature set
actually used is selected in `config.py`. Each function receives a `BandContext`
holding the band data for one sample, and returns one scalar. A feature may use
only the part of the context it needs.

This mirrors the structure of `features.py`, where each function receives a
`RawGeometry` and returns one column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BandContext:
    """
    Everything a band feature might need, for one sample.

    Attributes
    ----------
    freq_ghz        : frequency axis of the export, GHz
    phi_major_deg   : major-axis reflection phase over that axis, degrees
    mag_major       : major-axis reflection magnitude over that axis, linear
    f0_ghz          : design frequency, GHz
    slope_step_ghz  : half-width of the window used for the slope fit, GHz
    resonance_dip_db: |S| below this counts as a resonance, dB
    """

    freq_ghz: np.ndarray
    phi_major_deg: np.ndarray
    mag_major: np.ndarray
    f0_ghz: float
    slope_step_ghz: float = 0.1
    resonance_dip_db: float = -3.0


# --------------------------------------------------------------------------- #
# Band features
# --------------------------------------------------------------------------- #

def feat_dphi_df(ctx: BandContext) -> float:
    """
    Major-axis phase slope at the design frequency, in deg/GHz.

    Obtained as the gradient of a least-squares linear fit of the unwrapped
    phase over a window of f0 +/- slope_step_ghz. A linear fit is used rather
    than a two-point difference because the band is narrow and finite
    differencing would amplify numerical noise.

    The slope is the standard proxy for element bandwidth: what limits a
    reflectarray is the variation of phase with frequency, not its absolute
    value. Returns NaN when the window holds fewer than two points.
    """
    freq = ctx.freq_ghz
    if freq.size < 2:
        return float("nan")

    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ctx.phi_major_deg)))

    lo = ctx.f0_ghz - ctx.slope_step_ghz
    hi = ctx.f0_ghz + ctx.slope_step_ghz
    inside = (freq >= lo) & (freq <= hi)
    if np.count_nonzero(inside) < 2:
        return float("nan")

    slope, _ = np.polyfit(freq[inside], unwrapped[inside], 1)
    return float(slope)


def feat_band_spread(ctx: BandContext) -> float:
    """
    Peak-to-peak major-axis phase variation across the whole band, in degrees.

    Unwrapped first, so that a genuine 300 degree excursion is reported as 300
    rather than as the 60 degrees left after wrapping. This is the honest
    wideband measure: it is the phase variation the element would show if used
    over the band, and the absence of a resonance gives a small value, which is
    correct rather than missing.
    """
    if ctx.phi_major_deg.size == 0:
        return float("nan")
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ctx.phi_major_deg)))
    return float(np.ptp(unwrapped))


def feat_resonance_in_band(ctx: BandContext) -> float:
    """
    1 when the major-axis reflection dips below the threshold inside the band.

    A resonance shows up as a notch in |S|, which is the robust indicator.
    Returned as 0/1 rather than a float flag so that it can be used for
    filtering without introducing NaN, and so that "no resonance" is a valid
    value rather than a missing one.
    """
    if ctx.mag_major.size == 0:
        return 0.0
    with np.errstate(divide="ignore"):
        mag_db = 20.0 * np.log10(np.maximum(ctx.mag_major, 1e-12))
    return 1.0 if np.any(mag_db < ctx.resonance_dip_db) else 0.0


# Type alias for readability: a band feature maps one context to one scalar.
BandFeature = Callable[[BandContext], float]
