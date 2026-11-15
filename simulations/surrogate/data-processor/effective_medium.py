"""
Effective-medium model for the elliptical pillar cell.

The pillar is modelled as an *elliptical cylinder*, not a prolate spheroid: it
has a constant elliptical cross-section and flat ends, so the cylinder is the
correct shape class. This also gives exact, simple depolarisation factors with
no eccentricity, no inverse hyperbolic functions and no special handling of the
near-circular limit.

The depolarisation factor differs along the major and minor axes, so the two
polarisations see different effective permittivities. That is the mechanism
behind the inter-axis phase difference, and exposing it as a feature gives the
surrogate the physics directly instead of asking it to rediscover it from
geometry.

Everything here is closed-form and cheap, so it is computed once per sample at
dataset-build time.

References
----------
Sihvola, Electromagnetic Mixing Formulas and Applications (1999)
    - why depolarisation factors enter the theory at all
    - the anisotropic Maxwell-Garnett rule
Zhang & Wu, Sci. Rep. 4, 7892 (2014)
    - the depolarisation factors of an elliptical cylinder
"""

from __future__ import annotations

import numpy as np


def depolarisation_elliptical_cylinder(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Depolarisation factors of an elliptical cylinder, semi-axes a and b.

        N_a = b / (a + b),   N_b = a / (a + b),   N_z = 0

    with N_a + N_b + N_z = 1. The axial factor vanishes because the cylinder is
    taken as infinite along its axis, so no depolarisation occurs there.

    The factors are named after the semi-axis they belong to rather than after
    a coordinate direction: the ellipse is rotated by 45 degrees relative to
    the array axes, so x/y/z labels would be ambiguous in this codebase.

    Limits
    ------
    a = b   -> (1/2, 1/2, 0)   circular cylinder
    b -> 0  -> (0, 1, 0)       thin strip

    Parameters
    ----------
    a : semi-major axis (mm)
    b : semi-minor axis (mm), b = a / ratio

    Returns
    -------
    (N_a, N_b, N_z), arrays matching the input shape.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        n_a = b / total
        n_b = a / total

    n_z = np.zeros_like(n_a)
    return n_a, n_b, n_z


def maxwell_garnett_axis(
    eps_m: np.ndarray | float,
    eps_p: np.ndarray | float,
    fill: np.ndarray,
    n_j: np.ndarray,
) -> np.ndarray:
    """
    Maxwell-Garnett effective permittivity along one axis of an ellipsoidal
    (or cylindrical) inclusion.

    Explicit form of the anisotropic rule:

        eps_eff = eps_m * [eps_m + (eps_p - eps_m)(N + f(1 - N))]
                        / [eps_m + (eps_p - eps_m) N (1 - f)]

    This is the rearrangement of the implicit shape-weighted form used by
    Zhang & Wu,

        (eps_eff - eps_m) / (N eps_eff + (1 - N) eps_m)
            = f (eps_p - eps_m) / (N eps_p + (1 - N) eps_m)

    and reduces to the standard spherical Maxwell-Garnett expression when
    N = 1/3. Returns eps_m as f -> 0.

    Parameters
    ----------
    eps_m : host permittivity (air, so 1.0)
    eps_p : inclusion permittivity (the printed dielectric)
    fill  : volume fraction of the inclusion
    n_j   : depolarisation factor along this axis
    """
    eps_m = np.asarray(eps_m, dtype=float)
    eps_p = np.asarray(eps_p, dtype=float)
    fill = np.asarray(fill, dtype=float)
    n_j = np.asarray(n_j, dtype=float)

    delta = eps_p - eps_m

    numerator = eps_m + delta * (n_j + fill * (1.0 - n_j))
    denominator = eps_m + delta * n_j * (1.0 - fill)

    return eps_m * numerator / denominator


def effective_permittivities(
    a: np.ndarray,
    b: np.ndarray,
    eps_r: np.ndarray,
    fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Effective permittivity along the major and minor axes.

    The host is air (eps_m = 1) and the inclusion is the printed dielectric
    (eps_p = eps_r). The two axes use N_a and N_b respectively, which is what
    makes the result anisotropic.

    Parameters
    ----------
    a     : semi-major axis (mm)
    b     : semi-minor axis (mm)
    eps_r : relative permittivity of the printed material
    fill  : fill fraction, pi*a*b / P^2

    Returns
    -------
    (eps_eff_major, eps_eff_minor)
    """
    n_a, n_b, _ = depolarisation_elliptical_cylinder(a, b)

    eps_major = maxwell_garnett_axis(1.0, eps_r, fill, n_a)
    eps_minor = maxwell_garnett_axis(1.0, eps_r, fill, n_b)

    return eps_major, eps_minor
