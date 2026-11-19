"""
Network definition and loss.

Kept separate from the tuning loop so the deep-ensemble stage trains exactly the
same architecture and objective. A hyperparameter set produced by cv_tune.py is
therefore directly reusable.

Design notes
------------
Targets are the four real/imaginary components of the two principal reflection
coefficients plus the major-axis phase slope. Predicting complex coefficients
rather than phase avoids the 0/360 degree wrap discontinuity, which no regressor
can learn across.

No dropout: the uncertainty comes from a deep ensemble, which requires each
member to be a well-trained deterministic function.

No batch normalisation: its running statistics are estimated on the training
distribution and misbehave on inputs far from it, which is precisely the
extrapolation regime that matters here. Inputs and outputs are z-scored instead.
"""

from __future__ import annotations

import torch
from torch import nn

# Activation lookup, keyed by the names used in config.SEARCH_SPACE.
_ACTIVATIONS = {
    "SiLU": nn.SiLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
}


def build_mlp(
    n_inputs: int,
    n_outputs: int,
    n_layers: int,
    width: int,
    activation: str,
) -> nn.Module:
    """
    Fully connected network with a linear output layer.

    Depth counts hidden layers; the output layer is always linear because the
    target is an unbounded real vector inside the unit disk, and an output
    activation would squash the distribution and interfere with the passivity
    penalty.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; expected one of {list(_ACTIVATIONS)}"
        )

    act = _ACTIVATIONS[activation]

    layers: list[nn.Module] = []
    in_features = n_inputs
    for _ in range(n_layers):
        layers.append(nn.Linear(in_features, width))
        layers.append(act())
        in_features = width
    layers.append(nn.Linear(in_features, n_outputs))

    return nn.Sequential(*layers)


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_slope: float,
    lambda_pass: float,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    slope_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Weighted objective.

        L = MSE_4(complex targets) + w_slope * MSE(slope) + lambda_pass * passivity

    The reflection coefficients are the primary objective and carry the larger
    weight. Because all targets are z-scored before the loss is evaluated,
    w_slope is directly interpretable as the relative weight of the slope term
    rather than a scale artefact.

    Columns 0..3 of the prediction are Re/Im of the two principal coefficients
    and column 4 is the phase slope. The MSE terms are computed in standardised
    space, which is what makes the weighting meaningful.

    `slope_clip` bounds the standardised slope target before squaring. The slope
    is a derivative and therefore heavy-tailed: a handful of configurations near
    resonance have slopes orders of magnitude above the rest, and their squared
    error dominates the mean. Clipping bounds that influence without discarding
    the points, which matters because those are exactly the configurations where
    the slope is physically interesting.

    The passivity term must be evaluated on the *physical* coefficient, not the
    standardised one, so the prediction is un-scaled first using the per-column
    target statistics. It is zero whenever |R| <= 1 and therefore cannot distort
    well-behaved predictions; it only acts where the model would otherwise
    predict gain, a failure mode an unconstrained regressor can exhibit near
    resonance.

    Returns the total loss and a dict of its components for logging.
    """
    complex_mse = ((pred[:, :4] - target[:, :4]) ** 2).mean()

    slope_err = pred[:, 4] - target[:, 4]
    if slope_clip is not None:
        slope_err = torch.clamp(slope_err, -slope_clip, slope_clip)
    slope_mse = (slope_err ** 2).mean()

    # Un-standardise the four coefficient components before applying the
    # physical constraint.
    physical = pred[:, :4] * y_std[:4] + y_mean[:4]
    r_major = torch.sqrt(physical[:, 0] ** 2 + physical[:, 1] ** 2 + 1e-12)
    r_minor = torch.sqrt(physical[:, 2] ** 2 + physical[:, 3] ** 2 + 1e-12)

    passivity = (
        torch.relu(r_major - 1.0) ** 2 + torch.relu(r_minor - 1.0) ** 2
    ).mean()

    total = complex_mse + w_slope * slope_mse + lambda_pass * passivity
    return total, {
        "complex_mse": float(complex_mse.detach()),
        "slope_mse": float(slope_mse.detach()),
        "passivity": float(passivity.detach()),
    }
