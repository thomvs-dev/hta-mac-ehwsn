"""Closed-form moments for HEART-CH's rectified Gaussian harvest model."""

from __future__ import annotations

from math import erf, pi, sqrt

import numpy as np


def rectified_gaussian_moments(
    mean,
    variance,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return g1=E[max(0,scale*Y)] and g2=E[max(0,scale*Y)^2].

    This implements HEART-CH manuscript Eqs. 13-14 for
    Y ~ Normal(mean, variance) and non-negative ``scale``.
    """
    mu = np.asarray(mean, dtype=np.float64)
    var = np.asarray(variance, dtype=np.float64)
    if np.any(var < 0.0) or scale < 0.0:
        raise ValueError("variance and scale must be non-negative")
    sigma = np.sqrt(var)
    safe_sigma = np.maximum(sigma, 1e-15)
    eta = mu / safe_sigma
    phi = np.exp(-0.5 * eta**2) / sqrt(2.0 * pi)
    cdf = 0.5 * (
        1.0 + np.vectorize(erf, otypes=[np.float64])(eta / sqrt(2.0))
    )
    g1 = scale * (sigma * phi + mu * cdf)
    g2 = scale**2 * ((mu**2 + var) * cdf + mu * sigma * phi)

    deterministic = sigma == 0.0
    if np.any(deterministic):
        clipped = np.maximum(0.0, scale * mu)
        g1 = np.where(deterministic, clipped, g1)
        g2 = np.where(deterministic, clipped**2, g2)
    return g1, g2


def next_rectified_statistics(
    transition,
    mean,
    variance,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next-round mean and variance conditioned on each current state."""
    matrix = np.asarray(transition, dtype=np.float64)
    g1, g2 = rectified_gaussian_moments(mean, variance, scale)
    forecast = matrix @ g1
    second = matrix @ g2
    return forecast, np.maximum(0.0, second - forecast**2)
