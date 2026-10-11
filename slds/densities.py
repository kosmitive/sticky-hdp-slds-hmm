"""Gaussian log-densities used by the vectorized Gibbs updates."""

from __future__ import annotations

import numpy as np


def gaussian_logpdf(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Log density of a Gaussian with one shared covariance.

    Args:
        points: Observations, shape ``(..., dim)``.
        mean: Means broadcastable to ``points``.
        covariance: Positive definite covariance, shape ``(dim, dim)``.

    Returns:
        Log density with shape ``points.shape[:-1]``.
    """

    diff = np.asarray(points, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    dim = diff.shape[-1]
    flat = diff.reshape(-1, dim)
    _sign, logdet = np.linalg.slogdet(covariance)
    solved = np.linalg.solve(covariance, flat.T).T
    quad = np.einsum("ni,ni->n", flat, solved)
    log_density = -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)
    return log_density.reshape(diff.shape[:-1])
