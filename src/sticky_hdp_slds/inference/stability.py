"""Schur-Cohn test for the HAR companion polynomial.

The companion matrix puts the distributed-lag coefficients in its first row.
All of its eigenvalues lie inside radius ``rho`` exactly when the scaled
polynomial is Schur stable. The reflection coefficients are the Jury table,
so the check never calls a nonsymmetric eigensolver.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from sticky_hdp_slds._dtype import enable_x64

enable_x64()

LAG = 22
WEEK = 5


def distributed_lags(slopes: Float[Array, "3"]) -> Float[Array, "22"]:
    """Expand HAR lags 1, 5, and 22 into 22 distributed-lag weights.

    Args:
        slopes: Daily, weekly, and monthly coefficients.

    Returns:
        First row of the companion matrix.
    """

    daily = slopes[0]
    week = slopes[1] / WEEK
    month = slopes[2] / LAG
    coef = jnp.full(LAG, month)
    coef = coef.at[1:WEEK].set(week + month)
    coef = coef.at[0].set(daily + week + month)
    return coef


def schur_stable(companion_row: Float[Array, "N"], radius: Float[Array, ""]) -> Bool[Array, ""]:
    """Return whether every companion root has absolute value below ``radius``.

    Args:
        companion_row: First row of the companion matrix.
        radius: Positive bound. The volatility model uses 0.999.

    Returns:
        True when every reflection coefficient has absolute value below one.
    """

    degree = companion_row.shape[0]
    powers = radius ** jnp.arange(1, degree + 1, dtype=companion_row.dtype)
    polynomial = jnp.concatenate(
        [jnp.ones((1,), dtype=companion_row.dtype), -companion_row / powers]
    )
    stable = jnp.asarray(True)
    for _step in range(degree):
        gamma = polynomial[-1] / polynomial[0]
        stable = stable & jnp.isfinite(gamma) & (jnp.abs(gamma) < 1.0)
        reflected = jnp.flip(polynomial)
        polynomial = polynomial[:-1] - gamma * reflected[:-1]
    return stable


def har_stable(slopes: Float[Array, "3"], radius: Float[Array, ""]) -> Bool[Array, ""]:
    """Schur-Cohn test for one HAR slope vector.

    Args:
        slopes: Daily, weekly, and monthly coefficients.
        radius: Root bound passed to :func:`schur_stable`.

    Returns:
        True when the companion roots lie inside ``radius``.
    """

    return schur_stable(distributed_lags(slopes), radius)
