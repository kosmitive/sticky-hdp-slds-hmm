"""Conjugate draws used by the regime HAR and the SLDS.

Inverse-Wishart draws use the Bartlett decomposition of the corresponding
Wishart. A matrix-normal draw has row covariance across rows and column
covariance across columns.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray

from sticky_hdp_slds._dtype import enable_x64

enable_x64()


def sample_gaussian(
    key: PRNGKeyArray, mean: Float[Array, "D"], covariance: Float[Array, "D D"]
) -> Float[Array, "D"]:
    """Draw one Gaussian vector.

    Args:
        key: JAX random key.
        mean: Mean.
        covariance: Symmetric positive definite covariance.

    Returns:
        One draw.
    """

    chol = jnp.linalg.cholesky(0.5 * (covariance + covariance.T))
    return mean + chol @ random.normal(key, mean.shape)


def sample_wishart(
    key: PRNGKeyArray, degrees: Float[Array, ""], scale: Float[Array, "D D"]
) -> Float[Array, "D D"]:
    """Draw one Wishart matrix.

    Args:
        key: JAX random key.
        degrees: Degrees of freedom, greater than ``dim - 1``.
        scale: Scale matrix.

    Returns:
        A positive definite draw.
    """

    dim = scale.shape[-1]
    chi_key, normal_key = random.split(key)
    index = jnp.arange(dim, dtype=degrees.dtype)
    chi = random.gamma(chi_key, (degrees - index) / 2.0) * 2.0
    lower = random.normal(normal_key, (dim, dim))
    factor = jnp.tril(lower, k=-1) + jnp.diag(jnp.sqrt(jnp.maximum(chi, 1e-18)))
    chol = jnp.linalg.cholesky(0.5 * (scale + scale.T))
    draw = chol @ factor
    return draw @ draw.T


def sample_invwishart(
    key: PRNGKeyArray, degrees: Float[Array, ""], scale: Float[Array, "D D"]
) -> Float[Array, "D D"]:
    """Draw one inverse-Wishart matrix.

    Args:
        key: JAX random key.
        degrees: Degrees of freedom.
        scale: Scale matrix. The mean of the draw is ``scale / (df - dim - 1)``.

    Returns:
        A positive definite draw.
    """

    precision = jnp.linalg.inv(0.5 * (scale + scale.T))
    return jnp.linalg.inv(sample_wishart(key, degrees, precision))


def sample_matrix_normal(
    key: PRNGKeyArray,
    mean: Float[Array, "R C"],
    rowcov: Float[Array, "R R"],
    colcov: Float[Array, "C C"],
) -> Float[Array, "R C"]:
    """Draw one matrix-normal matrix.

    Args:
        key: JAX random key.
        mean: Mean matrix.
        rowcov: Covariance across rows.
        colcov: Covariance across columns.

    Returns:
        One draw with the same shape as ``mean``.
    """

    rows, cols = mean.shape
    shock = random.normal(key, (rows, cols))
    left = jnp.linalg.cholesky(0.5 * (rowcov + rowcov.T))
    right = jnp.linalg.cholesky(0.5 * (colcov + colcov.T))
    return mean + left @ shock @ right.T
