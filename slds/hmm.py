"""Forward-filter backward-sample for a finite Markov chain.

The filter starts from the stationary distribution, predicts with the
transition matrix, then reweights by the Gaussian emission. The backward
sampler draws the mode path from those filtered probabilities.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from slds._dtype import enable_x64
from slds.types import RegimeState

enable_x64()


def stationary_distribution(transition: Float[Array, "K K"]) -> Float[Array, "K"]:
    """Stationary distribution from a linear solve, without an eigensolver.

    Args:
        transition: Row-stochastic matrix.

    Returns:
        Probability vector. Rows that do not yield a finite solution fall
        back to the uniform distribution.
    """

    regimes = transition.shape[0]
    eye = jnp.eye(regimes, dtype=transition.dtype)
    system = eye - transition.T + jnp.ones((regimes, regimes), dtype=transition.dtype)
    weights = jnp.linalg.solve(system, jnp.ones(regimes, dtype=transition.dtype))
    weights = jnp.clip(weights, 0.0, None)
    total = jnp.sum(weights)
    uniform = jnp.full(regimes, 1.0 / regimes, dtype=transition.dtype)
    return jnp.where(jnp.isfinite(total) & (total > 0.0), weights / total, uniform)


def gaussian_log_emissions(
    series: Float[Array, "T"],
    features: Float[Array, "T F"],
    state: RegimeState,
) -> Float[Array, "T K"]:
    """Log emission density of each regime at each time.

    Args:
        series: Daily log variance.
        features: HAR features aligned with ``series``.
        state: Current coefficients and scales.

    Returns:
        Array of shape ``(time, regimes)``.
    """

    regimes = state.sigma.shape[0]
    residual = series[:, None] - (features @ state.coef[regimes:])[:, None] - state.coef[:regimes]
    safe_sigma = jnp.maximum(state.sigma, 1e-8)
    return -0.5 * (residual / safe_sigma) ** 2 - jnp.log(safe_sigma) - 0.5 * jnp.log(2.0 * jnp.pi)


def forward_filter(
    log_emissions: Float[Array, "T K"],
    transition: Float[Array, "K K"],
    initial: Float[Array, "K"],
) -> Float[Array, "T K"]:
    """Normalized forward filter.

    Args:
        log_emissions: Per-time emission log densities.
        transition: Row-stochastic matrix.
        initial: Distribution before the first emission. The regime HAR uses
            the stationary distribution.

    Returns:
        Filtered regime probabilities. Row ``t`` conditions on emissions
        through ``t``.
    """

    def step(
        alpha: Float[Array, "K"], emission: Float[Array, "K"]
    ) -> tuple[Float[Array, "K"], Float[Array, "K"]]:
        predicted = alpha @ transition
        score = jnp.log(jnp.maximum(predicted, 1e-300)) + emission
        posterior = jnp.exp(score - jnp.max(score))
        posterior = posterior / jnp.sum(posterior)
        return posterior, posterior

    initial_score = jnp.log(jnp.maximum(initial, 1e-300)) + log_emissions[0]
    first = jnp.exp(initial_score - jnp.max(initial_score))
    first = first / jnp.sum(first)
    _final, rest = lax.scan(step, first, log_emissions[1:])
    return jnp.concatenate([first[None], rest], axis=0)


def _forecast_log_emissions(
    series: Float[Array, "T"],
    features: Float[Array, "T F"],
    intercept: Float[Array, "K"],
    slopes: Float[Array, "F"],
    sigma: Float[Array, "K"],
) -> Float[Array, "T K"]:
    """Gaussian log emission density of one regime-HAR draw."""

    safe_sigma = jnp.maximum(sigma, 1e-8)
    residual = series[:, None] - (features @ slopes)[:, None] - intercept
    return -0.5 * (residual / sigma) ** 2 - jnp.log(safe_sigma) - 0.5 * jnp.log(2.0 * jnp.pi)


def _filter_one_draw(
    series: Float[Array, "T"],
    features: Float[Array, "T F"],
    intercept: Float[Array, "K"],
    slopes: Float[Array, "F"],
    sigma: Float[Array, "K"],
    transition: Float[Array, "K K"],
) -> Float[Array, "T K"]:
    """Filter one draw from its stationary distribution."""

    emissions = _forecast_log_emissions(series, features, intercept, slopes, sigma)
    return forward_filter(emissions, transition, stationary_distribution(transition))


@jax.jit
def filter_regime_draws(
    series: Float[Array, "T"],
    features: Float[Array, "T F"],
    intercept: Float[Array, "D K"],
    slopes: Float[Array, "D F"],
    sigma: Float[Array, "D K"],
    transition: Float[Array, "D K K"],
) -> Float[Array, "T D K"]:
    """Filtered regime probabilities for every forecast draw.

    Draws are mapped with ``vmap``. Origins are positions in the returned
    time axis, so one call covers every origin in a refit block.

    Args:
        series: Log variance from the start of the refit window through the
            last origin in the block.
        features: HAR features aligned with ``series``.
        intercept: Per-draw regime intercepts.
        slopes: Per-draw shared HAR slopes.
        sigma: Per-draw regime scales.
        transition: Per-draw transition matrices.

    Returns:
        Probabilities of shape ``(time, draws, regimes)``. Row ``t``
        conditions on observations through ``t``.
    """

    stacked = jax.vmap(_filter_one_draw, in_axes=(None, None, 0, 0, 0, 0))(
        series, features, intercept, slopes, sigma, transition
    )
    return jnp.swapaxes(stacked, 0, 1)


def backward_sample(
    key: PRNGKeyArray,
    filtered: Float[Array, "T K"],
    transition: Float[Array, "K K"],
) -> Int[Array, "T"]:
    """Draw a mode path from the filtered probabilities and the transitions.

    Args:
        key: JAX random key.
        filtered: Output of :func:`forward_filter`.
        transition: Row-stochastic matrix.

    Returns:
        Integer mode at each time.
    """

    length, regimes = filtered.shape
    uniforms = random.uniform(key, (length,))

    def categorical(weights: Float[Array, "K"], draw: Float[Array, ""]) -> Int[Array, ""]:
        cumulative = jnp.cumsum(weights)
        total = cumulative[-1]
        safe = jnp.where(total > 0.0, cumulative / total, jnp.linspace(1.0 / regimes, 1.0, regimes))
        return jnp.sum(safe < draw).astype(jnp.int32).clip(0, regimes - 1)

    last = categorical(filtered[-1], uniforms[-1])

    def step(
        mode_next: Int[Array, ""],
        inputs: tuple[Float[Array, "K"], Float[Array, ""]],
    ) -> tuple[Int[Array, ""], Int[Array, ""]]:
        belief, draw = inputs
        mode = categorical(belief * transition[:, mode_next], draw)
        return mode, mode

    _first, earlier = lax.scan(step, last, (filtered[:-1], uniforms[:-1]), reverse=True)
    return jnp.concatenate([earlier, last[None]], axis=0)


def filter_and_sample(
    key: PRNGKeyArray,
    series: Float[Array, "T"],
    features: Float[Array, "T F"],
    state: RegimeState,
) -> tuple[Float[Array, "T K"], Int[Array, "T"]]:
    """Filter one regime-HAR state and draw its mode path.

    Args:
        key: JAX random key.
        series: Daily log variance.
        features: HAR features.
        state: Coefficients, scales, and transitions.

    Returns:
        Filtered probabilities and one sampled mode path.
    """

    emissions = gaussian_log_emissions(series, features, state)
    filtered = forward_filter(
        emissions, state.transition, stationary_distribution(state.transition)
    )
    modes = backward_sample(key, filtered, state.transition)
    return filtered, modes


def complete_loglik(
    series: Float[Array, "T"],
    features: Float[Array, "T F"],
    state: RegimeState,
) -> Float[Array, ""]:
    """Complete-data log likelihood of one regime-HAR state.

    The emission term is the Gaussian density of the sampled regime path.
    The transition term is the sum of log transition probabilities along
    that path.

    Args:
        series: Daily log variance.
        features: HAR features.
        state: Conditioned state.

    Returns:
        Scalar log likelihood.
    """

    regimes = state.sigma.shape[0]
    mean = state.coef[state.modes] + features @ state.coef[regimes:]
    variance = jnp.maximum(state.sigma[state.modes] ** 2, 1e-12)
    emission = -0.5 * jnp.sum((series - mean) ** 2 / variance + jnp.log(2.0 * jnp.pi * variance))
    steps = jnp.sum(jnp.log(state.transition[state.modes[:-1], state.modes[1:]] + 1e-300))
    return emission + steps
