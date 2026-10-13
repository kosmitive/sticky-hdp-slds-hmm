"""Sticky regime-HAR Gibbs sampler.

The observation model is a daily regression: shared HAR slopes, a separate
intercept and scale in each regime, and a Dirichlet transition prior with
mean self-stay 0.98. Sweeps run under ``lax.scan``. Chains, and equal-length
rolling windows, run under ``vmap``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from sticky_hdp_slds._dtype import enable_x64
from sticky_hdp_slds.inference.hmm import complete_loglik, filter_and_sample
from sticky_hdp_slds.inference.stability import har_stable
from sticky_hdp_slds.inference.types import RegimeData, RegimeDraw, RegimePrior, RegimeState

enable_x64()

SELF_STAY = 0.98
PRIOR_ALPHA = 1.0
PRIOR_KAPPA = 97.0
VARIANCE_A = 2.0
INTERCEPT_PRIOR_SD = 1.0
SLOPE_PRIOR_SD = 0.25
RADIUS_LIMIT = 0.999
MAX_REJECTS = 40


def make_prior(
    regimes: int,
    *,
    variance_b: Float[Array, ""],
    prior_mean: Float[Array, "P"],
) -> RegimePrior:
    """Build the volatility-model prior for one chain.

    Args:
        regimes: Number of regimes. Static.
        variance_b: Inverse-gamma scale, usually from the initial residual.
        prior_mean: Gaussian mean of the intercepts and slopes.

    Returns:
        Prior pytree.
    """

    prior_var = jnp.concatenate(
        [
            jnp.full(regimes, INTERCEPT_PRIOR_SD**2),
            jnp.full(3, SLOPE_PRIOR_SD**2),
        ]
    )
    return RegimePrior(
        alpha=jnp.asarray(PRIOR_ALPHA),
        kappa=jnp.asarray(PRIOR_KAPPA),
        variance_a=jnp.asarray(VARIANCE_A),
        variance_b=variance_b,
        prior_mean=prior_mean,
        prior_var=prior_var,
        radius_limit=jnp.asarray(RADIUS_LIMIT),
    )


def init_state(
    key: PRNGKeyArray, data: RegimeData, regimes: int
) -> tuple[RegimeState, RegimePrior]:
    """Initialize one chain from a pooled HAR regression.

    Args:
        key: JAX random key.
        data: Training series and features.
        regimes: Number of regimes.

    Returns:
        Initial state and the prior, including the residual scale.
    """

    count = data.series.shape[0]
    design = jnp.concatenate([jnp.ones((count, 1)), data.features], axis=1)
    beta = jnp.linalg.lstsq(design, data.series, rcond=None)[0]
    slopes = beta[1:]

    def shrink(current: Float[Array, "3"], _step: Int[Array, ""]) -> tuple[Float[Array, "3"], None]:
        stable = har_stable(current, jnp.asarray(RADIUS_LIMIT))
        return jnp.where(stable, current, current * 0.8), None

    slopes, _unused = jax.lax.scan(shrink, slopes, jnp.arange(30))
    slopes = jnp.where(
        har_stable(slopes, jnp.asarray(RADIUS_LIMIT)), slopes, jnp.zeros_like(slopes)
    )
    key, noise_key = random.split(key)
    pooled_scale = jnp.maximum(
        jnp.std(data.series - design @ jnp.concatenate([beta[:1], slopes])), 1e-3
    )
    offsets = jnp.linspace(-0.75, 0.75, regimes) * pooled_scale
    offsets = offsets + 0.05 * random.normal(noise_key, (regimes,))
    coef = jnp.concatenate([jnp.full(regimes, beta[0]) + offsets, slopes])
    # The initial residual scale is scored against the first intercept,
    # after the offsets have been added.
    residual_scale = jnp.maximum(jnp.std(data.series - data.features @ slopes - coef[0]), 1e-3)
    sigma = jnp.full(regimes, residual_scale)
    off = (1.0 - SELF_STAY) / (regimes - 1)
    transition = jnp.full((regimes, regimes), off).at[jnp.diag_indices(regimes)].set(SELF_STAY)
    modes = jnp.zeros(count, dtype=jnp.int32)
    state = RegimeState(coef=coef, sigma=sigma, transition=transition, modes=modes)
    prior_mean = jnp.concatenate([jnp.full(regimes, jnp.mean(coef[:regimes])), slopes])
    variance_b = (VARIANCE_A - 1.0) * residual_scale**2
    return state, make_prior(regimes, variance_b=variance_b, prior_mean=prior_mean)


def _sample_coefficients(
    key: PRNGKeyArray,
    data: RegimeData,
    state: RegimeState,
    prior: RegimePrior,
) -> tuple[Float[Array, "P"], Array]:
    regimes = state.sigma.shape[0]
    design = jnp.concatenate(
        [jax.nn.one_hot(state.modes, regimes, dtype=data.series.dtype), data.features],
        axis=1,
    )
    scale = 1.0 / jnp.maximum(state.sigma[state.modes], 1e-6)
    weighted_x = design * scale[:, None]
    weighted_y = data.series * scale
    precision = jnp.diag(1.0 / prior.prior_var) + weighted_x.T @ weighted_x
    precision = 0.5 * (precision + precision.T) + 1e-8 * jnp.eye(regimes + 3)
    chol = jnp.linalg.cholesky(precision)
    mean = jax.scipy.linalg.cho_solve(
        (chol, True), prior.prior_mean / prior.prior_var + weighted_x.T @ weighted_y
    )

    def body(
        carry: tuple[Int[Array, ""], PRNGKeyArray, Float[Array, "P"], Array],
    ) -> tuple[Int[Array, ""], PRNGKeyArray, Float[Array, "P"], Array]:
        attempt, draw_key, _draw, _ok = carry
        draw_key, subkey = random.split(draw_key)
        shock = random.normal(subkey, (regimes + 3,))
        draw = mean + jax.scipy.linalg.solve_triangular(chol.T, shock, lower=False)
        ok = har_stable(draw[regimes:], prior.radius_limit)
        return attempt + 1, draw_key, draw, ok

    _attempt, _key, draw, ok = jax.lax.while_loop(
        lambda carry: (carry[0] < MAX_REJECTS) & ~carry[3],
        body,
        (jnp.asarray(0), key, state.coef, jnp.asarray(False)),
    )
    good = ok & jnp.all(jnp.isfinite(chol))
    return jnp.where(good, draw, state.coef), good


def _sample_sigma(
    key: PRNGKeyArray,
    data: RegimeData,
    state: RegimeState,
    prior: RegimePrior,
) -> Float[Array, "K"]:
    regimes = state.sigma.shape[0]
    residual = data.series - state.coef[state.modes] - data.features @ state.coef[regimes:]
    indicators = jax.nn.one_hot(state.modes, regimes, dtype=data.series.dtype)
    counts = jnp.sum(indicators, axis=0)
    rss = indicators.T @ residual**2
    draws = random.gamma(key, prior.variance_a + 0.5 * counts)
    return jnp.maximum(jnp.sqrt((prior.variance_b + 0.5 * rss) / draws), 1e-4)


def _sample_transition(
    key: PRNGKeyArray,
    state: RegimeState,
    prior: RegimePrior,
) -> Float[Array, "K K"]:
    regimes = state.sigma.shape[0]
    counts = jnp.zeros((regimes, regimes)).at[state.modes[:-1], state.modes[1:]].add(1.0)
    concentration = prior.alpha + prior.kappa * jnp.eye(regimes) + counts
    return random.dirichlet(key, concentration)


def _order_regimes(state: RegimeState) -> RegimeState:
    regimes = state.sigma.shape[0]
    order = jnp.argsort(state.coef[:regimes])
    inverse = jnp.argsort(order)
    coef = state.coef.at[:regimes].set(state.coef[:regimes][order])
    return RegimeState(
        coef=coef,
        sigma=state.sigma[order],
        transition=state.transition[order][:, order],
        modes=inverse[state.modes].astype(jnp.int32),
    )


def regime_sweep(
    key: PRNGKeyArray,
    state: RegimeState,
    data: RegimeData,
    prior: RegimePrior,
) -> tuple[RegimeState, RegimeDraw]:
    """One pure Gibbs sweep of the sticky regime HAR.

    Args:
        key: JAX random key. It is split across the four blocks.
        state: State from the previous sweep.
        data: Series and HAR features.
        prior: Hyperparameters, constant across sweeps.

    Returns:
        Updated state and the draw recorded for diagnostics.
    """

    mode_key, coef_key, sigma_key, trans_key = random.split(key, 4)
    _filtered, modes = filter_and_sample(mode_key, data.series, data.features, state)
    proposed = state._replace(modes=modes)
    coef, accepted = _sample_coefficients(coef_key, data, proposed, prior)
    proposed = proposed._replace(coef=coef)
    sigma = _sample_sigma(sigma_key, data, proposed, prior)
    proposed = proposed._replace(sigma=sigma)
    transition = _sample_transition(trans_key, proposed, prior)
    proposed = _order_regimes(proposed._replace(transition=transition))
    regimes = proposed.sigma.shape[0]
    occupancy = jnp.mean(jax.nn.one_hot(proposed.modes, regimes, dtype=proposed.coef.dtype), axis=0)
    draw = RegimeDraw(
        coef=proposed.coef,
        sigma=proposed.sigma,
        transition=proposed.transition,
        occupancy=occupancy,
        accepted=accepted,
        loglik=complete_loglik(data.series, data.features, proposed),
    )
    return proposed, draw


def _scan_chain(
    key: PRNGKeyArray,
    state: RegimeState,
    data: RegimeData,
    prior: RegimePrior,
    sweeps: int,
) -> RegimeDraw:
    keys = random.split(key, sweeps)

    def step(current: RegimeState, draw_key: PRNGKeyArray) -> tuple[RegimeState, RegimeDraw]:
        return regime_sweep(draw_key, current, data, prior)

    _final, draws = jax.lax.scan(step, state, keys)
    return draws


def _sample_chains(
    key: PRNGKeyArray,
    data: RegimeData,
    regimes: int,
    chains: int,
    sweeps: int,
) -> RegimeDraw:
    def one_chain(chain_key: PRNGKeyArray) -> RegimeDraw:
        init_key, sample_key = random.split(chain_key)
        state, prior = init_state(init_key, data, regimes)
        return _scan_chain(sample_key, state, data, prior, sweeps)

    return jax.vmap(one_chain)(random.split(key, chains))


@partial(jax.jit, static_argnames=("regimes", "chains", "sweeps"))
def sample_regime_har(
    key: PRNGKeyArray,
    data: RegimeData,
    *,
    regimes: int = 3,
    chains: int = 4,
    sweeps: int = 2000,
) -> RegimeDraw:
    """Run ``chains`` independent Gibbs chains with ``lax.scan`` and ``vmap``.

    Args:
        key: Master key. Each chain receives one split.
        data: One training window.
        regimes: Number of sticky regimes.
        chains: Number of chains. Static.
        sweeps: Sweeps per chain, including burn-in. Static.

    Returns:
        Draws stacked on a leading chain axis. Each field has shape
        ``(chains, sweeps, ...)``.
    """

    return _sample_chains(key, data, regimes, chains, sweeps)


@partial(jax.jit, static_argnames=("regimes", "chains", "sweeps"))
def sample_regime_windows(
    key: PRNGKeyArray,
    series: Float[Array, "R T"],
    features: Float[Array, "R T F"],
    regimes: int,
    chains: int,
    sweeps: int,
) -> RegimeDraw:
    """Fit every rolling window in one compiled call.

    Args:
        key: Master key, split across windows and then chains.
        series: Batch of equal-length training series.
        features: HAR features for those series.
        regimes: Number of regimes.
        chains: Chains per window.
        sweeps: Sweeps per chain.

    Returns:
        Draws with a leading window axis, then chain and sweep.
    """

    def one_window(
        window_key: PRNGKeyArray,
        window_series: Float[Array, "T"],
        window_features: Float[Array, "T F"],
    ) -> RegimeDraw:
        data = RegimeData(series=window_series, features=window_features)
        return _sample_chains(window_key, data, regimes, chains, sweeps)

    window_keys = random.split(key, series.shape[0])
    return jax.vmap(one_window)(window_keys, series, features)


def fit_regime_har(
    series: np.ndarray,
    features: np.ndarray,
    *,
    regimes: int = 3,
    sweeps: int = 2000,
    chains: int = 4,
    seed: int = 23,
) -> dict[str, np.ndarray]:
    """Run the sampler and return second-half draws as NumPy arrays.

    Args:
        series: Daily log variance on one training window.
        features: HAR features aligned with ``series``. Rows that are not
            finite are dropped.
        regimes: Number of regimes.
        sweeps: Sweeps per chain, including the first-half burn-in.
        chains: Number of chains.
        seed: Seed for the JAX key.

    Returns:
        Second-half draws, complete-data log likelihood, occupancy, the
        fraction of coefficient proposals that were accepted, and split-free
        R-hat values for intercepts, slopes, scales, and self-stays.
    """

    usable = np.isfinite(features).all(axis=1) & np.isfinite(series)
    observed = jnp.asarray(series[usable], dtype=jnp.float64)
    design = jnp.asarray(features[usable], dtype=jnp.float64)
    data = RegimeData(series=observed, features=design)
    draws = sample_regime_har(
        random.PRNGKey(seed),
        data,
        regimes=regimes,
        chains=chains,
        sweeps=sweeps,
    )
    packed = jax.tree_util.tree_map(np.asarray, draws)
    return pack_regime_draws(packed, sweeps=sweeps, regimes=regimes)


def pack_regime_draws(draws: RegimeDraw, *, sweeps: int, regimes: int) -> dict[str, np.ndarray]:
    """Keep the second half of each chain and compute R-hat.

    Args:
        draws: Chain-major draws from :func:`sample_regime_har`.
        sweeps: Sweeps that were run, including burn-in.
        regimes: Number of regimes.

    Returns:
        Second-half coefficients, scales, transitions, log likelihood,
        occupancy, acceptance, and split-free R-hat.
    """

    kept = slice(sweeps // 2, None)
    coef = np.asarray(draws.coef[:, kept])
    sigma = np.asarray(draws.sigma[:, kept])
    transition = np.asarray(draws.transition[:, kept])
    accepted = np.asarray(draws.accepted)
    loglik = np.asarray(draws.loglik)
    flat_coef = coef.reshape(-1, coef.shape[-1])
    flat_sigma = sigma.reshape(-1, sigma.shape[-1])
    flat_trans = transition.reshape(-1, regimes, regimes)
    occupancy = np.asarray(draws.occupancy[:, kept])
    pieces = (
        coef[:, :, :regimes],
        coef[:, :, regimes:],
        sigma,
        np.diagonal(transition, axis1=-2, axis2=-1),
    )
    rhat = np.asarray(
        [_gelman_rubin(piece[:, :, dim]) for piece in pieces for dim in range(piece.shape[-1])],
        dtype=np.float64,
    )
    return {
        "coef": flat_coef,
        "sigma": flat_sigma,
        "trans": flat_trans,
        "c": flat_coef[:, :regimes],
        "beta": flat_coef[:, regimes:],
        "loglik": loglik,
        "occupancy": occupancy,
        "acceptance": np.asarray([float(np.mean(accepted))]),
        "rhat": rhat,
    }


def _gelman_rubin(chains: np.ndarray) -> float:
    """Split-free R-hat. ``chains`` has shape ``(chains, draws)``."""

    chain_count, length = chains.shape
    if chain_count < 2 or length < 2:
        return float("nan")
    chain_mean = chains.mean(axis=1)
    chain_var = chains.var(axis=1, ddof=1)
    between = float(length * chain_mean.var(ddof=1))
    within = float(chain_var.mean())
    if within <= 0.0:
        return 1.0 if between <= 0.0 else float("inf")
    var_hat = (length - 1) / float(length) * within + between / float(length)
    return float(np.sqrt(var_hat / within))
