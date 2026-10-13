"""Sticky HDP updates for the weak-limit truncation.

The transition rows, global weights, and concentrations follow Fox et al.,
arXiv:1003.3829, Section II-B and Algorithm 1 step 5. Table counts are the
Chinese-restaurant auxiliary variables of the sticky HDP-HMM: the n-th
customer at restaurant ``j`` eating dish ``k`` opens a table with probability
``(alpha beta_k + kappa 1_{j=k}) / (n + alpha beta_k + kappa 1_{j=k})``,
with ``n`` starting at 0. Override indicators are then removed from the
diagonal to form the informative counts used by the top-level Dirichlet.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from slds._dtype import enable_x64
from slds.types import SLDSPrior, SLDSState

enable_x64()


def count_transitions(modes: Int[Array, "N T"], mode_count: int) -> Float[Array, "K K"]:
    """Count transitions ``z_t -> z_{t+1}`` over every trace.

    Args:
        modes: Mode index at each stored time.
        mode_count: Truncation level. Modes that never occur stay at zero.

    Returns:
        Integer counts stored as floats, shape ``(modes, modes)``.
    """

    if modes.shape[1] < 2:
        return jnp.zeros((mode_count, mode_count), dtype=jnp.float64)
    previous = modes[:, :-1].reshape(-1)
    nxt = modes[:, 1:].reshape(-1)
    index = previous * mode_count + nxt
    counts = jnp.bincount(index, length=mode_count * mode_count)
    return counts.reshape(mode_count, mode_count).astype(jnp.float64)


def sample_table_counts(
    key: PRNGKeyArray,
    transitions: Float[Array, "K K"],
    beta: Float[Array, "K"],
    alpha: Float[Array, ""],
    kappa: Float[Array, ""],
    customers: int,
) -> Float[Array, "K K"]:
    """Draw the auxiliary table counts for every restaurant and dish.

    Args:
        key: JAX random key.
        transitions: Transition counts.
        beta: Global weights.
        alpha: HDP concentration, ``(1 - rho) * (alpha + kappa)``.
        kappa: Sticky mass, ``rho * (alpha + kappa)``.
        customers: Static upper bound on a cell count. Seats beyond the
            observed count are ignored. The bound used by the sweep is the
            number of transitions in the batch.

    Returns:
        Table counts ``m``, shape ``(modes, modes)``.
    """

    if customers <= 0:
        return jnp.zeros_like(transitions)
    seats = jnp.arange(customers, dtype=transitions.dtype)
    sticky = kappa * jnp.eye(beta.shape[0], dtype=beta.dtype)
    term = alpha * beta[None, :] + sticky
    denom = seats[None, None, :] + term[:, :, None]
    active = seats[None, None, :] < transitions[:, :, None]
    probability = jnp.where(active, term[:, :, None] / jnp.maximum(denom, 1e-30), 0.0)
    draws = random.bernoulli(key, jnp.clip(probability, 0.0, 1.0))
    return jnp.sum(draws, axis=-1).astype(transitions.dtype)


def sample_override_counts(
    key: PRNGKeyArray,
    tables: Float[Array, "K K"],
    beta: Float[Array, "K"],
    alpha: Float[Array, ""],
    kappa: Float[Array, ""],
) -> tuple[Float[Array, "K"], Float[Array, "K K"]]:
    """Draw sticky override indicators and the informative table counts.

    Args:
        key: JAX random key.
        tables: Table counts ``m``.
        beta: Global weights.
        alpha: HDP concentration.
        kappa: Sticky mass.

    Returns:
        Override counts and ``m_hat``, which is ``m`` with those overrides
        removed from the diagonal.
    """

    rho = kappa / jnp.maximum(alpha + kappa, 1e-30)
    probability = rho / jnp.maximum(rho + beta * (1.0 - rho), 1e-30)
    overrides = random.binomial(
        key,
        jnp.rint(jnp.diag(tables)).astype(jnp.int32),
        jnp.clip(probability, 0.0, 1.0),
    ).astype(tables.dtype)
    informative = tables - jnp.diag(overrides)
    return overrides, informative


def _gamma_scale(
    key: PRNGKeyArray, shape: Float[Array, ""], scale: Float[Array, ""]
) -> Float[Array, ""]:
    draw = random.gamma(key, jnp.maximum(shape, 1e-6))
    return draw * jnp.maximum(scale, 1e-12)


def sample_rho(
    key: PRNGKeyArray,
    tables: Float[Array, "K K"],
    overrides: Float[Array, "K"],
    prior: Float[Array, "2"],
) -> Float[Array, ""]:
    """Draw the sticky weight from its beta conditional.

    Args:
        key: JAX random key.
        tables: Table counts ``m``.
        overrides: Diagonal override counts.
        prior: Beta prior parameters ``(a, b)``.

    Returns:
        One draw of ``rho``.
    """

    sticky = jnp.sum(overrides)
    shape_a = sticky + prior[0]
    shape_b = jnp.sum(tables) - sticky + prior[1]
    return random.beta(key, jnp.maximum(shape_a, 1e-6), jnp.maximum(shape_b, 1e-6))


def sample_gamma(
    key: PRNGKeyArray,
    informative: Float[Array, "K K"],
    gamma: Float[Array, ""],
    prior: Float[Array, "2"],
) -> Float[Array, ""]:
    """Draw the top-level concentration.

    Args:
        key: JAX random key.
        informative: Table counts after the sticky overrides are removed.
        gamma: Current top-level concentration, used by the beta auxiliary.
        prior: Gamma prior, shape and rate.

    Returns:
        One draw of ``gamma``. The gamma prior is shape and rate, and the
        draw uses scale ``1 / rate``.
    """

    eta_key, mix_key, draw_key = random.split(key, 3)
    occupied = jnp.sum(jnp.sum(informative, axis=0) > 0)
    total = jnp.sum(informative)
    eta = random.beta(eta_key, gamma + 1.0, total + 1e-6)
    mixing_probability = total / jnp.maximum(total + gamma, 1e-30)
    mixing = random.bernoulli(mix_key, jnp.clip(mixing_probability, 0.0, 1.0)).astype(gamma.dtype)
    shape = prior[0] + occupied - mixing
    scale = 1.0 / jnp.maximum(prior[1] - jnp.log(jnp.maximum(eta, 1e-30)), 1e-8)
    return _gamma_scale(draw_key, shape, scale)


def sample_alpha_kappa(
    key: PRNGKeyArray,
    transitions: Float[Array, "K K"],
    tables: Float[Array, "K K"],
    alpha_kappa: Float[Array, ""],
    prior: Float[Array, "2"],
) -> Float[Array, ""]:
    """Draw ``alpha + kappa`` from its gamma conditional.

    Args:
        key: JAX random key.
        transitions: Customer counts.
        tables: Table counts ``m``.
        alpha_kappa: Current sum, used by the beta and Bernoulli auxiliaries.
        prior: Gamma prior, shape and rate.

    Returns:
        One draw. Empty truncation rows contribute a factor of one.
    """

    beta_key, bernoulli_key, draw_key = random.split(key, 3)
    customers = jnp.sum(transitions, axis=1)
    active = customers > 0
    auxiliary = random.beta(beta_key, alpha_kappa + 1.0, jnp.maximum(customers, 1.0))
    probability = customers / jnp.maximum(customers + alpha_kappa, 1e-30)
    indicators = random.bernoulli(bernoulli_key, jnp.clip(probability, 0.0, 1.0)).astype(
        alpha_kappa.dtype
    )
    summed_indicators = jnp.sum(jnp.where(active, indicators, 0.0))
    summed_log = jnp.sum(jnp.where(active, jnp.log(jnp.maximum(auxiliary, 1e-30)), 0.0))
    shape = prior[0] + jnp.sum(tables) - summed_indicators
    scale = 1.0 / jnp.maximum(prior[1] - summed_log, 1e-8)
    return _gamma_scale(draw_key, shape, scale)


def sample_transition_rows(
    key: PRNGKeyArray,
    transitions: Float[Array, "K K"],
    beta: Float[Array, "K"],
    alpha_kappa: Float[Array, ""],
    rho: Float[Array, ""],
) -> Float[Array, "K K"]:
    """Draw each transition row from its Dirichlet conditional.

    Args:
        key: JAX random key.
        transitions: Customer counts.
        beta: Global weights.
        alpha_kappa: Sum of the HDP concentration and the sticky mass.
        rho: Sticky weight.

    Returns:
        Row-stochastic transition matrix.
    """

    kappa = rho * alpha_kappa
    alpha = (1.0 - rho) * alpha_kappa
    concentration = (
        alpha * beta[None, :] + transitions + kappa * jnp.eye(beta.shape[0], dtype=beta.dtype)
    )
    return random.dirichlet(key, jnp.maximum(concentration, 1e-8))


def sample_beta(
    key: PRNGKeyArray,
    informative: Float[Array, "K K"],
    gamma: Float[Array, ""],
) -> Float[Array, "K"]:
    """Draw the global weights from their Dirichlet conditional.

    Args:
        key: JAX random key.
        informative: Table counts after sticky overrides are removed.
        gamma: Top-level concentration.

    Returns:
        Probability vector of length equal to the truncation.
    """

    mode_count = informative.shape[0]
    concentration = gamma / mode_count + jnp.sum(informative, axis=0)
    return random.dirichlet(key, jnp.maximum(concentration, 1e-8))


def update_hdp(
    key: PRNGKeyArray,
    state: SLDSState,
    prior: SLDSPrior,
) -> SLDSState:
    """One conjugate sticky-HDP block.

    Args:
        key: JAX random key, split across the six draws.
        state: State whose mode path supplies the transition counts.
        prior: Gamma and beta hyperparameters. ``alpha_kappa`` and ``gamma``
            are shape and rate.

    Returns:
        State with updated ``beta``, ``transition``, ``alpha_plus_kappa``,
        ``rho``, and ``gamma``. Latent paths and dynamics are unchanged.
    """

    table_key, alpha_key, rho_key, gamma_key, beta_key, pi_key = random.split(key, 6)
    transitions = count_transitions(state.modes, state.beta.shape[0])
    customers = state.modes.shape[0] * max(state.modes.shape[1] - 1, 0)
    kappa = state.rho * state.alpha_plus_kappa
    alpha = (1.0 - state.rho) * state.alpha_plus_kappa
    tables = sample_table_counts(table_key, transitions, state.beta, alpha, kappa, customers)
    overrides, informative = sample_override_counts(
        random.fold_in(table_key, 1), tables, state.beta, alpha, kappa
    )
    alpha_kappa = sample_alpha_kappa(
        alpha_key, transitions, tables, state.alpha_plus_kappa, prior.alpha_kappa
    )
    rho = sample_rho(rho_key, tables, overrides, prior.rho)
    gamma = sample_gamma(gamma_key, informative, state.gamma, prior.gamma)
    beta = sample_beta(beta_key, informative, gamma)
    transition = sample_transition_rows(pi_key, transitions, beta, alpha_kappa, rho)
    return state._replace(
        beta=beta,
        transition=transition,
        alpha_plus_kappa=alpha_kappa,
        rho=rho,
        gamma=gamma,
    )
