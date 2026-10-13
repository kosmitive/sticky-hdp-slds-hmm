"""Information-form Kalman filter for the sticky HDP-SLDS.

The recursions follow Fox et al., arXiv:1003.3829, the forward information
filter in the supplement (the code cites Eq. 181 for the prior precision)
and the backward information filter used to block-sample the latent state.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from slds._dtype import enable_x64
from slds.conjugate import sample_gaussian

enable_x64()


def emission_information(
    emission: Float[Array, "O D"], emission_cov: Float[Array, "O O"]
) -> tuple[Float[Array, "D O"], Float[Array, "D D"]]:
    """Return ``C.T @ R^{-1}`` and ``C.T @ R^{-1} @ C``.

    Args:
        emission: Observation matrix.
        emission_cov: Shared emission covariance.

    Returns:
        Cross information and state information of one observation.
    """

    precision = jnp.linalg.inv(0.5 * (emission_cov + emission_cov.T))
    cross = emission.T @ precision
    return cross, cross @ emission


def forward_information(
    modes: Int[Array, "T"],
    observations: Float[Array, "T O"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    emission: Float[Array, "O D"],
    emission_cov: Float[Array, "O O"],
    latent_prior: Float[Array, "D D"],
) -> tuple[Float[Array, "T1 D D"], Float[Array, "T1 D"]]:
    """Forward information filter for one trace.

    Args:
        modes: Mode at each stored time.
        observations: Observations aligned with ``modes``.
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.
        emission: Observation matrix.
        emission_cov: Emission covariance.
        latent_prior: Covariance of the latent state before the first update.

    Returns:
        Information matrices and vectors of length ``steps + 1``.
    """

    cross, info = emission_information(emission, emission_cov)
    dim = dynamics.shape[-1]
    eye = jnp.eye(dim, dtype=dynamics.dtype)
    mats0 = jnp.linalg.inv(latent_prior)
    vecs0 = jnp.zeros(dim, dtype=dynamics.dtype)

    def step(carry, inputs):
        mats, vecs = carry
        mode, observation = inputs
        dyn = dynamics[mode]
        inv_dyn = jnp.linalg.inv(dyn)
        inv_var = jnp.linalg.inv(process_cov[mode])
        predicted = inv_dyn.T @ mats @ inv_dyn
        gain = predicted @ jnp.linalg.inv(predicted + inv_var)
        complement = eye - gain
        new_mats = complement @ predicted @ complement.T + gain @ inv_var @ gain.T + info
        new_vecs = complement @ inv_dyn.T @ vecs + cross @ observation
        return (new_mats, new_vecs), (new_mats, new_vecs)

    _final, (rest_mats, rest_vecs) = lax.scan(step, (mats0, vecs0), (modes, observations))
    return jnp.concatenate([mats0[None], rest_mats], axis=0), jnp.concatenate(
        [vecs0[None], rest_vecs], axis=0
    )


def backward_information(
    modes: Int[Array, "T"],
    observations: Float[Array, "T O"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    emission: Float[Array, "O D"],
    emission_cov: Float[Array, "O O"],
) -> tuple[Float[Array, "T1 D D"], Float[Array, "T1 D"]]:
    """Backward information filter for one trace.

    Args:
        modes: Mode at each stored time.
        observations: Observations aligned with ``modes``.
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.
        emission: Observation matrix.
        emission_cov: Emission covariance.

    Returns:
        Information matrices and vectors of length ``steps + 1``. Index 0 is
        the message about the latent parent that is not stored.
    """

    cross, info = emission_information(emission, emission_cov)
    dim = dynamics.shape[-1]
    eye = jnp.eye(dim, dtype=dynamics.dtype)
    steps = modes.shape[0]
    terminal_mats = info
    terminal_vecs = cross @ observations[-1]

    def step(carry, inputs):
        mats_next, vecs_next = carry
        mode, observation, time = inputs
        dyn = dynamics[mode]
        inv_var = jnp.linalg.inv(process_cov[mode])
        gain = mats_next @ jnp.linalg.inv(mats_next + inv_var)
        complement = eye - gain
        inner = complement @ mats_next @ complement.T + gain @ inv_var @ gain.T
        mats = dyn.T @ inner @ dyn
        vecs = dyn.T @ complement @ vecs_next
        mats = jnp.where(time > 0, mats + info, mats)
        vecs = jnp.where(time > 0, vecs + cross @ observation, vecs)
        return (mats, vecs), (mats, vecs)

    times = jnp.arange(steps - 1, -1, -1)
    # The observation added at time t > 0 is y[t-1]. At time 0 nothing is added.
    obs_for_time = jnp.concatenate([jnp.zeros((1, observations.shape[-1])), observations[:-1]], 0)
    obs_rev = obs_for_time[::-1]
    modes_rev = modes[::-1]
    _final, (rev_mats, rev_vecs) = lax.scan(
        step, (terminal_mats, terminal_vecs), (modes_rev, obs_rev, times)
    )
    return jnp.concatenate([rev_mats[::-1], terminal_mats[None]], 0), jnp.concatenate(
        [rev_vecs[::-1], terminal_vecs[None]], 0
    )


def sample_latents(
    key: PRNGKeyArray,
    modes: Int[Array, "T"],
    backward_mats: Float[Array, "T1 D D"],
    backward_vecs: Float[Array, "T1 D"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    latent_prior: Float[Array, "D D"],
) -> Float[Array, "T D"]:
    """Block-sample one latent path from the backward information filter.

    Args:
        key: JAX random key.
        modes: Mode sequence.
        backward_mats: Backward information matrices, length ``steps + 1``.
        backward_vecs: Backward information vectors.
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.
        latent_prior: Covariance of the unstored parent state.

    Returns:
        Latent path of length ``steps``. The parent draw is discarded.
    """

    prior_precision = jnp.linalg.inv(latent_prior)
    precision = prior_precision + backward_mats[0]
    covariance = jnp.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    parent_key, path_key = random.split(key)
    parent = sample_gaussian(parent_key, covariance @ backward_vecs[0], covariance)

    def step(previous, inputs):
        draw_key, mode, mats_next, vecs_next = inputs
        inv_var = jnp.linalg.inv(process_cov[mode])
        covariance_t = jnp.linalg.inv(inv_var + mats_next)
        covariance_t = 0.5 * (covariance_t + covariance_t.T)
        middle = inv_var @ dynamics[mode] @ previous + vecs_next
        draw = sample_gaussian(draw_key, covariance_t @ middle, covariance_t)
        return draw, draw

    keys = random.split(path_key, modes.shape[0])
    _final, path = lax.scan(step, parent, (keys, modes, backward_mats[1:], backward_vecs[1:]))
    return path
