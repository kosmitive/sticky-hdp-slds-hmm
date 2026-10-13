"""Blocked Gibbs sweep for the sticky HDP-SLDS.

The default sweep is the weak-limit sampler of Fox et al., arXiv:1003.3829,
Section III-B. Modes are drawn jointly given the latent path (Section III-B3
and Appendix C-A). The latent path is then drawn by a backward information
filter and a forward sample (Section III-B4, Appendix C-B, Algorithms 5 and
6). Sticky HDP parameters, matrix-normal inverse-Wishart dynamics, and the
shared emission covariance follow Algorithm 1 steps 5–7 and Equation (31).

``sequential_modes=True`` inserts the marginalized single-site mode update of
Section III-B5 before the latent draw. That step is off unless the caller
asks for it. The pendulum example leaves it off.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from sticky_hdp_slds._dtype import enable_x64
from sticky_hdp_slds.inference.conjugate import (
    sample_gaussian,
    sample_invwishart,
    sample_matrix_normal,
)
from sticky_hdp_slds.inference.hdp import update_hdp
from sticky_hdp_slds.inference.lgssm import (
    backward_information,
    forward_information,
    sample_latents,
)
from sticky_hdp_slds.inference.types import SLDSData, SLDSPrior, SLDSState

enable_x64()


def gaussian_logpdf(
    points: Float[Array, "... D"],
    mean: Float[Array, "... D"],
    covariance: Float[Array, "D D"],
) -> Float[Array, "..."]:
    """Log density of a Gaussian with one shared covariance.

    Args:
        points: Observations.
        mean: Means broadcastable to ``points``.
        covariance: Positive definite covariance.

    Returns:
        Log density with the trailing dimension removed.
    """

    diff = points - mean
    dim = diff.shape[-1]
    flat = diff.reshape(-1, dim)
    _sign, logdet = jnp.linalg.slogdet(covariance)
    solved = jnp.linalg.solve(covariance, flat.T).T
    quad = jnp.sum(flat * solved, axis=-1)
    log_density = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + quad)
    return log_density.reshape(diff.shape[:-1])


def log_transition_likelihood(
    latents: Float[Array, "N T D"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
) -> Float[Array, "N T K"]:
    """Log transition density of every trace, time, and mode.

    Args:
        latents: Latent path. Time 0 has no stored parent, so its column is zero.
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.

    Returns:
        Array of shape ``(traces, steps, modes)``.
    """

    traces, steps, _dim = latents.shape
    mode_count = dynamics.shape[0]
    if steps < 2:
        return jnp.zeros((traces, steps, mode_count), dtype=latents.dtype)
    parents = latents[:, :-1]
    children = latents[:, 1:]

    def one_mode(
        matrix: Float[Array, "D D"], covariance: Float[Array, "D D"]
    ) -> Float[Array, "N Tm1"]:
        return gaussian_logpdf(children, parents @ matrix.T, covariance)

    rest = jax.vmap(one_mode)(dynamics, process_cov)
    rest = jnp.moveaxis(rest, 0, -1)
    zeros = jnp.zeros((traces, 1, mode_count), dtype=latents.dtype)
    return jnp.concatenate([zeros, rest], axis=1)


def log_backward_messages(
    log_lik: Float[Array, "N T K"], transition: Float[Array, "K K"]
) -> Float[Array, "N T K"]:
    """Normalized backward mode messages.

    ``log m_t(k) = logsumexp_j(log pi_kj + log lik(x_{t+1}|x_t, j) + log m_{t+1}(j))``.
    The message at the last time is one. The likelihood at the current time
    is applied when the mode is sampled, not inside the message.

    Args:
        log_lik: Output of :func:`log_transition_likelihood`.
        transition: Mode transition matrix.

    Returns:
        Normalized log messages of the same shape.
    """

    traces, steps, modes = log_lik.shape
    if steps < 2:
        return jnp.zeros_like(log_lik)
    log_pi = jnp.log(jnp.maximum(transition, 1e-300))
    initial = jnp.zeros((traces, modes), dtype=log_lik.dtype)

    def step(
        message_next: Float[Array, "N K"], likelihood_next: Float[Array, "N K"]
    ) -> tuple[Float[Array, "N K"], Float[Array, "N K"]]:
        terms = log_pi[None, :, :] + likelihood_next[:, None, :] + message_next[:, None, :]
        message = jax.scipy.special.logsumexp(terms, axis=2)
        message = message - jax.scipy.special.logsumexp(message, axis=1, keepdims=True)
        return message, message

    likelihood = jnp.swapaxes(log_lik[:, 1:], 0, 1)
    _final, earlier = lax.scan(step, initial, likelihood, reverse=True)
    terminal = jnp.zeros((traces, 1, modes), dtype=log_lik.dtype)
    return jnp.concatenate([jnp.swapaxes(earlier, 0, 1), terminal], axis=1)


def _categorical(key: PRNGKeyArray, log_scores: Float[Array, "... K"]) -> Int[Array, "..."]:
    finite = jnp.isfinite(log_scores)
    safe = jnp.where(finite, log_scores, -jnp.inf)
    peak = jnp.max(jnp.where(finite, safe, -1.0e300), axis=-1, keepdims=True)
    weights = jnp.where(finite, jnp.exp(safe - peak), 0.0)
    total = jnp.sum(weights, axis=-1, keepdims=True)
    uniform = jnp.ones_like(weights) / weights.shape[-1]
    probability = jnp.where(total > 0.0, weights / jnp.maximum(total, 1e-300), uniform)
    draw = random.categorical(key, jnp.log(jnp.maximum(probability, 1e-300)))
    return draw.astype(jnp.int32)


def sample_modes(
    key: PRNGKeyArray,
    log_lik: Float[Array, "N T K"],
    log_messages: Float[Array, "N T K"],
    transition: Float[Array, "K K"],
    beta: Float[Array, "K"],
) -> Int[Array, "N T"]:
    """Draw a mode path from the backward messages.

    Time 0 is scored under ``beta``. Later times use the row of ``pi``
    selected by the newly drawn previous mode.

    Args:
        key: JAX random key.
        log_lik: Transition log densities.
        log_messages: Normalized backward log messages.
        transition: Mode transition matrix.
        beta: Global weights.

    Returns:
        Mode index, shape ``(traces, steps)``.
    """

    steps = log_lik.shape[1]
    log_pi = jnp.log(jnp.maximum(transition, 1e-300))
    log_beta = jnp.log(jnp.maximum(beta, 1e-300))
    keys = random.split(key, steps)

    def step(
        previous: Int[Array, "N"],
        inputs: tuple[Int[Array, ""], Float[Array, "N K"], Float[Array, "N K"], PRNGKeyArray],
    ) -> tuple[Int[Array, "N"], Int[Array, "N"]]:
        time, likelihood, message, draw_key = inputs
        from_row = log_pi[previous]
        base = jnp.where(time > 0, from_row, log_beta)
        mode = _categorical(draw_key, base + likelihood + message)
        return mode, mode

    times = jnp.arange(steps)
    initial = jnp.zeros(log_lik.shape[0], dtype=jnp.int32)
    _final, modes = lax.scan(
        step,
        initial,
        (
            times,
            jnp.swapaxes(log_lik, 0, 1),
            jnp.swapaxes(log_messages, 0, 1),
            keys,
        ),
    )
    return jnp.swapaxes(modes, 0, 1).astype(jnp.int32)


def sample_latent_paths(
    key: PRNGKeyArray,
    modes: Int[Array, "N T"],
    observations: Float[Array, "N T O"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    emission: Float[Array, "O D"],
    emission_cov: Float[Array, "O O"],
    latent_prior: Float[Array, "D D"],
) -> Float[Array, "N T D"]:
    """Block-sample every latent path given the modes.

    Args:
        key: JAX random key, split across traces.
        modes: Mode sequence.
        observations: Observations aligned with ``modes``.
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.
        emission: Observation matrix.
        emission_cov: Shared emission covariance.
        latent_prior: Covariance of the unstored parent state.

    Returns:
        Latent path. The parent draw is discarded.
    """

    def one_trace(
        trace_key: PRNGKeyArray,
        trace_modes: Int[Array, "T"],
        trace_obs: Float[Array, "T O"],
    ) -> Float[Array, "T D"]:
        mats, vecs = backward_information(
            trace_modes, trace_obs, dynamics, process_cov, emission, emission_cov
        )
        return sample_latents(
            trace_key, trace_modes, mats, vecs, dynamics, process_cov, latent_prior
        )

    keys = random.split(key, modes.shape[0])
    return jax.vmap(one_trace)(keys, modes, observations)


def regression_statistics(
    modes: Int[Array, "N T"], latents: Float[Array, "N T D"], mode_count: int
) -> tuple[Float[Array, "K D D"], Float[Array, "K D D"], Float[Array, "K D D"], Float[Array, "K"]]:
    """Parent and child Gram matrices for each mode.

    Pairs exist only for ``t >= 1``. Matrix ``k`` matches
    ``split_regression_pairs``: parents times parents transposed, children
    times parents transposed, and children times children transposed.

    Args:
        modes: Mode sequence.
        latents: Latent path.
        mode_count: Truncation level, including modes that never occur.

    Returns:
        ``S_xx``, ``S_yx``, ``S_yy``, and the pair count.
    """

    return _statistics_for_truncation(modes, latents, mode_count)


def _statistics_for_truncation(
    modes: Int[Array, "N T"], latents: Float[Array, "N T D"], mode_count: int
) -> tuple[Float[Array, "K D D"], Float[Array, "K D D"], Float[Array, "K D D"], Float[Array, "K"]]:
    if latents.shape[1] < 2:
        dim = latents.shape[-1]
        zeros = jnp.zeros((mode_count, dim, dim), dtype=latents.dtype)
        return zeros, zeros, zeros, jnp.zeros(mode_count, dtype=latents.dtype)
    parents = latents[:, :-1]
    children = latents[:, 1:]
    labels = modes[:, 1:]

    def one_mode(
        mode: Int[Array, ""],
    ) -> tuple[Float[Array, "D D"], Float[Array, "D D"], Float[Array, "D D"], Float[Array, ""]]:
        mask = (labels == mode).astype(latents.dtype)
        weighted_parent = parents * mask[..., None]
        weighted_child = children * mask[..., None]
        sxx = jnp.einsum("ntd,nte->de", weighted_parent, parents)
        syx = jnp.einsum("ntd,nte->de", weighted_child, parents)
        syy = jnp.einsum("ntd,nte->de", weighted_child, children)
        return sxx, syx, syy, jnp.sum(mask)

    return jax.vmap(one_mode)(jnp.arange(mode_count))


def sample_mniw_dynamics(
    key: PRNGKeyArray,
    modes: Int[Array, "N T"],
    latents: Float[Array, "N T D"],
    prior: SLDSPrior,
    mode_count: int,
) -> tuple[Float[Array, "K D D"], Float[Array, "K D D"]]:
    """Draw ``(A, Sigma)`` from the matrix-normal inverse-Wishart posterior.

    Args:
        key: JAX random key, split across modes.
        modes: Mode sequence. The child mode owns the pair.
        latents: Latent path.
        prior: Shared matrix-normal mean and column precision, and the
            inverse-Wishart degrees of freedom and scale.
        mode_count: Truncation level. This is not the latent dimension.

    Returns:
        Dynamic matrices and process covariances.
    """

    return _sample_mniw(key, modes, latents, prior, mode_count)


def _sample_mniw(
    key: PRNGKeyArray,
    modes: Int[Array, "N T"],
    latents: Float[Array, "N T D"],
    prior: SLDSPrior,
    mode_count: int,
) -> tuple[Float[Array, "K D D"], Float[Array, "K D D"]]:
    sxx, syx, syy, counts = _statistics_for_truncation(modes, latents, mode_count)
    mean = prior.dynamics_mean
    column_precision = prior.dynamics_column_precision
    mk = mean @ column_precision
    mkm = mk @ mean.T
    keys = random.split(key, mode_count)

    def one_mode(
        draw_key: PRNGKeyArray,
        raw_xx: Float[Array, "D D"],
        raw_yx: Float[Array, "D D"],
        raw_yy: Float[Array, "D D"],
        count: Float[Array, ""],
    ) -> tuple[Float[Array, "D D"], Float[Array, "D D"]]:
        cov_key, mean_key = random.split(draw_key)
        s_hh = raw_xx + column_precision
        s_oh = raw_yx + mk
        s_oo = raw_yy + mkm
        new_mean = s_oh @ jnp.linalg.inv(s_hh)
        scale = s_oo - new_mean @ s_oh.T
        scale = 0.5 * (scale + scale.T)
        covariance = sample_invwishart(
            cov_key, prior.dynamics_df + count, prior.dynamics_scale + scale
        )
        matrix = sample_matrix_normal(mean_key, new_mean, covariance, jnp.linalg.inv(s_hh))
        return matrix, covariance

    return jax.vmap(one_mode)(keys, sxx, syx, syy, counts)


def ard_normal_equations(
    parents: Float[Array, "D N"],
    children: Float[Array, "D N"],
    noise_precision: Float[Array, "D D"],
    column_precision: Float[Array, "D"],
) -> tuple[Float[Array, "P"], Float[Array, "P P"]]:
    """Column-major normal equations for one ARD dynamic matrix.

    Args:
        parents: Parent states, shape ``(latent, count)``.
        children: Child states, shape ``(latent, count)``.
        noise_precision: Inverse process covariance.
        column_precision: ARD precision of each column of ``A``.

    Returns:
        Information vector and precision of ``vec(A)`` in Fortran order.
        These match ``slds.slds_ard_prior_sample._ard_normal_equations``.
    """

    dim = parents.shape[0]
    gram = parents @ parents.T
    precision = jnp.kron(gram, noise_precision) + jnp.diag(jnp.repeat(column_precision, dim))
    cross = noise_precision @ (children @ parents.T)
    # Column-major vectorization. JAX reshape is row-major, so the transpose
    # of a C-order flatten is the Fortran flatten of the original matrix.
    return cross.T.reshape(-1), precision


def sample_ard_dynamics(
    key: PRNGKeyArray,
    modes: Int[Array, "N T"],
    latents: Float[Array, "N T D"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    column_precision: Float[Array, "K D"],
    prior: SLDSPrior,
) -> tuple[Float[Array, "K D D"], Float[Array, "K D D"], Float[Array, "K D"]]:
    """One ARD round: dynamic matrix, column precision, then process covariance.

    Args:
        key: JAX random key.
        modes: Mode sequence.
        latents: Latent path.
        dynamics: Current dynamic matrices. Only their leading dimension, the
            truncation, is used. The new matrix does not depend on the old one.
        process_cov: Current process covariances. The matrix draw conditions
            on this value, then the covariance is redrawn.
        column_precision: Current ARD column precisions.
        prior: Inverse-Wishart scale and the gamma prior on the precisions.

    Returns:
        Updated matrices, covariances, and column precisions.
    """

    mode_count = dynamics.shape[0]
    sxx, syx, syy, counts = _statistics_for_truncation(modes, latents, mode_count)
    keys = random.split(key, mode_count)
    return _sample_ard(keys, sxx, syx, syy, counts, process_cov, column_precision, prior)


def _sample_ard(
    keys: Array,
    sxx: Float[Array, "K D D"],
    syx: Float[Array, "K D D"],
    syy: Float[Array, "K D D"],
    counts: Float[Array, "K"],
    process_cov: Float[Array, "K D D"],
    column_precision: Float[Array, "K D"],
    prior: SLDSPrior,
) -> tuple[Float[Array, "K D D"], Float[Array, "K D D"], Float[Array, "K D"]]:
    dim = process_cov.shape[-1]

    def one_mode(draw_key, raw_xx, raw_yx, raw_yy, count, covariance, precision):
        matrix_key, precision_key, cov_key = random.split(draw_key, 3)
        noise_precision = jnp.linalg.inv(0.5 * (covariance + covariance.T))
        info = (noise_precision @ raw_yx).T.reshape(-1)
        normal = jnp.kron(raw_xx, noise_precision) + jnp.diag(jnp.repeat(precision, dim))
        normal = 0.5 * (normal + normal.T) + 1e-8 * jnp.eye(dim * dim)
        cov = jnp.linalg.inv(normal)
        cov = 0.5 * (cov + cov.T)
        matrix = sample_gaussian(matrix_key, cov @ info, cov).reshape(dim, dim).T
        rate = prior.ard[1] + 0.5 * jnp.sum(matrix**2, axis=0)
        precision = random.gamma(precision_key, prior.ard[0] + dim / 2.0) / jnp.maximum(rate, 1e-8)
        residual = raw_yy - raw_yx @ matrix.T - matrix @ raw_yx.T + matrix @ raw_xx @ matrix.T
        residual = 0.5 * (residual + residual.T)
        covariance = sample_invwishart(
            cov_key, prior.dynamics_df + count, prior.dynamics_scale + residual
        )
        return matrix, covariance, precision

    return jax.vmap(one_mode)(keys, sxx, syx, syy, counts, process_cov, column_precision)


def sample_emission(
    key: PRNGKeyArray,
    observations: Float[Array, "N T O"],
    latents: Float[Array, "N T D"],
    prior: SLDSPrior,
) -> Float[Array, "O O"]:
    """Draw the shared emission covariance.

    Fox et al., Equation (31), give ``IW(T + r0, R0 + S_R)`` for one series.
    Several traces contribute ``traces * steps`` observations, the product
    of those two sizes.

    Args:
        key: JAX random key.
        observations: Observation batch.
        latents: Latent path.
        prior: Inverse-Wishart degrees of freedom, scale, and the emission
            matrix ``C``.

    Returns:
        Emission covariance.
    """

    mean = jnp.einsum("ntd,od->nto", latents, prior.emission)
    residual = observations - mean
    scale = prior.emission_scale + jnp.einsum("nto,ntp->op", residual, residual)
    scale = 0.5 * (scale + scale.T)
    degrees = prior.emission_df + observations.shape[0] * observations.shape[1]
    return sample_invwishart(key, degrees, scale)


def sequential_mode_scores(
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    forward_mat: Float[Array, "N D D"],
    forward_vec: Float[Array, "N D"],
    backward_mat: Float[Array, "N D D"],
    backward_vec: Float[Array, "N D"],
) -> Float[Array, "N K"]:
    """Marginal log score of each candidate mode at one time.

    Both ``A`` and ``Sigma`` belong to the candidate mode. The Kalman
    messages are the ones computed from the mode path that entered the
    step. They are not refreshed inside the time loop.

    Args:
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.
        forward_mat: Forward information about the parent state.
        forward_vec: Forward information vector.
        backward_mat: Backward information about the child state.
        backward_vec: Backward information vector.

    Returns:
        Log score of shape ``(traces, modes)``, before the transition prior.
    """

    inv_forward = jnp.linalg.inv(forward_mat)

    def one_mode(matrix: Float[Array, "D D"], covariance: Float[Array, "D D"]) -> Float[Array, "N"]:
        coupled = jnp.einsum("ij,njk->nik", matrix, inv_forward)
        predicted = jnp.einsum("nij,kj->nik", coupled, matrix)
        inv_precision = covariance + predicted
        precision = jnp.linalg.inv(inv_precision)
        info = jnp.einsum("nij,njk,nk->ni", precision, coupled, forward_vec)
        summed = precision + backward_mat
        inv_summed = jnp.linalg.inv(summed)
        summed_info = info + backward_vec
        sign_p, logdet_p = jnp.linalg.slogdet(precision)
        sign_s, logdet_s = jnp.linalg.slogdet(summed)
        quad = -0.5 * jnp.einsum("ni,nij,nj->n", info, inv_precision, info)
        quad = quad + 0.5 * jnp.einsum("ni,nij,nj->n", summed_info, inv_summed, summed_info)
        score = 0.5 * (logdet_p - logdet_s) + quad
        return jnp.where((sign_p > 0) & (sign_s > 0), score, -jnp.inf)

    return jnp.swapaxes(jax.vmap(one_mode)(dynamics, process_cov), 0, 1)


def sequential_sample_modes(
    key: PRNGKeyArray,
    modes: Int[Array, "N T"],
    dynamics: Float[Array, "K D D"],
    process_cov: Float[Array, "K D D"],
    transition: Float[Array, "K K"],
    beta: Float[Array, "K"],
    forward_mats: Float[Array, "N T1 D D"],
    forward_vecs: Float[Array, "N T1 D"],
    backward_mats: Float[Array, "N T1 D D"],
    backward_vecs: Float[Array, "N T1 D"],
) -> Int[Array, "N T"]:
    """Single-site mode sample, scanning backward through time.

    Args:
        key: JAX random key.
        modes: Mode path that defined the Kalman messages. Earlier times keep
            these labels until the scan reaches them.
        dynamics: Mode transition matrices.
        process_cov: Mode process covariances.
        transition: Mode transition matrix.
        beta: Global weights. Time 0 is scored under ``beta``.
        forward_mats: Forward information, length ``steps + 1``.
        forward_vecs: Forward information vectors.
        backward_mats: Backward information, length ``steps + 1``.
        backward_vecs: Backward information vectors.

    Returns:
        Updated mode path. This is the optional Section III-B5 step.
    """

    _traces, steps = modes.shape
    log_pi = jnp.log(jnp.maximum(transition, 1e-12))
    log_beta = jnp.log(jnp.maximum(beta, 1e-12))
    keys = random.split(key, steps)
    previous = jnp.concatenate(
        [jnp.zeros((1, modes.shape[0]), dtype=jnp.int32), jnp.swapaxes(modes[:, :-1], 0, 1)],
        axis=0,
    )

    def step(next_mode, inputs):
        time, prev_mode, draw_key, f_mat, f_vec, b_mat, b_vec = inputs
        scores = sequential_mode_scores(dynamics, process_cov, f_mat, f_vec, b_mat, b_vec)
        to_next = jnp.swapaxes(log_pi[:, next_mode], 0, 1)
        from_prev = log_pi[prev_mode]
        scores = scores + jnp.where(time < steps - 1, to_next, 0.0)
        scores = scores + jnp.where(time > 0, from_prev, log_beta)
        mode = _categorical(draw_key, scores)
        return mode, mode

    times = jnp.arange(steps)
    _final, sampled = lax.scan(
        step,
        jnp.zeros(modes.shape[0], dtype=jnp.int32),
        (
            times,
            previous,
            keys,
            jnp.swapaxes(forward_mats[:, :steps], 0, 1),
            jnp.swapaxes(forward_vecs[:, :steps], 0, 1),
            jnp.swapaxes(backward_mats[:, 1 : steps + 1], 0, 1),
            jnp.swapaxes(backward_vecs[:, 1 : steps + 1], 0, 1),
        ),
        reverse=True,
    )
    return jnp.swapaxes(sampled, 0, 1).astype(jnp.int32)


def _filter_batch(modes, observations, state, prior, forward: bool):
    def one(trace_modes, trace_obs):
        if forward:
            return forward_information(
                trace_modes,
                trace_obs,
                state.dynamics,
                state.process_cov,
                prior.emission,
                state.emission_cov,
                prior.latent_prior,
            )
        return backward_information(
            trace_modes,
            trace_obs,
            state.dynamics,
            state.process_cov,
            prior.emission,
            state.emission_cov,
        )

    return jax.vmap(one)(modes, observations)


def slds_sweep(
    key: PRNGKeyArray,
    state: SLDSState,
    data: SLDSData,
    prior: SLDSPrior,
    *,
    sequential_modes: bool = False,
    ard: bool = False,
) -> SLDSState:
    """One Gibbs sweep. The sequential mode update is off by default.

    Args:
        key: JAX random key.
        state: Latent path, mode path, and parameters.
        data: Observations.
        prior: Fixed hyperparameters.
        sequential_modes: When true, run the Section III-B5 single-site mode
            update after the blocked mode draw and before the latent draw.
            The default sweep does not.
        ard: When true, resample dynamics with the ARD conditional instead of
            the matrix-normal inverse-Wishart conditional.

    Returns:
        The next state. Every block is a pure function of this state.
    """

    mode_key, seq_key, latent_key, hdp_key, dyn_key, emission_key = random.split(key, 6)
    log_lik = log_transition_likelihood(state.latents, state.dynamics, state.process_cov)
    messages = log_backward_messages(log_lik, state.transition)
    modes = sample_modes(mode_key, log_lik, messages, state.transition, state.beta)
    if sequential_modes:
        forward_mats, forward_vecs = _filter_batch(modes, data.observations, state, prior, True)
        backward_mats, backward_vecs = _filter_batch(modes, data.observations, state, prior, False)
        modes = sequential_sample_modes(
            seq_key,
            modes,
            state.dynamics,
            state.process_cov,
            state.transition,
            state.beta,
            forward_mats,
            forward_vecs,
            backward_mats,
            backward_vecs,
        )
    latents = sample_latent_paths(
        latent_key,
        modes,
        data.observations,
        state.dynamics,
        state.process_cov,
        prior.emission,
        state.emission_cov,
        prior.latent_prior,
    )
    updated = state._replace(modes=modes, latents=latents)
    updated = update_hdp(hdp_key, updated, prior)
    if ard:
        dynamics, process_cov, column_precision = sample_ard_dynamics(
            dyn_key,
            updated.modes,
            updated.latents,
            updated.dynamics,
            updated.process_cov,
            updated.column_precision,
            prior,
        )
        updated = updated._replace(
            dynamics=dynamics, process_cov=process_cov, column_precision=column_precision
        )
    else:
        dynamics, process_cov = sample_mniw_dynamics(
            dyn_key, updated.modes, updated.latents, prior, state.dynamics.shape[0]
        )
        updated = updated._replace(dynamics=dynamics, process_cov=process_cov)
    emission_cov = sample_emission(emission_key, data.observations, updated.latents, prior)
    return updated._replace(emission_cov=emission_cov)


def log_joint(state: SLDSState, data: SLDSData, emission: Float[Array, "O D"]) -> Float[Array, ""]:
    """Complete-data log joint given the fixed emission matrix.

    Args:
        state: Conditioned state.
        data: Observations.
        emission: Observation matrix ``C``.

    Returns:
        Scalar log joint used by the pendulum diagnostic.
    """

    mean = jnp.einsum("ntd,od->nto", state.latents, emission)
    emit = jnp.sum(gaussian_logpdf(data.observations, mean, state.emission_cov))
    safe = jnp.clip(state.transition, 1e-12, None)
    safe = safe / jnp.sum(safe, axis=1, keepdims=True)
    transitions = jnp.sum(jnp.log(safe[state.modes[:, :-1], state.modes[:, 1:]]))
    initial = jnp.sum(jnp.log(jnp.clip(state.beta[state.modes[:, 0]], 1e-12, None)))
    parents = state.latents[:, :-1]
    children = state.latents[:, 1:]
    labels = state.modes[:, 1:]

    def one_mode(matrix, covariance, mode):
        predicted = gaussian_logpdf(children, parents @ matrix.T, covariance)
        return jnp.sum(jnp.where(labels == mode, predicted, 0.0))

    latent = jnp.sum(
        jax.vmap(one_mode)(state.dynamics, state.process_cov, jnp.arange(state.dynamics.shape[0]))
    )
    return emit + transitions + initial + latent


@partial(jax.jit, static_argnames=("sweeps", "sequential_modes", "ard"))
def sample_slds(
    key: PRNGKeyArray,
    state: SLDSState,
    data: SLDSData,
    prior: SLDSPrior,
    sweeps: int,
    *,
    sequential_modes: bool = False,
    ard: bool = False,
) -> tuple[SLDSState, Float[Array, "sweeps"]]:
    """Run ``sweeps`` with ``lax.scan`` and return every state plus the log joint.

    Args:
        key: JAX random key.
        state: Initial state. It is not included in the returned trace.
        data: Observations.
        prior: Hyperparameters.
        sweeps: Number of sweeps. Static.
        sequential_modes: Forwarded to :func:`slds_sweep`. Static.
        ard: Forwarded to :func:`slds_sweep`. Static.

    Returns:
        Stacked states, one per sweep, and the complete-data log joint after
        each sweep.
    """

    keys = random.split(key, sweeps)

    def step(
        current: SLDSState, draw_key: PRNGKeyArray
    ) -> tuple[SLDSState, tuple[SLDSState, Float[Array, ""]]]:
        updated = slds_sweep(
            draw_key,
            current,
            data,
            prior,
            sequential_modes=sequential_modes,
            ard=ard,
        )
        return updated, (updated, log_joint(updated, data, prior.emission))

    _final, (trace, loglik) = lax.scan(step, state, keys)
    return trace, loglik
