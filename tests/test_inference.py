"""Fast checks for the JAX Gibbs blocks.

Filters, the one-step information update, conjugate moments, and the
Schur-Cohn test are deterministic. Table counts are compared to the
closed-form Chinese-restaurant expectation.
"""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jax import random
from jaxtyping import TypeCheckError, jaxtyped

from sticky_hdp_slds.inference.conjugate import sample_gaussian, sample_invwishart
from sticky_hdp_slds.inference.hdp import sample_table_counts
from sticky_hdp_slds.inference.hmm import forward_filter, stationary_distribution
from sticky_hdp_slds.inference.lgssm import backward_information, forward_information
from sticky_hdp_slds.inference.slds import ard_normal_equations, regression_statistics, slds_sweep
from sticky_hdp_slds.inference.stability import distributed_lags, har_stable, schur_stable
from sticky_hdp_slds.inference.types import SLDSData, SLDSPrior, SLDSState


def np_to_jax(value):
    """Convert a NumPy value to a float64 or int32 JAX array."""

    import jax.numpy as jnp

    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return jnp.asarray(array, dtype=jnp.int32)
    return jnp.asarray(array, dtype=jnp.float64)


def test_float64_is_enabled() -> None:
    import jax

    assert jax.config.jax_enable_x64


def test_schur_cohn_matches_companion_eigenvalues() -> None:
    """The reflection test agrees with the companion eigenvalues."""

    rng = np.random.default_rng(4)
    radius = 0.999
    for _draw in range(40):
        slopes = rng.normal(scale=0.35, size=3)
        spectral = _companion_radius(slopes)
        stable = bool(har_stable(np_to_jax(slopes), np_to_jax(radius)))
        if abs(spectral - radius) < 1e-6:
            continue
        assert stable == (spectral < radius)
    explosive = np.array([2.0, 0.0, 0.0])
    assert _companion_radius(explosive) > 1.0
    assert not bool(har_stable(np_to_jax(explosive), np_to_jax(radius)))
    row = distributed_lags(np_to_jax(np.array([0.2, 0.2, 0.2])))
    assert bool(schur_stable(row, np_to_jax(radius)))


def test_filter_rows_are_normalized() -> None:
    """Each filtered distribution sums to one."""

    rng = np.random.default_rng(5)
    emissions = np_to_jax(rng.normal(size=(12, 3)))
    stay = np.array([0.9, 0.8, 0.85])
    transition = np.full((3, 3), 0.0)
    off = (1.0 - stay) / 2.0
    transition[:] = off[:, None]
    np.fill_diagonal(transition, stay)
    initial = stationary_distribution(np_to_jax(transition))
    filtered = np.asarray(forward_filter(emissions, np_to_jax(transition), initial))
    assert filtered.shape == (12, 3)
    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(filtered >= -1e-12)


def _prefix_marginal(emission: np.ndarray, transition: np.ndarray, initial: np.ndarray, time: int):
    """Filtering marginal at ``time`` from every path that stops there."""

    prefixes = [(a, b, c)[: time + 1] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    prefixes = list(dict.fromkeys(prefixes))
    weight = []
    for path in prefixes:
        score = np.log(initial[path[0]]) + emission[0, path[0]]
        for step in range(1, time + 1):
            score += np.log(transition[path[step - 1], path[step]]) + emission[step, path[step]]
        weight.append(score)
    log_weight = np.array(weight) - np.max(weight)
    prob = np.exp(log_weight)
    prob /= prob.sum()
    marginal = np.zeros(2)
    for path, mass in zip(prefixes, prob, strict=True):
        marginal[path[time]] += mass
    return marginal


def _ffbs_path_probability(filtered: np.ndarray, transition: np.ndarray, path: tuple[int, ...]):
    """Joint probability of one path under the forward-filter backward conditional."""

    log_prob = np.log(filtered[-1, path[-1]])
    for time in range(len(path) - 2, -1, -1):
        weights = filtered[time] * transition[:, path[time + 1]]
        weights = weights / weights.sum()
        log_prob += np.log(weights[path[time]])
    return float(np.exp(log_prob))


def test_filter_matches_exact_enumeration() -> None:
    """Two-state filter matches prefix enumeration, and FFBS matches the joint."""

    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.0, -2.0], [-2.0, 0.0], [0.0, -1.0]])
    initial = np.array([0.6, 0.4])
    filtered = np.asarray(
        forward_filter(np_to_jax(emission), np_to_jax(transition), np_to_jax(initial))
    )
    for time in range(3):
        marginal = _prefix_marginal(emission, transition, initial, time)
        assert np.allclose(filtered[time], marginal, atol=1e-8)

    paths = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    joint = []
    for path in paths:
        score = np.log(initial[path[0]]) + emission[0, path[0]]
        score += np.log(transition[path[0], path[1]]) + emission[1, path[1]]
        score += np.log(transition[path[1], path[2]]) + emission[2, path[2]]
        joint.append(score)
    log_joint = np.array(joint) - np.max(joint)
    posterior = np.exp(log_joint)
    posterior /= posterior.sum()
    for path, mass in zip(paths, posterior, strict=True):
        got = _ffbs_path_probability(filtered, transition, path)
        assert got == pytest.approx(mass, abs=1e-8)


def test_backward_sample_matches_enumerated_path_on_a_forced_chain() -> None:
    """A point-mass filter locks the backward sample onto one path."""

    from sticky_hdp_slds import backward_sample

    transition = np_to_jax(np.array([[0.5, 0.5], [0.5, 0.5]]))
    emissions = np_to_jax(np.array([[0.0, -40.0], [0.0, -40.0], [-40.0, 0.0]]))
    initial = np_to_jax(np.array([0.5, 0.5]))
    filtered = forward_filter(emissions, transition, initial)
    modes = backward_sample(random.PRNGKey(0), filtered, transition)
    assert np.array_equal(np.asarray(modes), np.array([0, 0, 1]))
    assert np.allclose(np.asarray(filtered).sum(axis=1), 1.0)


def test_information_filter_matches_the_scalar_update() -> None:
    """One predict-update step matches the scalar information formula."""

    dynamics = np_to_jax(np.array([[[0.5]]]))
    process = np_to_jax(np.array([[[0.2]]]))
    emission = np_to_jax(np.array([[1.0]]))
    emission_cov = np_to_jax(np.array([[0.3]]))
    latent_prior = np_to_jax(np.array([[1.25]]))
    modes = np_to_jax(np.array([0, 0]))
    observations = np_to_jax(np.array([[0.4], [0.1]]))
    mats, vecs = forward_information(
        modes, observations, dynamics, process, emission, emission_cov, latent_prior
    )
    mats = np.asarray(mats)
    vecs = np.asarray(vecs)
    assert mats.shape == (3, 1, 1)
    prior = 1.0 / 1.25
    assert np.isclose(mats[0, 0, 0], prior)
    predicted = prior / 0.5**2
    gain = predicted / (predicted + 1.0 / 0.2)
    complement = 1.0 - gain
    info = 1.0 / 0.3
    updated = complement**2 * predicted + gain**2 / 0.2 + info
    assert np.isclose(mats[1, 0, 0], updated, atol=1e-8)
    back_mats, back_vecs = backward_information(
        modes, observations, dynamics, process, emission, emission_cov
    )
    assert np.asarray(back_mats).shape[0] == 3
    assert np.all(np.isfinite(np.asarray(back_vecs)))
    assert np.all(np.isfinite(vecs))


def test_regression_grams_match_the_outer_products() -> None:
    rng = np.random.default_rng(10)
    modes = rng.integers(0, 3, size=(2, 9))
    latents = rng.normal(size=(2, 9, 2))
    sxx, syx, syy, counts = regression_statistics(np_to_jax(modes), np_to_jax(latents), 3)
    sxx, syx, syy, counts = map(np.asarray, (sxx, syx, syy, counts))
    for mode in range(3):
        parents = []
        children = []
        for trace in range(2):
            for time in range(1, 9):
                if modes[trace, time] == mode:
                    parents.append(latents[trace, time - 1])
                    children.append(latents[trace, time])
        parent = np.stack(parents).T if parents else np.zeros((2, 0))
        child = np.stack(children).T if children else np.zeros((2, 0))
        assert counts[mode] == parent.shape[1]
        assert np.allclose(sxx[mode], parent @ parent.T, atol=1e-10)
        assert np.allclose(syx[mode], child @ parent.T, atol=1e-10)
        assert np.allclose(syy[mode], child @ child.T, atol=1e-10)


def test_ard_normal_equations_match_the_kronecker_form() -> None:
    rng = np.random.default_rng(11)
    parents = rng.normal(size=(2, 6))
    children = rng.normal(size=(2, 6))
    noise_precision = np.array([[2.0, 0.1], [0.1, 1.5]])
    column = np.array([0.4, 1.2])
    normal, precision = ard_normal_equations(
        np_to_jax(parents),
        np_to_jax(children),
        np_to_jax(noise_precision),
        np_to_jax(column),
    )
    gram = parents @ parents.T
    expected_precision = np.kron(gram, noise_precision) + np.diag(np.repeat(column, 2))
    cross = noise_precision @ (children @ parents.T)
    expected_normal = cross.T.reshape(-1)
    assert np.allclose(np.asarray(precision), expected_precision, atol=1e-10)
    assert np.allclose(np.asarray(normal), expected_normal, atol=1e-10)


def test_inverse_wishart_mean_matches_the_closed_form() -> None:
    import jax

    degrees = 8.0
    scale = np.array([[1.2, 0.1], [0.1, 0.7]])
    keys = random.split(random.PRNGKey(3), 4000)
    draws = jax.vmap(lambda key: sample_invwishart(key, np_to_jax(degrees), np_to_jax(scale)))(keys)
    mean = np.asarray(draws).mean(axis=0)
    expected = scale / (degrees - 2 - 1)
    assert np.allclose(mean, expected, atol=0.05)


def test_gaussian_draw_matches_the_location_scale() -> None:
    import jax

    mean = np.array([0.4, -0.2])
    covariance = np.array([[0.5, 0.1], [0.1, 0.3]])
    keys = random.split(random.PRNGKey(4), 8000)
    draws = jax.vmap(lambda key: sample_gaussian(key, np_to_jax(mean), np_to_jax(covariance)))(keys)
    sample = np.asarray(draws)
    assert np.allclose(sample.mean(axis=0), mean, atol=0.02)
    assert np.allclose(np.cov(sample.T), covariance, atol=0.03)


def test_table_count_expectation() -> None:
    transitions = np.array([[4.0, 2.0], [0.0, 3.0]])
    beta = np.array([0.7, 0.3])
    alpha, kappa = 1.5, 6.0
    keys = random.split(random.PRNGKey(12), 400)
    import jax

    draws = jax.vmap(
        lambda key: sample_table_counts(
            key, np_to_jax(transitions), np_to_jax(beta), np_to_jax(alpha), np_to_jax(kappa), 4
        )
    )(keys)
    mean = np.asarray(draws).mean(axis=0)
    expected = np.zeros_like(transitions)
    for row in range(2):
        for col in range(2):
            term = alpha * beta[col] + (kappa if row == col else 0.0)
            count = int(transitions[row, col])
            expected[row, col] = sum(term / (seat + term) for seat in range(count))
    assert np.allclose(mean, expected, atol=0.15)


def test_jaxtyped_checks_the_filter() -> None:
    checked = jaxtyped(typechecker=beartype)(forward_filter)
    emissions = np_to_jax(np.zeros((5, 2)))
    transition = np_to_jax(np.array([[0.8, 0.2], [0.3, 0.7]]))
    initial = np_to_jax(np.array([0.5, 0.5]))
    out = checked(emissions, transition, initial)
    assert np.asarray(out).shape == (5, 2)
    with pytest.raises(TypeCheckError):
        checked(emissions, transition, np_to_jax(np.ones((2, 2))))


def test_blocked_sweep_is_finite_and_sticky() -> None:
    state, data, prior = _tiny_slds()
    for sweep in range(3):
        state = slds_sweep(random.PRNGKey(20 + sweep), state, data, prior, sequential_modes=False)
    rows = np.asarray(state.transition).sum(axis=1)
    assert np.allclose(rows, 1.0, atol=1e-8)
    assert np.all(np.isfinite(np.asarray(state.latents)))
    assert np.all(np.isfinite(np.asarray(state.dynamics)))
    assert float(np.asarray(state.rho)) > 0.0


def test_smoke_run_finishes() -> None:
    from examples.smoke_slds import run_smoke

    result = run_smoke()
    assert np.all(np.isfinite(result["log_joint"]))
    assert np.allclose(result["transition"].sum(axis=1), 1.0, atol=1e-6)


def _companion_radius(slopes: np.ndarray) -> float:
    daily, week, month = slopes
    coef = np.zeros(22)
    coef[0] = daily + week / 5.0 + month / 22.0
    coef[1:5] = week / 5.0 + month / 22.0
    coef[5:] = month / 22.0
    companion = np.zeros((22, 22))
    companion[0] = coef
    companion[1:, :-1] = np.eye(21)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _tiny_slds() -> tuple[SLDSState, SLDSData, SLDSPrior]:
    import jax.numpy as jnp

    observations = jnp.asarray(np.random.default_rng(14).normal(size=(2, 8, 2)))
    eye = jnp.eye(2)
    state = SLDSState(
        modes=jnp.zeros((2, 8), dtype=jnp.int32),
        latents=observations,
        dynamics=jnp.stack([0.5 * eye, 0.2 * eye]),
        process_cov=jnp.stack([0.2 * eye, 0.3 * eye]),
        emission_cov=0.1 * eye,
        beta=jnp.array([0.6, 0.4]),
        transition=jnp.array([[0.9, 0.1], [0.2, 0.8]]),
        alpha_plus_kappa=jnp.asarray(8.0),
        rho=jnp.asarray(0.75),
        gamma=jnp.asarray(1.5),
        column_precision=jnp.ones((2, 2)),
    )
    prior = SLDSPrior(
        alpha_kappa=jnp.array([2.0, 1.0]),
        rho=jnp.array([8.0, 2.0]),
        gamma=jnp.array([2.0, 1.0]),
        dynamics_mean=0.4 * eye,
        dynamics_df=jnp.asarray(8.0),
        dynamics_scale=0.2 * eye,
        dynamics_column_precision=eye,
        latent_prior=eye,
        emission_df=jnp.asarray(6.0),
        emission_scale=0.2 * eye,
        emission=eye,
        ard=jnp.array([1.0, 1.0]),
    )
    return state, SLDSData(observations=observations), prior
