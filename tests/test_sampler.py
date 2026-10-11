"""Regression tests for the sticky HDP-SLDS Gibbs sampler.

The checks below lock the corrections against Fox et al., arXiv:1003.3829,
and the sticky HDP-HMM auxiliary-variable sampler those updates rely on.
"""

from __future__ import annotations

import tempfile

import numpy as np
from scipy import stats

from examples.smoke_slds import run_smoke
from model.ard_prior import ARDPrior
from model.gt_sticky_hdp_slds_hmm import StickyHDPSLDSHMMConfiguration
from model.mniw_prior import MNIWPrior
from model.sticky_hdp_prior import StickyHDPPrior
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel


def _model(
    max_z: int = 2,
    ld: int = 2,
    od: int | None = None,
    n0: float = 8.0,
    alpha0: tuple[float, float] | None = None,
) -> StickyHDPSLDSHMMModel:
    """Build a model shell with dimensions set and parameters left empty."""

    if od is None:
        od = ld
    hdp = StickyHDPPrior(max_z, [2.0, 1.0], [8.0, 2.0], [2.0, 1.0], "uniform")
    if alpha0 is None:
        dyn = MNIWPrior(0.5 * np.eye(ld), np.eye(ld), n0, 0.2 * np.eye(ld))
    else:
        dyn = ARDPrior(list(alpha0), n0, 0.2 * np.eye(ld))
    return StickyHDPSLDSHMMModel(
        ld,
        od,
        hdp,
        dyn,
        np.eye(ld),
        [5.0, 0.1 * np.eye(od)],
        np.eye(od, ld),
    )


def test_stick_breaking_accepts_a_vector_base_measure() -> None:
    """Prior draws used to crash once a row base measure was an array."""

    model = _model()
    model.init_params()
    beta = model.hdp_params[3]
    pi = model.hdp_params[4]
    assert beta.shape == (2,)
    assert pi.shape == (2, 2)
    assert np.allclose(beta.sum(), 1.0)
    assert np.allclose(pi.sum(axis=1), 1.0)


def test_first_customer_always_opens_a_table() -> None:
    """Algorithm 1 counts the first customer with the seat index starting at 0."""

    model = _model()
    model.hdp_params = [1.0, 0.0, 1.0, np.array([0.5, 0.5]), np.eye(2)]
    model.transitions = np.zeros((2, 2))
    model.transitions[0, 0] = 1
    draws = []
    for _ in range(40):
        model.sample_post_auxiliary_vars()
        draws.append(model.m[0, 0])
    assert np.all(np.asarray(draws) == 1.0)


def test_table_count_matches_chinese_restaurant_expectation() -> None:
    """Expected table count is sum_{n=0}^{N-1} a / (a + n), not the shifted sum."""

    model = _model()
    concentration = 0.25
    model.hdp_params = [1.0, 0.0, 1.0, np.array([concentration, 1.0 - concentration]), np.eye(2)]
    # rho = 0 so kappa = 0 and the self-transition term is alpha * beta_0.
    # alpha = (1 - rho) * alpha_kappa = 1, beta_0 = 0.25, term = 0.25.
    count = 8
    model.transitions = np.zeros((2, 2))
    model.transitions[0, 0] = count
    expected = sum(concentration / (concentration + n) for n in range(count))
    draws = []
    for _ in range(600):
        model.sample_post_auxiliary_vars()
        draws.append(model.m[0, 0])
    assert abs(float(np.mean(draws)) - expected) < 0.2


def test_empty_rows_are_skipped_when_sampling_alpha_kappa(monkeypatch) -> None:
    """Restaurants with no customers do not get Beta/Bernoulli auxiliaries."""

    calls = {"beta": 0}

    def _beta(a, b):
        calls["beta"] += 1
        return 0.5

    monkeypatch.setattr("slds.slds_sample_sticky_hdp_params.beta.rvs", _beta)
    monkeypatch.setattr(
        "slds.slds_sample_sticky_hdp_params.bernoulli.rvs",
        lambda p: 0,
    )
    model = _model()
    model.m = np.zeros((2, 2))
    model.transitions = np.zeros((2, 2))
    model.hdp_params = [1.0, 0.5, 1.0, np.full(2, 0.5), np.eye(2)]
    model.sample_post_hdp_alpha_kappa()
    assert calls["beta"] == 0
    # Posterior collapses to the Gamma(2, rate=1) prior.
    assert model.hist_alpha_kappa[-1][0] == 2.0


def test_auxiliary_override_counts_do_not_replace_emission_noise() -> None:
    model = _model()
    model.w = np.ones((3, 4, 2))
    model.hdp_params = [1.0, 0.5, 1.0, np.full(2, 0.5), np.full((2, 2), 0.5)]
    model.transitions = np.eye(2)
    model.sample_post_auxiliary_vars()
    assert model.w.shape == (3, 4, 2)
    assert model.override_counts.shape == (2,)


def test_emission_degrees_of_freedom_count_observations(monkeypatch) -> None:
    captured = {}

    def _rvs(df, scale, *args, **kwargs):
        captured["df"] = df
        captured["scale"] = np.array(scale, copy=True)
        return np.eye(2)

    monkeypatch.setattr("slds.slds_sample_r.invwishart.rvs", _rvs)
    model = _model()
    model.y = np.zeros((3, 5, 2))
    model.x = np.zeros((3, 5, 2))
    model.C = np.eye(2)
    model.R = np.eye(2)
    prior_scale = np.array(model.emission_prior[1], copy=True)
    model.slds_post_sample_r()
    assert captured["df"] == 3 * 5 + 5
    assert np.allclose(model.emission_prior[1], prior_scale)


def test_regression_counts_ignore_the_initial_state(monkeypatch) -> None:
    captured = []

    def _rvs(df, scale, *args, **kwargs):
        captured.append(df)
        return np.eye(2)

    monkeypatch.setattr("slds.slds_mniw_prior_sample.invwishart.rvs", _rvs)
    monkeypatch.setattr(
        "slds.slds_mniw_prior_sample.matrix_normal.rvs",
        lambda *args, **kwargs: np.eye(2),
    )
    model = _model(n0=6)
    model.z = np.array([[0, 1, 1]])
    model.x = np.ones((1, 3, 2))
    model.slds_params = [np.zeros((2, 2, 2)), np.repeat(np.eye(2)[None, :, :], 2, axis=0)]
    model.slds_post_mniw_dynamics_param_sampling()
    # Mode 0 occurs only at t = 0, so it contributes no regression pair.
    assert captured[0] == 6
    assert captured[1] == 6 + 2


def test_mniw_posterior_concentrates_on_the_dynamics() -> None:
    np.random.seed(0)
    rng = np.random.default_rng(0)
    ld = 2
    true_a = np.array([[0.55, 0.10], [0.00, 0.35]])
    sigma = 0.05 * np.eye(ld)
    steps = 80
    x = np.zeros((1, steps, ld))
    x[0, 0] = rng.normal(scale=0.1, size=ld)
    for t in range(1, steps):
        x[0, t] = true_a @ x[0, t - 1] + rng.multivariate_normal(np.zeros(ld), sigma)
    model = _model(max_z=1, n0=4)
    model.dyn_prior.M = np.zeros((ld, ld))
    model.dyn_prior.K = 0.01 * np.eye(ld)
    model.dyn_prior.S0 = 0.01 * np.eye(ld)
    model.z = np.zeros((1, steps), dtype=np.int64)
    model.x = x
    model.slds_params = [np.zeros((1, ld, ld)), np.repeat(sigma[None, :, :], 1, axis=0)]
    samples = []
    for _ in range(25):
        model.slds_post_mniw_dynamics_param_sampling()
        samples.append(model.slds_params[0][0].copy())
    stacked = np.stack(samples)
    assert np.linalg.norm(stacked.mean(axis=0) - true_a) < 0.3
    assert stacked[:, 0, 0].std() < 0.25


def test_ard_gamma_prior_uses_scale_not_location() -> None:
    prior = ARDPrior([16.0, 8.0], 6, np.eye(2))
    draws = np.stack([prior.prior_sample(4)[2] for _ in range(40)])
    # Mean of Gamma(shape=16, rate=8) is 2. The loc/scale mix-up has mean 16.
    assert abs(float(draws.mean()) - 2.0) < 0.3


def test_ard_precision_shrinks_the_selected_column() -> None:
    model = _model(max_z=1, alpha0=(1.0, 1.0))
    model.slds_params = [
        np.zeros((1, 2, 2)),
        np.repeat(np.eye(2)[None, :, :], 1, axis=0),
        np.array([[1.0e6, 1.0e-3]]),
    ]
    empty = [np.zeros((2, 0))]
    samples = []
    sampler = model._StickyHDPSLDSHMMModelARDMixin__slds_post_ard_dynamics_sample_dyn_mat
    for _ in range(40):
        sampler(empty, empty)
        samples.append(model.slds_params[0][0].copy())
    stacked = np.stack(samples)
    assert stacked[:, :, 0].std() < 0.05
    assert stacked[:, :, 1].std() > 1.0


def test_ard_posterior_keeps_column_orientation(monkeypatch) -> None:
    """The posterior mean must be the least-squares matrix, not its transpose."""

    def _return_mean(mean, cov):
        return np.array(mean, copy=True)

    monkeypatch.setattr(
        "slds.slds_ard_prior_sample.np.random.multivariate_normal",
        _return_mean,
    )
    rng = np.random.default_rng(1)
    true_a = np.array([[0.2, 0.8], [0.0, -0.5]])
    steps = 40
    model = _model(max_z=1, alpha0=(1.0e-4, 1.0e-4))
    x = np.zeros((1, steps, 2))
    x[0, 0] = rng.normal(size=2)
    for t in range(1, steps):
        x[0, t] = true_a @ x[0, t - 1]
    model.x = x
    model.z = np.zeros((1, steps), dtype=np.int64)
    model.slds_params = [
        np.zeros((1, 2, 2)),
        np.repeat(np.eye(2)[None, :, :], 1, axis=0),
        np.full((1, 2), 1.0e-8),
    ]
    model._StickyHDPSLDSHMMModelARDMixin__slds_post_ard_dynamics_sample_dyn_mat(
        *model._StickyHDPSLDSHMMModelMNIWMixin__split_into_regression_problems()[:2]
    )
    estimate = model.slds_params[0][0]
    assert np.allclose(estimate, true_a, atol=1e-2)
    assert np.linalg.norm(estimate - true_a) < np.linalg.norm(estimate - true_a.T)


def test_ard_alpha_posterior_uses_half_the_sum_of_squares(monkeypatch) -> None:
    captured = {}

    def _gamma(shape, scale):
        captured["shape"] = shape
        captured["scale"] = np.array(scale, copy=True)
        return np.ones_like(scale)

    monkeypatch.setattr("slds.slds_ard_prior_sample.np.random.gamma", _gamma)
    model = _model(max_z=1, alpha0=(4.0, 2.0))
    matrix = np.array([[3.0, 0.0], [0.0, 4.0]])
    model.slds_params = [np.array([matrix]), np.array([np.eye(2)]), np.ones((1, 2))]
    model._StickyHDPSLDSHMMModelARDMixin__slds_post_ard_dynamics_sample_alpha()
    column_squares = np.sum(matrix ** 2, axis=0)
    assert captured["shape"] == 4.0 + 1.0
    assert np.allclose(captured["scale"], 1.0 / (2.0 + 0.5 * column_squares))


def test_ard_alpha_survives_store_and_load() -> None:
    model = _model(alpha0=(2.0, 1.0))
    model.slds_params = [
        np.zeros((2, 2, 2)),
        np.repeat(0.1 * np.eye(2)[None, :, :], 2, axis=0),
        np.arange(4, dtype=float).reshape(2, 2),
    ]
    model.hdp_params = [1.0, 0.5, 1.0, np.full(2, 0.5), np.full((2, 2), 0.5)]
    model.R = 0.1 * np.eye(2)
    model.z = np.zeros((1, 3), dtype=np.int64)
    model.x = np.zeros((1, 3, 2))
    model.e = np.zeros((1, 3, 2))
    model.y = np.zeros((1, 3, 2))
    model.w = np.zeros((1, 3, 2))
    with tempfile.TemporaryDirectory() as folder:
        model.store(folder)
        loaded = StickyHDPSLDSHMMModel.load(folder)
    assert len(loaded.slds_params) == 3
    assert np.allclose(loaded.slds_params[2], model.slds_params[2])


def test_log_likelihood_adds_every_mode_and_the_initial_state() -> None:
    model = _model()
    model.slds_params = [
        np.stack([0.5 * np.eye(2), -0.2 * np.eye(2)]),
        np.repeat(0.2 * np.eye(2)[None, :, :], 2, axis=0),
    ]
    model.hdp_params = [1.0, 0.5, 1.0, np.array([0.7, 0.3]), np.array([[0.8, 0.2], [0.4, 0.6]])]
    model.R = 0.3 * np.eye(2)
    model.C = np.eye(2)
    model.x = np.array([[[0.0, 0.0], [1.0, -1.0], [0.2, 0.4]]])
    model.z = np.array([[0, 1, 0]])
    model.y = np.array([[[0.1, -0.2], [0.8, -0.7], [0.0, 0.5]]])

    latent = 0.0
    for t in range(1, 3):
        mode = int(model.z[0, t])
        mean = model.slds_params[0][mode] @ model.x[0, t - 1]
        latent += stats.multivariate_normal.logpdf(model.x[0, t], mean, model.slds_params[1][mode])
    emission = 0.0
    for t in range(3):
        emission += stats.multivariate_normal.logpdf(model.y[0, t], model.C @ model.x[0, t], model.R)
    modes = np.log(0.7)
    modes += np.log(model.hdp_params[4][0, 1])
    modes += np.log(model.hdp_params[4][1, 0])
    assert np.allclose(model.calc_log_likelihood(), latent + emission + modes)


def test_forward_filter_starts_at_the_prior_precision() -> None:
    model = _model()
    model.latent_prior = 0.25 * np.eye(2)
    model.C = np.eye(2)
    model.R = 0.5 * np.eye(2)
    model.z = np.zeros((1, 1), dtype=np.int64)
    model.y = np.array([[[0.3, -0.1]]])
    model.slds_params = [np.array([0.5 * np.eye(2)]), np.array([0.2 * np.eye(2)])]
    mats, vecs = model.slds_forward_kalman_filter_gains()
    assert np.allclose(mats[0, 0], np.linalg.inv(model.latent_prior))

    precision = np.linalg.inv(model.latent_prior)
    dyn = model.slds_params[0][0]
    noise = model.slds_params[1][0]
    predicted = np.linalg.inv(noise + dyn @ np.linalg.inv(precision) @ dyn.T)
    updated = predicted + model.C.T @ np.linalg.inv(model.R) @ model.C
    info = model.C.T @ np.linalg.inv(model.R) @ model.y[0, 0]
    assert np.allclose(mats[0, 1], updated, atol=1e-6)
    assert np.allclose(vecs[0, 1], info, atol=1e-6)


def test_block_mode_sampler_matches_the_two_step_posterior() -> None:
    model = _model()
    beta = np.array([0.65, 0.35])
    pi = np.array([[0.8, 0.2], [0.3, 0.7]])
    model.hdp_params = [1.0, 0.5, 1.0, beta, pi]
    model.slds_params = [
        np.stack([0.9 * np.eye(2), -0.4 * np.eye(2)]),
        np.repeat(0.05 * np.eye(2)[None, :, :], 2, axis=0),
    ]
    model.x = np.array([[[0.2, -0.1], [0.15, -0.05]]])
    model.y = model.x.copy()
    model.z = np.zeros((1, 2), dtype=np.int64)

    lik = np.array([
        stats.multivariate_normal.pdf(
            model.x[0, 1],
            model.slds_params[0][k] @ model.x[0, 0],
            model.slds_params[1][k],
        )
        for k in range(2)
    ])
    joint = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            joint[i, j] = beta[i] * pi[i, j] * lik[j]
    joint /= joint.sum()

    counts = np.zeros((2, 2))
    for _ in range(800):
        model.slds_post_block_sample_z()
        counts[int(model.z[0, 0]), int(model.z[0, 1])] += 1
    frequencies = counts / counts.sum()
    assert np.allclose(frequencies, joint, atol=0.05)


def test_sequential_sampler_uses_the_candidate_dynamics() -> None:
    np.random.seed(2)
    rng = np.random.default_rng(2)
    model = _model()
    model.latent_prior = 0.05 * np.eye(2)
    model.C = np.eye(2)
    model.R = 0.01 * np.eye(2)
    model.hdp_params = [1.0, 0.2, 1.0, np.full(2, 0.5), np.full((2, 2), 0.5)]
    model.slds_params = [
        np.stack([0.95 * np.eye(2), -0.95 * np.eye(2)]),
        np.repeat(0.01 * np.eye(2)[None, :, :], 2, axis=0),
    ]
    steps = 6
    x = np.zeros((1, steps, 2))
    x[0, 0] = rng.normal(scale=0.05, size=2)
    for t in range(1, steps):
        x[0, t] = model.slds_params[0][0] @ x[0, t - 1]
        x[0, t] += rng.multivariate_normal(np.zeros(2), model.slds_params[1][0])
    model.x = x
    model.y = x + rng.normal(scale=0.05, size=x.shape)
    model.z = np.ones((1, steps), dtype=np.int64)
    chosen = []
    for _ in range(30):
        backward = model.slds_backward_kalman_filter_gains()
        forward = model.slds_forward_kalman_filter_gains()
        model.slds_post_seq_sample_z(forward[0], forward[1], backward[0], backward[1])
        chosen.append(np.mean(model.z == 0))
    assert float(np.mean(chosen)) > 0.7


def test_block_state_sampler_tracks_a_low_noise_trajectory() -> None:
    rng = np.random.default_rng(3)
    model = _model()
    model.latent_prior = 0.05 * np.eye(2)
    model.C = np.eye(2)
    model.R = 0.01 * np.eye(2)
    model.slds_params = [
        np.array([0.8 * np.eye(2)]),
        np.array([0.01 * np.eye(2)]),
    ]
    model.hdp_params = [1.0, 0.5, 1.0, np.array([1.0]), np.array([[1.0]])]
    steps = 8
    x = np.zeros((1, steps, 2))
    x[0, 0] = rng.normal(scale=0.05, size=2)
    for t in range(1, steps):
        x[0, t] = 0.8 * x[0, t - 1] + rng.normal(scale=0.05, size=2)
    model.x = x.copy()
    model.y = x + rng.normal(scale=0.05, size=x.shape)
    model.z = np.zeros((1, steps), dtype=np.int64)
    draws = []
    for _ in range(25):
        backward = model.slds_backward_kalman_filter_gains()
        model.slds_post_block_sample_x(backward[0], backward[1])
        draws.append(model.x.copy())
    average = np.mean(draws, axis=0)
    assert np.mean(np.abs(average - x)) < 0.15


def test_rounds_hdp_is_stored_on_its_own_field() -> None:
    gt = _model()
    gt.slds_params = [np.zeros((2, 2, 2)), np.repeat(np.eye(2)[None], 2, axis=0)]
    gt.hdp_params = [1.0, 0.5, 1.0, np.full(2, 0.5), np.full((2, 2), 0.5)]
    gt.R = np.eye(2)
    gt.z = np.zeros((1, 2), dtype=np.int64)
    gt.x = np.zeros((1, 2, 2))
    gt.y = np.zeros((1, 2, 2))
    gt.e = np.zeros_like(gt.x)
    gt.w = np.zeros_like(gt.y)
    with tempfile.TemporaryDirectory() as folder:
        gt.store(folder)
        conf = {
            "model_path": folder,
            "sample_x": False,
            "sample_block_z": False,
            "sample_seq_z": False,
            "sample_hdp_beta": False,
            "sample_hdp_alpha_kappa": False,
            "sample_hdp_rho": False,
            "sample_hdp_pi": False,
            "sample_hdp_gamma": False,
            "sample_dynamics": False,
            "sample_r": False,
            "gibbs_steps": 1,
            "rounds_hdp": 4,
            "rounds_ard": 2,
        }
        experiment = StickyHDPSLDSHMMConfiguration.init_for_test(conf)
    assert experiment.rounds_hdp == 4
    assert experiment.rounds_ard == 2


def test_simulated_emission_noise_matches_the_observations() -> None:
    model = _model()
    model.slds_params = [np.stack([0.4 * np.eye(2), 0.2 * np.eye(2)]), np.repeat(0.1 * np.eye(2)[None], 2, axis=0)]
    model.hdp_params = [1.0, 0.8, 1.0, np.full(2, 0.5), np.array([[0.9, 0.1], [0.2, 0.8]])]
    model.R = 0.05 * np.eye(2)
    model.simulate(3, 5, init=False)
    assert model.y.shape == (3, 5, 2)
    assert model.w.shape == model.y.shape
    assert model.x.shape == (3, 5, 2)
    assert model.z.shape == (3, 5)


def test_smoke_run_finishes_with_a_sticky_transition_row() -> None:
    report = run_smoke(steps=3, seed=7)
    assert np.isfinite(report["log_likelihood"])
    assert report["x_shape"] == report["y_shape"]
    assert report["w_shape"] == report["y_shape"]
    assert np.allclose(report["pi_row_sums"], 1.0)
    assert 0.0 < report["rho"] < 1.0
    assert report["sticky_diagonal"] > 0.7
