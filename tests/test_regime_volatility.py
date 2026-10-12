"""Checks for the rolling-origin sticky regime HAR."""

from __future__ import annotations

import numpy as np
from examples.regime_volatility import (
    LAG,
    PRIOR_ALPHA,
    PRIOR_KAPPA,
    diebold_mariano,
    direct_har_forecasts,
    evaluate_returns,
    expected_self_stay,
    feature_matrix,
    filter_draws,
    fit_gibbs,
    forward_square_mean,
    garch_log_forecasts,
    har_spectral_radius,
    log_squares,
)
from examples.sp500_forecast import garch_variance_leads


def test_prior_self_stay_is_near_98_percent() -> None:
    assert expected_self_stay(PRIOR_ALPHA, PRIOR_KAPPA, 3) == 0.98


def test_har_features_use_only_earlier_observations() -> None:
    series = np.arange(1, 31, dtype=np.float64)
    features = feature_matrix(series)
    assert np.isnan(features[:LAG]).all()
    assert features[22, 0] == series[21]
    assert np.isclose(features[22, 1], np.mean(series[17:22]))
    assert np.isclose(features[22, 2], np.mean(series[0:22]))
    assert np.isclose(features[25, 2], np.mean(series[3:25]))


def test_forward_target_is_the_log_ready_mean_of_future_squares() -> None:
    squares = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means = forward_square_mean(squares, 2)
    assert np.isclose(means[0], 1.5)
    assert np.isclose(means[1], 2.5)
    assert np.isclose(means[3], 4.5)
    assert np.isnan(means[4])


def test_companion_radius_tracks_the_daily_ar_coefficient() -> None:
    assert har_spectral_radius(np.array([0.5, 0.0, 0.0])) < 0.51
    assert har_spectral_radius(np.array([1.2, 0.0, 0.0])) > 1.1
    stable = np.array([0.2, 0.2, 0.3])
    assert har_spectral_radius(stable) < 1.0


def test_direct_har_does_not_read_the_target_window() -> None:
    rng = np.random.default_rng(4)
    squares = rng.random(180) + 0.05
    origin = np.array([150])
    first = direct_har_forecasts(squares, origin, 5)
    changed = squares.copy()
    changed[150:] += 10.0
    second = direct_har_forecasts(changed, origin, 5)
    assert np.allclose(first, second)
    changed_past = squares.copy()
    changed_past[100] += 3.0
    third = direct_har_forecasts(changed_past, origin, 5)
    assert not np.allclose(first, third)


def test_diebold_mariano_sign_matches_the_loss_gap() -> None:
    tied = diebold_mariano(np.array([1.0, -1.0, 1.0, -1.0]), np.zeros(4), 0)
    assert tied["stat"] == 0.0
    better = np.zeros(40)
    worse = np.ones(40)
    result = diebold_mariano(better, worse, 0)
    assert result["stat"] < 0.0
    assert result["pvalue"] < 0.01
    worse_model = diebold_mariano(worse, better, 1)
    assert worse_model["stat"] > 0.0


def test_filter_locks_onto_the_closer_regime() -> None:
    series = np.concatenate((np.zeros(40), np.full(40, 4.0)))
    features = np.zeros((series.size, 3))
    draws = {
        "c": np.array([[0.0, 4.0]]),
        "beta": np.zeros((1, 3)),
        "sigma": np.array([[0.2, 0.2]]),
        "trans": np.array([[[0.95, 0.05], [0.05, 0.95]]]),
    }
    probs = filter_draws(series, features, draws)
    assert np.allclose(probs.sum(axis=2), 1.0)
    assert probs[20, 0, 0] > 0.9
    assert probs[70, 0, 1] > 0.9


def test_gibbs_draws_stay_stationary_and_ordered() -> None:
    rng = np.random.default_rng(5)
    length = 280
    latent = np.zeros(length, dtype=np.int64)
    for time_index in range(1, length):
        latent[time_index] = latent[time_index - 1]
        if rng.random() > 0.97:
            latent[time_index] = 1 - latent[time_index]
    noise = rng.normal(0.0, 0.35, size=length)
    series = np.zeros(length)
    for time_index in range(1, length):
        level = -1.5 if latent[time_index] == 0 else 1.5
        series[time_index] = level + 0.15 * series[time_index - 1] + noise[time_index]
    features = feature_matrix(series)
    fit = fit_gibbs(series, features, regimes=2, sweeps=40, chains=2, seed=9)
    assert fit["c"].shape[1] == 2
    assert np.all(fit["c"][:, 0] <= fit["c"][:, 1] + 1e-8)
    assert har_spectral_radius(fit["beta"].mean(axis=0)) < 1.0
    assert float(fit["c"][:, 1].mean() - fit["c"][:, 0].mean()) > 0.5
    stay = np.diagonal(fit["trans"], axis1=1, axis2=2).mean()
    assert stay > 0.9


def test_origin_forecast_ignores_later_returns() -> None:
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, size=420)
    kwargs = {
        "holdout": 36,
        "train_days": 140,
        "sweeps": 8,
        "chains": 1,
        "refit_every": 36,
        "paths": 2,
        "seed": 11,
        "horizons": (1, 5),
        "regimes": 2,
        "fit_garch": False,
    }
    first = evaluate_returns(returns, **kwargs)
    changed = returns.copy()
    changed[-36:] += 0.05
    second = evaluate_returns(changed, **kwargs)
    assert np.isclose(first["regime"][1][0], second["regime"][1][0])
    assert np.isclose(first["baseline"][1]["har"][0], second["baseline"][1]["har"][0])
    assert np.isclose(first["baseline"][5]["ewma"][0], second["baseline"][5]["ewma"][0])
    assert first["probabilities"][0].shape == (2,)
    assert np.isclose(first["probabilities"][0].sum(), 1.0)


def test_garch_log_mean_matches_the_analytic_path() -> None:
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0, 0.01, size=360)
    origin = np.array([300, 320])
    forecasts, failures = garch_log_forecasts(returns, origin, (1, 5))
    assert failures == 0
    leads = garch_variance_leads(returns[:300], 5)
    assert np.isclose(forecasts[1][0], np.log(leads[0]))
    assert np.isclose(forecasts[5][0], np.log(np.mean(leads)))
    bumped = returns.copy()
    bumped[300:] += 0.2
    again, _failed = garch_log_forecasts(bumped, origin[:1], (1,))
    assert np.isclose(again[1][0], forecasts[1][0])


def test_log_squares_replaces_exact_zeros() -> None:
    returns = np.array([0.0, 0.02, -0.01])
    floor = 0.02**2
    logged = log_squares(returns, floor)
    assert np.all(np.isfinite(logged))
    assert np.isclose(logged[0], np.log(floor))
