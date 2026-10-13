"""Regression against the published JAX runs.

Tolerances are Monte Carlo bands around the stored reference, wide enough
for a one-draw difference and tight enough to catch a changed forecast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from examples.pendulum_forecast import fit_and_forecast, forecast_metrics

from sticky_hdp_slds import evaluate_returns, fit_regime_har, load_closes
from sticky_hdp_slds.volatility import LAG, datasets, feature_matrix, log_squares, positive_floor

_DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
DATASETS = datasets(_DATA)

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "published.json").read_text())
_RMSE_ATOL = 0.005
_MOMENT_ATOL = 0.05
_PENDULUM_RMSE_ATOL = 0.02
_PENDULUM_COVERAGE_ATOL = 0.02


def _btc_window() -> tuple[np.ndarray, np.ndarray]:
    spec = DATASETS["btc"]
    snapshot = load_closes(spec["path"])
    closes = snapshot["closes"]
    returns = np.log(closes[1:] / closes[:-1])
    holdout = int(spec["holdout"])
    train_days = 4 * int(spec["annualization_days"])
    left = returns.size - holdout
    start = max(LAG, left - train_days)
    floor = positive_floor(returns[:left])
    log_var = log_squares(returns[:left], floor)
    features = feature_matrix(log_var)
    usable = np.isfinite(log_var[start:left]) & np.isfinite(features[start:left]).all(axis=1)
    return log_var[start:left][usable], features[start:left][usable]


def test_posterior_moments_match_the_published_window() -> None:
    series, features = _btc_window()
    fit = fit_regime_har(series, features, regimes=3, sweeps=2000, chains=4, seed=23)
    reference = _FIXTURE["posterior_mean"]
    intercept = fit["coef"][:, :3].mean(axis=0)
    slope = fit["coef"][:, 3:].mean(axis=0)
    sigma = fit["sigma"].mean(axis=0)
    assert np.allclose(intercept, reference["intercept"], atol=_MOMENT_ATOL)
    assert np.allclose(slope, reference["slope"], atol=_MOMENT_ATOL)
    assert np.allclose(sigma, reference["sigma"], atol=_MOMENT_ATOL)


@pytest.mark.parametrize("name", ["sp500", "btc"])
def test_rolling_origin_rmse_matches_the_published_table(name: str) -> None:
    spec = DATASETS[name]
    snapshot = load_closes(spec["path"])
    closes = snapshot["closes"]
    returns = np.log(closes[1:] / closes[:-1])
    result = evaluate_returns(
        returns,
        holdout=int(spec["holdout"]),
        train_days=4 * int(spec["annualization_days"]),
        dates=np.asarray(snapshot["dates"]),
    )
    reference = _FIXTURE[f"{name}_rmse"]
    for horizon, expected in reference.items():
        got = result["scores"][int(horizon)]["regime"]["rmse"]
        assert got == pytest.approx(expected, abs=_RMSE_ATOL)


def test_pendulum_forecast_matches_the_published_scores() -> None:
    metrics = forecast_metrics(fit_and_forecast(300, 80))
    reference = _FIXTURE["pendulum"]
    assert metrics["rmse_model_10"] == pytest.approx(reference["rmse_10"], abs=_PENDULUM_RMSE_ATOL)
    assert metrics["rmse_model_full"] == pytest.approx(
        reference["rmse_full"], abs=_PENDULUM_RMSE_ATOL
    )
    assert metrics["coverage_10"] == pytest.approx(
        reference["coverage_10"], abs=_PENDULUM_COVERAGE_ATOL
    )
    assert metrics["coverage_full"] == pytest.approx(
        reference["coverage_full"], abs=_PENDULUM_COVERAGE_ATOL
    )
