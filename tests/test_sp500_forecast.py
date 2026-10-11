"""Offline checks for the S&P 500 transform and a tiny Gibbs forecast."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from examples.btc_forecast import _DATA_PATH as _BTC_PATH
from examples.sp500_forecast import (
    _DATA_PATH,
    ewma_variance_forecast,
    fit_and_forecast,
    fit_garch_student,
    forecast_metrics,
    load_closes,
    rolling_rms,
    series_from_prices,
    volatility_baseline_paths,
)


def _dates(count: int) -> list[datetime]:
    start = datetime(2020, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def test_log_return_and_realized_volatility_match_the_definition() -> None:
    closes = np.exp(np.array([0.0, 0.5, 1.0, 1.2, 1.5]))
    returns = np.log(closes[1:] / closes[:-1])
    assert np.allclose(returns, np.diff(np.log(closes)))
    rms = rolling_rms(returns, 2)
    expected = np.sqrt((returns[0] ** 2 + returns[1] ** 2) / 2.0)
    assert rms.shape == (3,)
    assert np.isclose(rms[0], expected)


def test_standardization_uses_only_the_training_window() -> None:
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=30)))
    # A jump in the holdout must not change the training scale.
    closes[-4:] *= 3.0
    series = series_from_prices(_dates(closes.size), closes, holdout=6, window=4, seed=3)
    train = series["observations_train"][0]
    assert train.shape[1] == 2
    assert np.allclose(train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train.std(axis=0), 1.0, atol=1e-6)
    shifted = closes.copy()
    shifted[-1] *= 2.0
    other = series_from_prices(_dates(closes.size), shifted, holdout=6, window=4, seed=3)
    assert np.allclose(other["observations_train"], series["observations_train"])
    assert not np.allclose(other["observations_hold"], series["observations_hold"])
    assert np.allclose(other["center"], series["center"])
    assert np.allclose(other["scale"], series["scale"])


def test_snapshot_csv_records_the_download_and_parses_offline() -> None:
    text = _DATA_PATH.read_text()
    assert "downloaded: 2026-10-11" in text
    assert "^GSPC" in text
    snapshot = load_closes()
    assert snapshot["downloaded"] == "2026-10-11"
    closes = snapshot["closes"]
    assert closes.shape[0] > 250 + 20
    assert np.all(closes > 0.0)
    assert snapshot["dates"][0] < snapshot["dates"][-1]


def test_ewma_holds_the_last_forecast_flat() -> None:
    returns = np.array([1.0, 2.0, 3.0])
    variance = ewma_variance_forecast(returns, last_known=0, lam=0.5)
    assert np.isclose(variance[0], 1.0)
    assert np.isclose(variance[1], 0.5 * 1.0 + 0.5 * 1.0**2)
    assert np.isclose(variance[2], variance[1])


def test_garch_student_fit_stays_stationary() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=400)
    fitted = fit_garch_student(returns)
    assert fitted["success"] == 1.0
    assert fitted["alpha"] >= 0.0
    assert fitted["beta"] >= 0.0
    assert fitted["persistence"] < 1.0
    assert fitted["nu"] > 2.0


def test_volatility_baselines_do_not_read_the_holdout_scale() -> None:
    rng = np.random.default_rng(2)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=80)))
    series = series_from_prices(_dates(closes.size), closes, holdout=10, window=5, seed=4)
    baselines = volatility_baseline_paths(series)
    assert baselines["ewma"].shape == (10,)
    assert np.all(np.isfinite(baselines["ewma"]))
    assert baselines["garch_success"] == 1.0
    assert np.all(np.isfinite(baselines["garch"]))


def test_bitcoin_snapshot_parses_offline() -> None:
    text = _BTC_PATH.read_text()
    assert "downloaded: 2026-10-11" in text
    assert "BTC-USD" in text
    snapshot = load_closes(_BTC_PATH)
    assert snapshot["closes"].shape[0] > 250 + 20


def test_tiny_synthetic_forecast_is_finite() -> None:
    rng = np.random.default_rng(0)
    shock = rng.normal(0.0, 0.01, size=80)
    shock[40:] *= 4.0
    closes = 100.0 * np.exp(np.cumsum(shock))
    series = series_from_prices(_dates(closes.size), closes, holdout=8, window=5, seed=5)
    result = fit_and_forecast(series, sweeps=2)
    metrics = forecast_metrics(result)
    assert np.all(np.isfinite(result["log_joint"]))
    assert result["futures"].shape[0] == 1
    assert result["futures"].shape[2] == 8
    assert metrics["rmse_return_model"] >= 0.0
    assert metrics["rmse_vol_flat"] >= 0.0
    assert 0.0 <= metrics["coverage_pooled"] <= 1.0
