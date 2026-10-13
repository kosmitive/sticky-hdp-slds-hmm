"""Offline checks for the S&P 500 transform and a tiny Gibbs forecast."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from examples.btc_forecast import _DATA_PATH as _BTC_PATH
from examples.sp500_forecast import (
    _DATA_PATH,
    _predicted_log_rv,
    annualized_vol_percent,
    ewma_variance_forecast,
    fit_and_forecast,
    fit_garch_student,
    forecast_metrics,
    garch_variance_leads,
    load_closes,
    log_vol_winners,
    raw_log_rv,
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
    assert closes.shape[0] == 9261
    assert np.all(closes > 0.0)
    dates = snapshot["dates"]
    assert dates[0].strftime("%Y-%m-%d") == "1990-01-02"
    assert dates[-1].strftime("%Y-%m-%d") == "2026-10-09"
    gaps = [
        (
            dates[i - 1].strftime("%Y-%m-%d"),
            dates[i].strftime("%Y-%m-%d"),
            (dates[i] - dates[i - 1]).days,
        )
        for i in range(1, len(dates))
        if (dates[i] - dates[i - 1]).days > 5
    ]
    assert gaps == [("2001-09-10", "2001-09-17", 7)]


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
    assert fitted["scale"] >= 1.0


def test_garch_variance_leads_match_the_recursion() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=500)
    fitted = fit_garch_student(returns)
    leads = garch_variance_leads(returns, 60)
    one_step = (
        fitted["omega"]
        + fitted["alpha"] * returns[-1] ** 2
        + fitted["beta"] * fitted["sigma2_last"]
    )
    assert np.isclose(leads[0], one_step)
    unconditional = fitted["omega"] / (1.0 - fitted["persistence"])
    for horizon in (5, 20, 60):
        expected = unconditional + fitted["persistence"] ** (horizon - 1) * (
            one_step - unconditional
        )
        assert np.isclose(leads[horizon - 1], expected, rtol=1e-8)


def test_log_rv_uses_realized_squares_then_forecast_variance() -> None:
    returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    variance = np.zeros_like(returns)
    variance[4] = 0.09
    variance[5] = 0.16
    predicted = _predicted_log_rv(returns, variance, 3, 2, 2, 3)
    first = np.log(np.sqrt(np.mean([0.3**2, 0.4**2, 0.09])))
    second = np.log(np.sqrt(np.mean([0.4**2, 0.09, 0.16])))
    assert np.allclose(predicted, [first, second])


def test_garch_forecast_ignores_returns_after_the_training_cut() -> None:
    rng = np.random.default_rng(2)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=120)))
    series = series_from_prices(_dates(closes.size), closes, holdout=15, window=5, seed=4)
    original = volatility_baseline_paths(series, simulations=40, seed=1)
    corrupted = dict(series)
    corrupted["returns"] = np.array(series["returns"], copy=True)
    last_known = int(series["observations_train"].shape[1]) + int(series["window"]) - 2
    corrupted["returns"][last_known + 1 :] = 0.25
    refit = volatility_baseline_paths(corrupted, simulations=40, seed=1)
    assert np.allclose(original["garch"], refit["garch"])
    assert np.allclose(original["garch_lower"], refit["garch_lower"])
    assert np.allclose(original["garch_upper"], refit["garch_upper"])
    assert np.all(original["garch_lower"] < original["garch_upper"])


def test_log_vol_winners_names_the_lowest_rmse() -> None:
    metrics = {
        "rmse_log_vol_model_h1": 0.2,
        "rmse_log_vol_flat_h1": 0.1,
        "rmse_log_vol_ewma_h1": 0.3,
        "rmse_log_vol_garch_h1": 0.05,
        "rmse_log_vol_model_h5": 0.4,
        "rmse_log_vol_flat_h5": 0.4,
        "rmse_log_vol_ewma_h5": 0.5,
        "rmse_log_vol_garch_h5": 0.4,
    }
    winners = log_vol_winners(metrics)
    assert winners[1] == "GARCH"
    assert winners[5] == "SLDS and trailing and GARCH"


def test_volatility_baselines_do_not_read_the_holdout_scale() -> None:
    rng = np.random.default_rng(2)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=80)))
    series = series_from_prices(_dates(closes.size), closes, holdout=10, window=5, seed=4)
    baselines = volatility_baseline_paths(series)
    assert baselines["ewma"].shape == (10,)
    assert np.all(np.isfinite(baselines["ewma"]))
    assert baselines["garch_success"] == 1.0
    assert np.all(np.isfinite(baselines["garch"]))


def test_annualized_volatility_uses_sqrt_252() -> None:
    log_rv = np.log(0.01)
    assert np.isclose(float(annualized_vol_percent(log_rv)), 100.0 * 0.01 * np.sqrt(252.0))
    assert np.isclose(
        float(annualized_vol_percent(log_rv, days=365)), 100.0 * 0.01 * np.sqrt(365.0)
    )
    assert np.allclose(raw_log_rv(np.array([1.0]), 0.5, 2.0), 2.5)


def test_bitcoin_snapshot_is_a_daily_calendar() -> None:
    from examples.volatility_forecast import DATASETS

    spec = DATASETS["btc"]
    assert spec["window"] == 30
    assert spec["holdout"] == 365
    assert spec["annualization_days"] == 365
    snapshot = load_closes(_BTC_PATH)
    dates = snapshot["dates"]
    assert dates[0].strftime("%Y-%m-%d") == "2014-09-17"
    assert dates[-1].strftime("%Y-%m-%d") == "2026-10-10"
    data_rows = [
        line
        for line in _BTC_PATH.read_text().splitlines()
        if line and not line.startswith("#") and not line.lower().startswith("date")
    ]
    assert data_rows[-1].startswith("2026-10-10,")
    assert not any(line.startswith("2026-10-11,") for line in data_rows)
    ordinals = np.array([day.toordinal() for day in dates])
    assert np.all(np.diff(ordinals) == 1)
    series = series_from_prices(
        snapshot["dates"],
        snapshot["closes"],
        holdout=365,
        window=30,
        annualization_days=365,
    )
    assert series["observations_hold"].shape[1] == 365
    assert int(series["window"]) == 30
    center = float(series["center"][1])
    scale = float(series["scale"][1])
    raw_last = float(series["observations_hold"][0, -1, 1]) * scale + center
    rms = np.sqrt(np.mean(np.asarray(series["returns"][-30:]) ** 2))
    assert np.isclose(raw_last, np.log(rms))


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
    baselines = volatility_baseline_paths(series)
    metrics = forecast_metrics(result, baselines)
    assert np.all(np.isfinite(result["log_joint"]))
    assert result["futures"].shape[0] == 1
    assert result["futures"].shape[2] == 8
    assert metrics["rmse_return_model"] >= 0.0
    assert metrics["rmse_vol_flat"] >= 0.0
    assert np.isfinite(metrics["rmse_log_vol_model_h1"])
    assert np.isfinite(metrics["rmse_log_vol_flat_h5"])
    assert np.isfinite(metrics["rmse_log_vol_ewma_h5"])
    assert np.isfinite(metrics["rmse_log_vol_garch_h5"])
    assert 0.0 <= metrics["coverage_garch_h5"] <= 1.0
    assert "rmse_log_vol_model_h20" not in metrics
    assert 0.0 <= metrics["coverage_vol"] <= 1.0
    assert 0.0 <= metrics["coverage_vol_h5"] <= 1.0
    assert any(key.endswith("_ann_vol_pct") for key in metrics)
