"""Offline checks for the close snapshots and the GARCH baseline."""

from __future__ import annotations

import numpy as np
from examples.market_data import (
    BTC_PATH,
    DATASETS,
    SP500_PATH,
    fit_garch_student,
    garch_variance_leads,
    load_closes,
)


def test_sp500_snapshot_records_the_download_and_parses_offline() -> None:
    text = SP500_PATH.read_text()
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
            dates[index - 1].strftime("%Y-%m-%d"),
            dates[index].strftime("%Y-%m-%d"),
            (dates[index] - dates[index - 1]).days,
        )
        for index in range(1, len(dates))
        if (dates[index] - dates[index - 1]).days > 5
    ]
    assert gaps == [("2001-09-10", "2001-09-17", 7)]


def test_bitcoin_snapshot_is_a_daily_calendar() -> None:
    spec = DATASETS["btc"]
    assert spec["holdout"] == 365
    assert spec["annualization_days"] == 365
    snapshot = load_closes(BTC_PATH)
    dates = snapshot["dates"]
    assert dates[0].strftime("%Y-%m-%d") == "2014-09-17"
    assert dates[-1].strftime("%Y-%m-%d") == "2026-10-10"
    ordinals = np.array([day.toordinal() for day in dates])
    assert np.all(np.diff(ordinals) == 1)
    text = BTC_PATH.read_text()
    assert "downloaded: 2026-10-11" in text
    assert "BTC-USD" in text


def test_garch_student_fit_stays_stationary() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=400)
    fitted = fit_garch_student(returns)
    assert fitted["success"] == 1.0
    assert fitted["persistence"] < 1.0
    assert fitted["nu"] > 2.0


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
