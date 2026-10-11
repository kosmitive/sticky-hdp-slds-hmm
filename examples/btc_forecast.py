"""Forecast the sticky HDP-SLDS on daily Bitcoin closes.

The transform matches the S&P 500 example: daily log returns and the log of
20-day realized volatility, standardized on the training window only. The
snapshot is Yahoo Finance ``BTC-USD``. Stooq's daily file was behind a
browser check on the download date, so it is not the source. Priors are the
same weakly informative settings as the equity run and were not tuned on
the holdout. Bitcoin trades every day, so the sample includes weekends.

    python examples/btc_forecast.py
    python examples/btc_forecast.py 40
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.sp500_forecast import DEFAULT_SWEEPS, run_forecast

_DATA_PATH = Path(__file__).resolve().parent / "data" / "btc_daily.csv"


def main() -> None:
    """Fit the committed Bitcoin snapshot and score the 250-day holdout."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEPS
    run_forecast(_DATA_PATH, "btc", "Bitcoin", sweeps)


if __name__ == "__main__":
    main()
