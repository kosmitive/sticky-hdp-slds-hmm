"""Forecast Bitcoin's 30-day realized volatility with a sticky HDP-SLDS.

Bitcoin trades every calendar day, so the log returns are on a 7-day
calendar. The target is the log of a 30-day root mean square, annualized
with ``sqrt(365)``. The last 365 days are held out. The return is the
second observation. Priors match the S&P 500 run and were not tuned on
the holdout.

The snapshot is the Yahoo Finance chart API ``BTC-USD`` daily close,
interval 1d, downloaded 2026-10-11. The row dated 2026-10-11, the
download date, is dropped as an incomplete session. The script does not
fetch anything.

    python examples/btc_forecast.py
    python examples/volatility_forecast.py btc
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.sp500_forecast import DEFAULT_SWEEPS
from examples.volatility_forecast import _BTC_PATH as _DATA_PATH
from examples.volatility_forecast import run_dataset

__all__ = ["_DATA_PATH"]


def main() -> None:
    """Fit the committed Bitcoin snapshot and score the 365-day holdout."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEPS
    run_dataset("btc", sweeps)


if __name__ == "__main__":
    main()
