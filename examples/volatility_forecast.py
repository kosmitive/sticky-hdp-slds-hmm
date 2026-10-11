"""Forecast realized volatility for one daily close snapshot.

The dataset name picks the file, the realized-volatility window, the
holdout, and the annualization. Both series are daily. The S&P 500 uses
trading days, a 20-day window, a 250-day holdout, and ``sqrt(252)``.
Bitcoin trades every calendar day, including weekends, so its log returns
are on a 7-day calendar, the window is 30 days, the holdout is the last
365 days, and annualization uses ``sqrt(365)``. The sampler, the priors,
and the baselines are the same. Nothing is chosen on the holdout.

    python examples/volatility_forecast.py sp500
    python examples/volatility_forecast.py btc
    python examples/volatility_forecast.py btc 40
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.sp500_forecast import _DATA_PATH, DEFAULT_SWEEPS, run_forecast

_BTC_PATH = Path(__file__).resolve().parent / "data" / "btc_daily.csv"

DATASETS = {
    "sp500": {
        "path": _DATA_PATH,
        "prefix": "sp500",
        "name": "S&P 500",
        "window": 20,
        "holdout": 250,
        "annualization_days": 252,
    },
    "btc": {
        "path": _BTC_PATH,
        "prefix": "btc",
        "name": "Bitcoin",
        "window": 30,
        "holdout": 365,
        "annualization_days": 365,
    },
}


def run_dataset(name: str, sweeps: int) -> None:
    """Fit one named dataset and write its figures and scores.

    Args:
        name: ``sp500`` or ``btc``.
        sweeps: Gibbs sweeps. Forecasts use the second half.
    """

    spec = DATASETS[name]
    run_forecast(
        spec["path"],
        spec["prefix"],
        spec["name"],
        sweeps,
        window=int(spec["window"]),
        holdout=int(spec["holdout"]),
        annualization_days=int(spec["annualization_days"]),
    )


def main() -> None:
    """Run the dataset named in the first argument."""

    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        names = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Usage: python examples/volatility_forecast.py <{names}> [sweeps]")
    sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SWEEPS
    run_dataset(sys.argv[1], sweeps)


if __name__ == "__main__":
    main()
