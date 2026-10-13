"""Daily close snapshots and the GARCH baseline used by the volatility study.

The files in ``examples/data`` are Yahoo Finance chart API daily closes.
Nothing here fetches a new series.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from arch import arch_model

_DATA_DIR = Path(__file__).resolve().parent / "data"
SP500_PATH = _DATA_DIR / "sp500_daily.csv"
BTC_PATH = _DATA_DIR / "btc_daily.csv"

DATASETS = {
    "sp500": {
        "path": SP500_PATH,
        "prefix": "sp500",
        "name": "S&P 500",
        "window": 20,
        "holdout": 250,
        "annualization_days": 252,
    },
    "btc": {
        "path": BTC_PATH,
        "prefix": "btc",
        "name": "Bitcoin",
        "window": 30,
        "holdout": 365,
        "annualization_days": 365,
    },
}


def load_closes(path: Path = SP500_PATH) -> dict[str, object]:
    """Read a committed close snapshot.

    Args:
        path: CSV with ``date,close`` rows and ``#`` comment lines. One
            comment records the download date.

    Returns:
        Dates, closes, the source line, and the download date.
    """

    dates: list[datetime] = []
    closes: list[float] = []
    source = ""
    downloaded = ""
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "downloaded:" in stripped:
                downloaded = stripped.split(":", 1)[1].strip()
            elif "source:" in stripped:
                source = stripped.split(":", 1)[1].strip()
            continue
        if stripped.lower().startswith("date"):
            continue
        day, price = stripped.split(",")
        dates.append(datetime.strptime(day, "%Y-%m-%d"))
        closes.append(float(price))
    if len(closes) < 3:
        raise ValueError(f"{path} does not have enough rows.")
    return {
        "dates": dates,
        "closes": np.asarray(closes, dtype=np.float64),
        "source": source,
        "downloaded": downloaded,
    }


def _fit_arch_garch(returns: np.ndarray):
    """Fit a zero-mean GARCH(1,1) Student-t model with ``arch``."""

    model = arch_model(
        np.asarray(returns, dtype=np.float64),
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=True,
    )
    return model.fit(disp="off", show_warning=False)


def _garch_params(fitted: object) -> dict[str, float]:
    """Read original-unit GARCH parameters off an ``arch`` result."""

    scale = float(fitted.scale)
    omega = float(fitted.params["omega"]) / scale**2
    alpha = float(fitted.params["alpha[1]"])
    beta = float(fitted.params["beta[1]"])
    nu = float(fitted.params["nu"])
    sigma2_last = float(fitted.conditional_volatility[-1]) ** 2 / scale**2
    stationary = alpha >= 0.0 and beta >= 0.0 and (alpha + beta) < 1.0 + 1e-6 and nu > 2.0
    converged = int(fitted.convergence_flag) == 0 and np.isfinite(sigma2_last)
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "nu": nu,
        "persistence": alpha + beta,
        "sigma2_last": sigma2_last,
        "scale": scale,
        "success": 1.0 if stationary and converged else 0.0,
    }


def fit_garch_student(returns: np.ndarray) -> dict[str, float]:
    """Fit GARCH(1,1) with Student-t errors on the returns passed in.

    Estimates are in the original return units. ``success`` is 1 when
    ``arch`` reports convergence, the variance is finite, and ``nu > 2``.
    """

    return _garch_params(_fit_arch_garch(returns))


def garch_variance_leads(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Analytic GARCH variance of each future return, in original units.

    Lead ``h`` is the variance of the return ``h`` days after the last
    entry of ``returns``. ``arch`` returns rescaled units; this divides
    by ``scale**2``.
    """

    fitted = _fit_arch_garch(returns)
    scale = float(fitted.scale)
    forecast = fitted.forecast(horizon=horizon, method="analytic", reindex=False)
    leads = np.asarray(forecast.variance, dtype=np.float64)[-1] / scale**2
    return np.maximum(leads, 1e-18)
