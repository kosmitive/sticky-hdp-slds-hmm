"""HAR, expanding-mean, EWMA, and GARCH baselines.

GARCH uses the optional ``volatility`` extra (``arch``). The import happens
inside the fit, so the rest of the package imports without it.
"""

from __future__ import annotations

import numpy as np

from sticky_hdp_slds.volatility.series import (
    corsi_features,
    forward_square_mean,
    positive_floor,
)

EWMA_LAMBDA = 0.94


def _ols_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coef, dtype=np.float64)


def direct_har_forecasts(squares: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """Direct HAR(1, 5, 22) forecast of log mean variance, refit at each origin.

    Training rows are earlier origins whose ``horizon``-day target ends at
    or before the current origin. The fit is ordinary least squares with an
    intercept. Exact zeros in the daily feature are replaced by the smallest
    positive square observed before the first origin.
    """

    floor = positive_floor(squares[: max(int(origins[0]), 1)]) if origins.size else 1e-8
    features = corsi_features(squares, floor)
    target = np.log(np.maximum(forward_square_mean(squares, horizon), 1e-18))
    valid = np.flatnonzero(np.isfinite(features).all(axis=1) & np.isfinite(target))
    forecasts = np.full(origins.size, np.nan, dtype=np.float64)
    if valid.size < 30:
        return forecasts
    design = np.column_stack((np.ones(valid.size), features[valid]))
    response = target[valid]
    for index, origin in enumerate(origins):
        # s + horizon <= origin, and ``valid`` stores those origins s.
        usable = int(np.searchsorted(valid, int(origin) - horizon, side="right"))
        if usable < 30:
            continue
        beta = _ols_fit(design[:usable], response[:usable])
        row = np.array(
            [1.0, features[int(origin), 0], features[int(origin), 1], features[int(origin), 2]],
            dtype=np.float64,
        )
        if np.all(np.isfinite(row)):
            forecasts[index] = float(row @ beta)
    return forecasts


def long_run_mean_forecasts(squares: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """Expanding mean of the log-variance target, using only completed windows."""

    target = np.log(np.maximum(forward_square_mean(squares, horizon), 1e-18))
    valid = np.flatnonzero(np.isfinite(target))
    forecasts = np.full(origins.size, np.nan, dtype=np.float64)
    if valid.size == 0:
        return forecasts
    cumulative = np.cumsum(target[valid])
    for index, origin in enumerate(origins):
        usable = int(np.searchsorted(valid, int(origin) - horizon, side="right"))
        if usable < 1:
            continue
        forecasts[index] = float(cumulative[usable - 1] / usable)
    return forecasts


def ewma_variance(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """Causal EWMA variance. Entry ``t`` is the forecast after ``returns[:t]``."""

    values = np.asarray(returns, dtype=np.float64)
    variance = np.empty(values.size + 1, dtype=np.float64)
    variance[0] = float(values[0] ** 2) if values.size else 1e-6
    for index, value in enumerate(values):
        variance[index + 1] = lam * variance[index] + (1.0 - lam) * value**2
    return variance


def ewma_log_forecasts(returns: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Log of the flat EWMA variance forecast.

    The multi-step RiskMetrics forecast is constant, so ``log`` of that
    variance is the forecast of ``log`` mean squared return at every horizon.
    """

    variance = ewma_variance(returns)
    return np.log(np.maximum(variance[np.asarray(origins, dtype=np.int64)], 1e-18))


def garch_log_forecasts(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], int]:
    """``log`` of the mean analytic GARCH variance over each horizon.

    Each origin is fit once, on ``returns[:origin]`` only. The variance path
    is walked out to the longest horizon and averaged over the prefix that
    each horizon needs. A fit that does not converge, or whose persistence
    is not below ``1 + 1e-6``, is counted as a failure and stored as NaN.
    """

    ordered = tuple(sorted(horizons))
    longest = ordered[-1] if ordered else 1
    forecasts = {horizon: np.full(origins.size, np.nan, dtype=np.float64) for horizon in ordered}
    failures = 0
    for index, origin in enumerate(np.asarray(origins, dtype=np.int64)):
        history = np.asarray(returns[: int(origin)], dtype=np.float64)
        if history.size < 50:
            failures += 1
            continue
        fitted = fit_garch_student(history)
        if fitted["success"] != 1.0 or not np.isfinite(fitted["omega"]):
            failures += 1
            continue
        variance = (
            fitted["omega"]
            + fitted["alpha"] * history[-1] ** 2
            + fitted["beta"] * fitted["sigma2_last"]
        )
        path = np.empty(longest, dtype=np.float64)
        persistence = fitted["persistence"]
        for step in range(longest):
            path[step] = variance
            variance = fitted["omega"] + persistence * variance
        for horizon in ordered:
            forecasts[horizon][index] = float(np.log(max(float(np.mean(path[:horizon])), 1e-18)))
    return forecasts, failures


def _fit_arch_garch(returns: np.ndarray):
    """Fit a zero-mean GARCH(1,1) Student-t model with ``arch``."""

    from arch import arch_model

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
