"""Rolling-origin volatility forecasts from a sticky regime HAR.

The scored target at an origin is the log of the mean squared return over
the next ``h`` days. Origins in the holdout are scored one at a time. HAR,
the long-run mean, EWMA, and GARCH see only returns strictly before the
target window.

The regime model is a finite Markov-switching regression on daily
``log(r^2)``, not the sticky HDP-SLDS. Three regimes share one HAR slope
on lags 1, 5, and 22 and keep their own intercepts and innovation scales.
The Dirichlet prior on each transition row has mean self-stay 0.98. Slopes
are rejected unless the companion spectral radius stays below 0.999.
The scored forecast is a direct regression for each horizon: the target
is ``log(mean r^2)`` over the next ``h`` days, and each regime has its
own intercept and HAR slopes. Those regressions are weighted by the
filtered regime distribution at the origin after it has been moved
forward with the transition matrix. An additive bias correction uses
only windows that have already ended. Parameters are refit on a trailing
four-year window.

    python examples/regime_volatility.py sp500
    python examples/regime_volatility.py btc
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.market_data import DATASETS, fit_garch_student, load_closes

# Fixed before any holdout score was read.
HORIZONS = (1, 5, 20, 60)
N_REGIMES = 3
PRIOR_ALPHA = 1.0
PRIOR_KAPPA = 97.0
LAG = 22
WEEK = 5
REFIT_YEARS = 4
REFIT_EVERY = 21
SWEEPS = 2000
CHAINS = 4
FORECAST_DRAWS = 40
MIN_REGIME_WEIGHT = 30.0
SEED = 23
EWMA_LAMBDA = 0.94
_IMG_DIR = _ROOT / "docs" / "img"


def expected_self_stay(alpha: float, kappa: float, regimes: int) -> float:
    """Prior mean of a diagonal transition entry.

    Row ``i`` is Dirichlet with concentration ``alpha`` off the diagonal and
    ``alpha + kappa`` on the diagonal. The mean self-stay is
    ``(alpha + kappa) / (regimes * alpha + kappa)``.
    """

    return (alpha + kappa) / (regimes * alpha + kappa)


def positive_floor(returns: np.ndarray) -> float:
    """Smallest positive squared return, used to keep ``log(r^2)`` finite."""

    squares = np.asarray(returns, dtype=np.float64) ** 2
    positive = squares[squares > 0.0]
    if positive.size == 0:
        return 1e-8
    return float(positive.min())


def log_squares(returns: np.ndarray, floor: float) -> np.ndarray:
    """``log(max(r^2, floor))``."""

    return np.log(np.maximum(np.asarray(returns, dtype=np.float64) ** 2, floor))


def feature_matrix(log_variance: np.ndarray) -> np.ndarray:
    """HAR features of daily log variance: lag 1, mean of 5, mean of 22.

    Row ``t`` is built from ``log_variance`` strictly before ``t``. Rows
    before index 22 are NaN.

    Args:
        log_variance: Daily ``log(r^2)`` series.

    Returns:
        Array of shape ``(T, 3)``.
    """

    series = np.asarray(log_variance, dtype=np.float64)
    features = np.full((series.size, 3), np.nan, dtype=np.float64)
    if series.size <= LAG:
        return features
    cumulative = np.cumsum(series)
    index = np.arange(LAG, series.size)
    features[index, 0] = series[index - 1]
    features[index, 1] = (cumulative[index - 1] - cumulative[index - 1 - WEEK]) / float(WEEK)
    previous = np.zeros(index.size, dtype=np.float64)
    late = index >= LAG + 1
    previous[late] = cumulative[index[late] - 1 - LAG]
    features[index, 2] = (cumulative[index - 1] - previous) / float(LAG)
    return features


def har_spectral_radius(slopes: np.ndarray) -> float:
    """Spectral radius of the HAR companion matrix.

    The regressors are ``y[t-1]``, the mean of the previous 5 observations,
    and the mean of the previous 22. The function expands those three
    weights into the 22 distributed-lag coefficients and returns the largest
    absolute eigenvalue of the companion matrix.
    """

    daily = float(slopes[0])
    week = float(slopes[1]) / float(WEEK)
    month = float(slopes[2]) / float(LAG)
    coef = np.zeros(LAG, dtype=np.float64)
    coef[0] = daily + week + month
    coef[1:WEEK] = week + month
    coef[WEEK:] = month
    companion = np.zeros((LAG, LAG), dtype=np.float64)
    companion[0, :] = coef
    companion[1:, :-1] = np.eye(LAG - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def forward_square_mean(squares: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of the next ``horizon`` squared returns, at each origin.

    Origin ``t`` uses ``squares[t : t + horizon]`` and does not use
    ``squares[t]`` as a feature. Entries that would run past the sample
    are NaN.
    """

    values = np.asarray(squares, dtype=np.float64)
    out = np.full(values.size, np.nan, dtype=np.float64)
    if horizon < 1 or values.size < horizon:
        return out
    cumulative = np.cumsum(values)
    ends = cumulative[horizon - 1 :]
    previous = np.concatenate(([0.0], cumulative[: values.size - horizon]))
    out[: values.size - horizon + 1] = (ends - previous) / float(horizon)
    return out


def _ending_square_mean(squares: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``window`` squares ending at each index, inclusive."""

    cumulative = np.cumsum(squares)
    out = np.full(squares.size, np.nan, dtype=np.float64)
    if squares.size < window:
        return out
    previous = np.concatenate(([0.0], cumulative[:-window]))
    out[window - 1 :] = (cumulative[window - 1 :] - previous) / float(window)
    return out


def corsi_features(squares: np.ndarray, floor: float) -> np.ndarray:
    """Corsi features at each origin: log daily, log 5-day, log 22-day variance.

    Row ``t`` uses squares strictly before ``t``. The daily term floors a
    zero square at ``floor``. The weekly and monthly terms are logs of
    averages of raw squares.
    """

    daily = np.full(squares.size, np.nan, dtype=np.float64)
    if squares.size > 1:
        daily[1:] = np.log(np.maximum(squares[:-1], floor))
    week = np.full(squares.size, np.nan, dtype=np.float64)
    month = np.full(squares.size, np.nan, dtype=np.float64)
    week_mean = _ending_square_mean(squares, WEEK)
    month_mean = _ending_square_mean(squares, LAG)
    if squares.size > WEEK:
        week[WEEK:] = np.log(np.maximum(week_mean[WEEK - 1 : -1], 1e-18))
    if squares.size > LAG:
        month[LAG:] = np.log(np.maximum(month_mean[LAG - 1 : -1], 1e-18))
    return np.column_stack((daily, week, month))


def diebold_mariano(
    loss_model: np.ndarray, loss_base: np.ndarray, hac_lag: int
) -> dict[str, float]:
    """Diebold-Mariano statistic for the mean loss gap, with Newey-West variance.

    The loss differential is ``loss_model - loss_base``. A negative statistic
    means the model loss is smaller. The long-run variance uses the Bartlett
    kernel with ``hac_lag`` lags (``h - 1`` for an ``h``-step forecast) and
    autocovariances divided by the sample size. The p-value is two-sided
    normal. If the kernel estimate is not positive, the lag-0 variance is
    used and ``nw_fallback`` is 1.
    """

    gap = np.asarray(loss_model, dtype=np.float64) - np.asarray(loss_base, dtype=np.float64)
    count = int(gap.size)
    if count < 2:
        return {"stat": float("nan"), "pvalue": float("nan"), "nw_fallback": 1.0, "n": float(count)}
    centered = gap - gap.mean()
    gamma0 = float(np.dot(centered, centered) / count)
    long_run = gamma0
    lag = max(int(hac_lag), 0)
    for step in range(1, min(lag, count - 1) + 1):
        weight = 1.0 - step / float(lag + 1)
        gamma = float(np.dot(centered[step:], centered[:-step]) / count)
        long_run += 2.0 * weight * gamma
    fallback = 0.0
    if not np.isfinite(long_run) or long_run <= 0.0:
        long_run = gamma0
        fallback = 1.0
    mean_gap = float(gap.mean())
    if long_run <= 0.0:
        stat = 0.0 if mean_gap == 0.0 else float(np.sign(mean_gap) * np.inf)
    else:
        stat = mean_gap / float(np.sqrt(long_run / count))
    if np.isfinite(stat):
        pvalue = float(math.erfc(abs(stat) / math.sqrt(2.0)))
    else:
        pvalue = 0.0
    return {"stat": float(stat), "pvalue": pvalue, "nw_fallback": fallback, "n": float(count)}


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


def propagated_regime_weights(
    probability: np.ndarray, transition: np.ndarray, horizon: int
) -> np.ndarray:
    """Average the filtered regime distribution over the next ``horizon`` days.

    ``probability`` has shape ``(draws, regimes)`` and is the filter at the
    last observed day. The returned row is the mean of
    ``probability @ transition**k`` for ``k = 1, ..., horizon``.
    """

    current = np.asarray(probability, dtype=np.float64)
    transition = np.asarray(transition, dtype=np.float64)
    total = np.zeros_like(current)
    steps = int(horizon)
    for _step in range(steps):
        current = np.einsum("ni,nik->nk", current, transition)
        current = current / np.maximum(current.sum(axis=1, keepdims=True), 1e-300)
        total += current
    return total / float(steps)


def predictive_weights(
    filtered: np.ndarray, transition: np.ndarray, positions: np.ndarray, horizon: int
) -> np.ndarray:
    """Predictive regime weights at many origins.

    ``filtered`` is ``(time, draws, regimes)`` and ``positions`` indexes the
    last observed day of each origin. The result has shape
    ``(origins, draws, regimes)``.
    """

    current = np.asarray(filtered, dtype=np.float64)[np.asarray(positions, dtype=np.int64)]
    transition = np.asarray(transition, dtype=np.float64)
    total = np.zeros_like(current)
    for _step in range(int(horizon)):
        current = np.einsum("ndk,dki->ndi", current, transition)
        current = current / np.maximum(current.sum(axis=-1, keepdims=True), 1e-300)
        total += current
    return total / float(horizon)


def fit_direct_regime_coefficients(
    features: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    min_weight: float = MIN_REGIME_WEIGHT,
) -> np.ndarray:
    """Weighted direct HAR for one Gibbs draw.

    ``features`` is ``(n, 3)``, ``weights`` is ``(n, regimes)``, and
    ``targets`` is ``(n,)``. The return value is ``(regimes, 4)``:
    intercept, then slopes on lags 1, 5, and 22. A regime whose weights
    sum to less than ``min_weight`` uses the pooled regression.
    """

    design = np.column_stack((np.ones(features.shape[0]), np.asarray(features, dtype=np.float64)))
    target = np.asarray(targets, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(design).all(axis=1) & np.isfinite(weight).all(axis=1)
    design = design[finite]
    target = target[finite]
    weight = np.maximum(weight[finite], 0.0)
    pooled = _ols_coefficients(design, target)
    if pooled is None:
        pooled = np.zeros(design.shape[1] if design.size else 4, dtype=np.float64)
    regimes = weight.shape[1]
    coef = np.empty((regimes, pooled.size), dtype=np.float64)
    for regime in range(regimes):
        column = weight[:, regime]
        if float(column.sum()) < min_weight:
            coef[regime] = pooled
            continue
        scale = np.sqrt(column)
        fitted = _ols_coefficients(design * scale[:, None], target * scale)
        coef[regime] = pooled if fitted is None else fitted
    return coef


def _ols_coefficients(design: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    if design.shape[0] < design.shape[1] or not np.isfinite(design).all():
        return None
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=1e-8)
    if rank < design.shape[1] or not np.all(np.isfinite(coef)):
        return None
    return np.asarray(coef, dtype=np.float64)


def mixture_log_forecast(
    features: np.ndarray, weights: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Regime-weighted direct prediction, one column per Gibbs draw.

    ``features`` is ``(n, 3)``, ``weights`` is ``(n, draws, regimes)``, and
    ``coefficients`` is ``(draws, regimes, 4)``.
    """

    level = coefficients[None, :, :, 0] + np.einsum(
        "nf,drf->ndr", np.asarray(features, dtype=np.float64), coefficients[:, :, 1:]
    )
    return np.sum(weights * level, axis=2)


def training_bias(
    raw: np.ndarray, actual: np.ndarray, candidates: np.ndarray, origin: int, horizon: int
) -> np.ndarray:
    """Mean residual on origins whose ``horizon`` window has already ended.

    ``raw`` has shape ``(n, draws)``. The correction is one number per draw.
    """

    usable = np.asarray(candidates) + int(horizon) <= int(origin)
    if not np.any(usable):
        return np.zeros(raw.shape[1], dtype=np.float64)
    return (raw[usable] - np.asarray(actual, dtype=np.float64)[usable, None]).mean(axis=0)


def _score_block(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, object]:
    observed = np.asarray(actual, dtype=np.float64)
    scores: dict[str, object] = {}
    squared: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, values in forecasts.items():
        mask = np.isfinite(observed) & np.isfinite(values)
        masks[name] = mask
        error = values[mask] - observed[mask]
        squared[name] = (values - observed) ** 2
        scores[name] = {
            "rmse": float(np.sqrt(np.mean(error**2))) if mask.any() else float("nan"),
            "mse": float(np.mean(error**2)) if mask.any() else float("nan"),
            "bias": float(np.mean(error)) if mask.any() else float("nan"),
            "n": int(mask.sum()),
        }
    comparisons = {}
    for name, mask in masks.items():
        if name == "regime":
            continue
        both = mask & masks["regime"]
        comparisons[name] = diebold_mariano(
            squared["regime"][both], squared[name][both], horizon - 1
        )
    scores["dm"] = comparisons
    scores["n"] = scores["regime"]["n"]
    return scores


def evaluate_returns(
    returns: np.ndarray,
    *,
    holdout: int,
    train_days: int,
    sweeps: int = SWEEPS,
    chains: int = CHAINS,
    refit_every: int = REFIT_EVERY,
    seed: int = SEED,
    horizons: tuple[int, ...] = HORIZONS,
    regimes: int = N_REGIMES,
    fit_garch: bool = True,
    dates: np.ndarray | None = None,
) -> dict[str, object]:
    """Score the regime HAR and the baselines on one return series.

    Args:
        returns: Daily log returns.
        holdout: Number of trailing forecast origins. The first origin is
            ``len(returns) - holdout``.
        train_days: Trailing observations used at each refit, after the
            22-day lag.
        sweeps: Gibbs sweeps per chain.
        chains: Number of chains.
        refit_every: Origins between refits.
        seed: Master seed.
        horizons: Forecast horizons, in days.
        regimes: Number of sticky regimes.
        fit_garch: Fit the GARCH baseline. Tests can turn this off.
        dates: Optional date for each return, used by the plots.

    Returns:
        RMSE, mean squared error, bias, Diebold-Mariano tests, sampler
        diagnostics, and the direct forecasts.
    """

    values = np.asarray(returns, dtype=np.float64)
    count = int(values.size)
    if holdout < max(horizons) or count - holdout <= LAG + 30:
        raise ValueError("Need a longer series for this holdout.")
    started = time.perf_counter()
    first_origin = count - holdout
    origins = np.arange(first_origin, count, dtype=np.int64)
    squares = values**2
    garch_failures = 0
    baseline: dict[int, dict[str, np.ndarray]] = {}
    garch_by_horizon: dict[int, np.ndarray] = {}
    if fit_garch:
        # One fit per origin covers every horizon. The origin set is the
        # h=1 set; longer horizons use the prefix of that set.
        garch_by_horizon, garch_failures = garch_log_forecasts(values, origins, horizons)
    for horizon in horizons:
        usable = origins[origins + horizon <= count]
        block = {
            "har": direct_har_forecasts(squares, usable, horizon),
            "mean": long_run_mean_forecasts(squares, usable, horizon),
            "ewma": ewma_log_forecasts(values, usable),
        }
        if fit_garch:
            block["garch"] = garch_by_horizon[horizon][: usable.size]
        baseline[horizon] = block
    regime_forecast, diagnostics, probabilities, origin_dates = _regime_forecasts(
        values,
        origins,
        train_days=train_days,
        sweeps=sweeps,
        chains=chains,
        refit_every=refit_every,
        seed=seed,
        horizons=horizons,
        regimes=regimes,
        dates=dates,
    )
    actual = {
        horizon: np.log(
            np.maximum(
                forward_square_mean(squares, horizon)[origins[origins + horizon <= count]], 1e-18
            )
        )
        for horizon in horizons
    }
    scores = {}
    for horizon in horizons:
        usable = origins + horizon <= count
        forecasts = {"regime": regime_forecast[horizon]}
        forecasts.update(baseline[horizon])
        scores[horizon] = _score_block(actual[horizon], forecasts, horizon)
    diagnostics["garch_failures"] = garch_failures
    diagnostics["elapsed_sec"] = float(time.perf_counter() - started)
    diagnostics["zero_returns"] = int(np.sum(values == 0.0))
    diagnostics["holdout"] = int(holdout)
    diagnostics["train_days"] = int(train_days)
    diagnostics["sweeps"] = int(sweeps)
    diagnostics["chains"] = int(chains)
    diagnostics["refit_every"] = int(refit_every)
    diagnostics["seed"] = int(seed)
    diagnostics["regimes"] = int(regimes)
    diagnostics["self_stay_prior"] = float(expected_self_stay(PRIOR_ALPHA, PRIOR_KAPPA, regimes))
    return {
        "scores": scores,
        "diagnostics": diagnostics,
        "actual": actual,
        "regime": regime_forecast,
        "baseline": baseline,
        "probabilities": probabilities,
        "origins": origins,
        "dates": origin_dates,
        "horizons": horizons,
    }


def _max_finite(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _thin_draws(fit: dict[str, np.ndarray], limit: int) -> dict[str, np.ndarray]:
    """Evenly thin each chain's retained draws before the forecast filter."""

    chains = int(fit["loglik"].shape[0])
    total = int(fit["c"].shape[0])
    keep = total // chains
    if keep <= limit:
        return fit
    chosen = []
    for chain in range(chains):
        base = chain * keep
        slots = np.linspace(0, keep - 1, limit, dtype=np.int64)
        chosen.append(base + slots)
    index = np.concatenate(chosen)
    thinned = dict(fit)
    for key in ("c", "beta", "sigma", "trans", "coef"):
        thinned[key] = fit[key][index]
    return thinned


def _jax_window_fits(
    returns: np.ndarray,
    boundaries: list[int],
    origins: np.ndarray,
    train_days: int,
    sweeps: int,
    chains: int,
    seed: int,
    regimes: int,
) -> list[dict[str, np.ndarray]]:
    """Fit every refit window with one vmapped JAX Gibbs run.

    Windows are exactly ``train_days`` long, so they stack. Each window
    uses its own log-variance floor, the smallest positive square in
    ``returns[:left]``. The forecast filter is the jitted local recursion
    in ``filter_regime_draws``.

    Args:
        returns: Daily log returns.
        boundaries: First origin of each refit block.
        origins: Every holdout origin.
        train_days: Trailing length of each window.
        sweeps: Sweeps per chain, including burn-in.
        chains: Chains per window.
        seed: Seed for the JAX key.
        regimes: Number of regimes.

    Returns:
        One second-half draw dictionary per boundary.
    """

    import jax
    import jax.numpy as jnp
    from jax import random
    from slds.regime_har import pack_regime_draws, sample_regime_windows

    chunks_y = []
    chunks_x = []
    for left in boundaries:
        start = max(LAG, left - train_days)
        floor = positive_floor(returns[:left])
        log_var = log_squares(returns[:left], floor)
        features = feature_matrix(log_var)
        window_y = np.asarray(log_var[start:left], dtype=np.float64)
        window_x = np.asarray(features[start:left], dtype=np.float64)
        usable = np.isfinite(window_y) & np.isfinite(window_x).all(axis=1)
        chunks_y.append(window_y[usable])
        chunks_x.append(window_x[usable])
    lengths = {chunk.shape[0] for chunk in chunks_y}
    if len(lengths) != 1:
        raise ValueError(f"JAX refit windows are not equal length: {sorted(lengths)}")
    draws = sample_regime_windows(
        random.PRNGKey(seed),
        jnp.asarray(np.stack(chunks_y)),
        jnp.asarray(np.stack(chunks_x)),
        regimes,
        chains,
        sweeps,
    )
    packed = jax.tree_util.tree_map(np.asarray, draws)
    fits = []
    for index in range(len(chunks_y)):
        window = jax.tree_util.tree_map(lambda array, i=index: array[i], packed)
        fits.append(pack_regime_draws(window, sweeps=sweeps, regimes=regimes))
    del origins
    return fits


def _regime_forecasts(
    returns: np.ndarray,
    origins: np.ndarray,
    *,
    train_days: int,
    sweeps: int,
    chains: int,
    refit_every: int,
    seed: int,
    horizons: tuple[int, ...],
    regimes: int,
    dates: np.ndarray | None,
) -> tuple[dict[int, np.ndarray], dict[str, object], np.ndarray, np.ndarray]:
    count = returns.size
    forecasts = {
        horizon: np.full(int(np.sum(origins + horizon <= count)), np.nan, dtype=np.float64)
        for horizon in horizons
    }
    probabilities = np.full((origins.size, regimes), np.nan, dtype=np.float64)
    if dates is None:
        origin_dates = origins.astype(np.float64)
    else:
        # ``dates[t]`` is the date of ``returns[t - 1]`` when ``dates`` is
        # indexed like the close series (one longer than ``returns``).
        if len(dates) == count + 1:
            origin_dates = dates[origins]
        else:
            origin_dates = dates[origins - 1]
    boundaries = list(range(int(origins[0]), int(origins[-1]) + 1, refit_every))
    jax_fits = _jax_window_fits(
        returns, boundaries, origins, train_days, sweeps, chains, seed, regimes
    )
    refit_stats = []
    first_loglik = None
    import jax.numpy as jnp
    from slds.hmm import filter_regime_draws

    columns = {
        horizon: {
            int(origin): slot for slot, origin in enumerate(origins[origins + horizon <= count])
        }
        for horizon in horizons
    }
    for block_index, left in enumerate(boundaries):
        right = (
            boundaries[block_index + 1]
            if block_index + 1 < len(boundaries)
            else int(origins[-1]) + 1
        )
        floor = positive_floor(returns[:left])
        log_var = log_squares(returns[:right], floor)
        features = feature_matrix(log_var)
        start = max(LAG, left - train_days)
        fit = jax_fits[block_index]
        if first_loglik is None:
            first_loglik = fit["loglik"]
        refit_stats.append(_summarize_fit(fit, left, start))
        forecast_fit = _thin_draws(fit, FORECAST_DRAWS)
        filtered = np.asarray(
            filter_regime_draws(
                jnp.asarray(log_var[start:right], dtype=jnp.float64),
                jnp.asarray(features[start:right], dtype=jnp.float64),
                jnp.asarray(forecast_fit["c"], dtype=jnp.float64),
                jnp.asarray(forecast_fit["beta"], dtype=jnp.float64),
                jnp.asarray(forecast_fit["sigma"], dtype=jnp.float64),
                jnp.asarray(forecast_fit["trans"], dtype=jnp.float64),
            )
        )
        print(
            f"refit origin {left} window {left - start} draws {fit['c'].shape[0]} "
            f"forecast_draws {forecast_fit['c'].shape[0]} "
            f"rhat_max {_max_finite(fit['rhat']):.3f} "
            f"self_stay {refit_stats[-1]['self_stay_mean']:.3f} "
            f"accept {float(fit['acceptance'][0]):.2f}",
            flush=True,
        )
        direct = _direct_forecast_block(
            returns,
            features,
            filtered,
            forecast_fit["trans"],
            start=start,
            left=left,
            right=right,
            horizons=horizons,
        )
        for origin in range(left, right):
            position = origin - 1 - start
            if position < 0 or position >= filtered.shape[0]:
                continue
            origin_slot = int(origin - origins[0])
            probabilities[origin_slot] = filtered[position].mean(axis=0)
            for horizon in horizons:
                column = columns[horizon].get(origin)
                if column is None or origin not in direct[horizon]:
                    continue
                forecasts[horizon][column] = direct[horizon][origin]
    diagnostics = {
        "refits": refit_stats,
        "rhat_max": _max_finite(np.array([item["rhat_max"] for item in refit_stats])),
        "loglik_first": first_loglik,
        "forecast": "direct",
    }
    return forecasts, diagnostics, probabilities, origin_dates


def _direct_forecast_block(
    returns: np.ndarray,
    features: np.ndarray,
    filtered: np.ndarray,
    transition: np.ndarray,
    *,
    start: int,
    left: int,
    right: int,
    horizons: tuple[int, ...],
) -> dict[int, dict[int, float]]:
    """Bias-corrected direct forecasts for origins in ``[left, right)``.

    Coefficients use origins whose target window ends at or before ``left``.
    The bias at a later origin also uses windows that ended after the refit
    and before that origin.
    """

    squares = np.asarray(returns, dtype=np.float64) ** 2
    out: dict[int, dict[int, float]] = {horizon: {} for horizon in horizons}
    earliest = max(LAG, start + 1)
    for horizon in horizons:
        training = np.arange(earliest, left - horizon + 1, dtype=np.int64)
        if training.size == 0:
            continue
        training = training[np.isfinite(features[training]).all(axis=1)]
        if training.size == 0:
            continue
        train_pos = training - 1 - start
        train_weights = predictive_weights(filtered, transition, train_pos, horizon)
        train_target = np.log(np.maximum(forward_square_mean(squares, horizon)[training], 1e-18))
        train_features = features[training]
        draw_count = transition.shape[0]
        coefficients = np.empty((draw_count, transition.shape[1], 4), dtype=np.float64)
        for draw in range(draw_count):
            coefficients[draw] = fit_direct_regime_coefficients(
                train_features, train_weights[:, draw, :], train_target
            )
        candidates = np.arange(earliest, right, dtype=np.int64)
        candidates = candidates[candidates + horizon <= squares.size]
        candidates = candidates[np.isfinite(features[candidates]).all(axis=1)]
        if candidates.size == 0:
            continue
        raw = mixture_log_forecast(
            features[candidates],
            predictive_weights(filtered, transition, candidates - 1 - start, horizon),
            coefficients,
        )
        actual = np.log(np.maximum(forward_square_mean(squares, horizon)[candidates], 1e-18))
        lookup = {int(origin): index for index, origin in enumerate(candidates)}
        for origin in range(left, right):
            row = lookup.get(origin)
            if row is None:
                continue
            bias = training_bias(raw, actual, candidates, origin, horizon)
            out[horizon][origin] = float(np.mean(raw[row] - bias))
    return out


def _summarize_fit(fit: dict[str, np.ndarray], origin: int, start: int) -> dict[str, object]:
    self_stay = np.diagonal(fit["trans"], axis1=1, axis2=2)
    slopes = fit["beta"].mean(axis=0)
    return {
        "origin": int(origin),
        "start": int(start),
        "rhat_max": _max_finite(fit["rhat"]),
        "rhat": fit["rhat"],
        "self_stay_mean": float(self_stay.mean()),
        "self_stay_by_regime": self_stay.mean(axis=0),
        "duration_days": 1.0 / np.maximum(1.0 - self_stay.mean(axis=0), 1e-6),
        "intercepts": fit["c"].mean(axis=0),
        "slopes": slopes,
        "sigma": fit["sigma"].mean(axis=0),
        "occupancy": fit["occupancy"].mean(axis=(0, 1)),
        "spectral_radius": float(har_spectral_radius(slopes)),
        "acceptance": float(fit["acceptance"][0]),
        "loglik_second_mean": float(fit["loglik"][:, fit["loglik"].shape[1] // 2 :].mean()),
        "loglik_second_std": float(fit["loglik"][:, fit["loglik"].shape[1] // 2 :].std()),
    }


def _print_report(name: str, result: dict[str, object]) -> None:
    diagnostics = result["diagnostics"]
    print(f"{name} rolling-origin log mean r^2", flush=True)
    print(
        "  "
        + " ".join(
            f"{key} {diagnostics[key]}"
            for key in (
                "holdout",
                "train_days",
                "sweeps",
                "chains",
                "refit_every",
                "seed",
                "regimes",
                "self_stay_prior",
                "zero_returns",
                "garch_failures",
            )
        ),
        flush=True,
    )
    print(
        f"  rhat_max {diagnostics['rhat_max']:.4f} elapsed_sec {diagnostics['elapsed_sec']:.1f}",
        flush=True,
    )
    for item in diagnostics["refits"]:
        intercepts = " ".join(f"{value:.3f}" for value in item["intercepts"])
        slopes = " ".join(f"{value:.3f}" for value in item["slopes"])
        stay = " ".join(f"{value:.3f}" for value in item["self_stay_by_regime"])
        share = " ".join(f"{value:.3f}" for value in item["occupancy"])
        print(
            f"  refit {item['origin']} rhat {item['rhat_max']:.3f} stay {stay} "
            f"dur {' '.join(f'{value:.1f}' for value in item['duration_days'])} "
            f"c {intercepts} b {slopes} radius {item['spectral_radius']:.3f} "
            f"share {share} ll {item['loglik_second_mean']:.1f} "
            f"ll_std {item['loglik_second_std']:.1f}",
            flush=True,
        )
    for horizon, scores in result["scores"].items():
        parts = [f"h {horizon} n {scores['n']}"]
        for model in ("regime", "har", "ewma", "garch", "mean"):
            if model not in scores:
                continue
            parts.append(f"{model} {scores[model]['rmse']:.6f} bias {scores[model]['bias']:.6f}")
        print("  " + " ".join(parts), flush=True)
        for model, test in scores["dm"].items():
            print(
                f"    dm vs {model} stat {test['stat']:.4f} p {test['pvalue']:.4g} "
                f"fallback {test['nw_fallback']:.0f}",
                flush=True,
            )


def save_regime_plots(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    """Write the forecast, horizon, probability, and convergence figures."""

    destination.mkdir(parents=True, exist_ok=True)
    _plot_forecast(name, asset_name, result, destination)
    _plot_horizon(name, asset_name, result, destination)
    _plot_probabilities(name, asset_name, result, destination)
    _plot_convergence(name, asset_name, result, destination)


def _plot_forecast(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    horizon = 20 if 20 in result["actual"] else result["horizons"][-1]
    actual = result["actual"][horizon]
    dates = np.asarray(result["dates"])[: actual.size]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(dates, actual, color="black", lw=1.3, label="realized")
    series = [
        ("regime", "#1f4e79", 1.4),
        ("har", "#b85c38", 1.1),
        ("ewma", "#2a9d8f", 1.0),
        ("garch", "#6d597a", 1.0),
        ("mean", "#888888", 1.0),
    ]
    for key, color, width in series:
        if key == "regime":
            values = result["regime"][horizon]
        else:
            values = result["baseline"][horizon][key]
        ax.plot(dates, values, color=color, lw=width, label=key)
    ax.set_ylabel(f"log mean r^2 over {horizon} days")
    ax.set_xlabel("forecast origin")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{asset_name}: rolling-origin {horizon}-day log variance")
    if np.issubdtype(np.asarray(dates).dtype, np.datetime64) or _is_datetime(dates):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_forecast.png", dpi=140)
    plt.close(fig)


def _is_datetime(dates: np.ndarray) -> bool:
    if dates.size == 0:
        return False
    return hasattr(dates[0], "strftime")


def _plot_horizon(name: str, asset_name: str, result: dict[str, object], destination: Path) -> None:
    horizons = list(result["horizons"])
    models = ["regime", "har", "ewma", "garch", "mean"]
    colors = ["#1f4e79", "#b85c38", "#2a9d8f", "#6d597a", "#888888"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.15
    positions = np.arange(len(horizons), dtype=np.float64)
    for offset, (model, color) in enumerate(zip(models, colors, strict=True)):
        heights = []
        for horizon in horizons:
            if model not in result["scores"][horizon]:
                heights.append(np.nan)
            else:
                heights.append(result["scores"][horizon][model]["rmse"])
        ax.bar(positions + (offset - 2) * width, heights, width=width, color=color, label=model)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(horizon) for horizon in horizons])
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("RMSE of log mean r^2")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{asset_name}: rolling-origin RMSE")
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_horizon.png", dpi=140)
    plt.close(fig)


def _plot_probabilities(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    dates = np.asarray(result["dates"])
    probs = result["probabilities"]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    for regime in range(probs.shape[1]):
        axes[0].plot(dates, probs[:, regime], lw=1.1, label=f"regime {regime}")
    axes[0].set_ylabel("filtered probability")
    axes[0].legend(ncol=3, frameon=False)
    horizon = 1 if 1 in result["actual"] else result["horizons"][0]
    axes[1].plot(
        dates[: result["actual"][horizon].size], result["actual"][horizon], color="black", lw=1.0
    )
    axes[1].plot(
        dates[: result["regime"][horizon].size], result["regime"][horizon], color="#1f4e79", lw=1.0
    )
    axes[1].set_ylabel("log next-day r^2")
    axes[1].set_xlabel("forecast origin")
    axes[0].set_title(f"{asset_name}: filtered regimes, ordered from low to high intercept")
    if _is_datetime(dates):
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_probs.png", dpi=140)
    plt.close(fig)


def _plot_convergence(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    loglik = result["diagnostics"]["loglik_first"]
    if loglik is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    sweeps = np.arange(1, loglik.shape[1] + 1)
    half = loglik.shape[1] // 2
    for chain in range(loglik.shape[0]):
        ax.plot(sweeps, loglik[chain], lw=1.0, label=f"chain {chain + 1}")
    ax.axvline(half + 1, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("Gibbs sweep")
    ax.set_ylabel("complete-data log likelihood")
    ax.legend(frameon=False)
    rhat = float(result["diagnostics"]["refits"][0]["rhat_max"])
    ax.set_title(f"{asset_name}: first refit, max R-hat {rhat:.3f}")
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_convergence.png", dpi=140)
    plt.close(fig)


def run_dataset(
    name: str,
    sweeps: int | None = None,
    chains: int | None = None,
) -> dict[str, object]:
    """Fit one named dataset and write its figures and scores."""

    spec = DATASETS[name]
    snapshot = load_closes(spec["path"])
    closes = snapshot["closes"]
    returns = np.log(closes[1:] / closes[:-1])
    result = evaluate_returns(
        returns,
        holdout=int(spec["holdout"]),
        train_days=REFIT_YEARS * int(spec["annualization_days"]),
        sweeps=SWEEPS if sweeps is None else sweeps,
        chains=CHAINS if chains is None else chains,
        dates=np.asarray(snapshot["dates"]),
    )
    _print_report(name, result)
    save_regime_plots(spec["prefix"], spec["name"], result, _IMG_DIR)
    print(f"  figures {_IMG_DIR}", flush=True)
    return result


def main() -> None:
    """Run the dataset named in the first argument."""

    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        names = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Usage: python examples/regime_volatility.py <{names}> [sweeps]")
    sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_dataset(sys.argv[1], sweeps=sweeps)


if __name__ == "__main__":
    main()
