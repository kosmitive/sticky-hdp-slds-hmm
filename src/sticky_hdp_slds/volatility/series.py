"""Daily log-variance features for the sticky regime HAR."""

from __future__ import annotations

import numpy as np

LAG = 22
WEEK = 5


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
