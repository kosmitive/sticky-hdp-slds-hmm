"""Diebold-Mariano tests for multi-step forecast losses."""

from __future__ import annotations

import math

import numpy as np


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
