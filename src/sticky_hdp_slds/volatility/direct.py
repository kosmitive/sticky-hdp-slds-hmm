"""Direct multi-horizon regime regression.

Each horizon has its own intercept and HAR slopes. Weights are the filtered
regime distribution after one through ``h`` steps of the transition matrix.
An additive bias uses only windows that have already ended.
"""

from __future__ import annotations

import numpy as np

MIN_REGIME_WEIGHT = 30.0


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
