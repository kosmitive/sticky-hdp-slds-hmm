"""Rolling-origin scores for the sticky regime HAR."""

from __future__ import annotations

import time

import numpy as np

from sticky_hdp_slds.volatility.baselines import (
    direct_har_forecasts,
    ewma_log_forecasts,
    garch_log_forecasts,
    long_run_mean_forecasts,
)
from sticky_hdp_slds.volatility.diebold import diebold_mariano
from sticky_hdp_slds.volatility.direct import (
    fit_direct_regime_coefficients,
    mixture_log_forecast,
    predictive_weights,
    training_bias,
)
from sticky_hdp_slds.volatility.series import (
    LAG,
    expected_self_stay,
    feature_matrix,
    forward_square_mean,
    har_spectral_radius,
    log_squares,
    positive_floor,
)

HORIZONS = (1, 5, 20, 60)
N_REGIMES = 3
PRIOR_ALPHA = 1.0
PRIOR_KAPPA = 97.0
REFIT_YEARS = 4
REFIT_EVERY = 21
SWEEPS = 2000
CHAINS = 4
FORECAST_DRAWS = 40
SEED = 23


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

    from sticky_hdp_slds.inference.regime_har import pack_regime_draws, sample_regime_windows

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

    from sticky_hdp_slds.inference.hmm import filter_regime_draws

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
