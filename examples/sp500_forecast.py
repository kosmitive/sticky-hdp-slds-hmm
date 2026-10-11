"""Forecast 20-day realized volatility with a sticky HDP-SLDS.

Prices are not the observations. A price level trends, so a linear mode
would spend its capacity on that drift. The target is the log of a 20-day
realized volatility, the root mean square of the last 20 daily log returns
``r_t = log(P_t / P_{t-1})``. The log keeps that positive scale roughly
symmetric. Figures also show the same forecast as an annualized volatility
in percent, ``100 * exp(log rv) * sqrt(252)``. The return is the second
observation, not the quantity the forecast is judged on.

Both channels are standardized with the training window only, so the
holdout does not set the units. The GARCH(1,1) Student-t baseline is fit
with ``arch`` on the training returns only.

The snapshot ``examples/data/sp500_daily.csv`` is a Yahoo Finance ``^GSPC``
daily close download. The script does not fetch anything. The last 250
trading days are held out. Priors below were chosen as weakly informative
on standardized channels and were not adjusted to the holdout.

    python examples/sp500_forecast.py
    python examples/sp500_forecast.py 40
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.mniw_prior import MNIWPrior
from model.sticky_hdp_prior import StickyHDPPrior
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel

from examples.pendulum_forecast import roll_forward

SEED = 23
TRUNCATION = 5
VOL_WINDOW = 20
HOLDOUT_DAYS = 250
DEFAULT_SWEEPS = 300
_LOG_VOL_HORIZONS = (1, 5, 20, 60)
_ANNUALIZATION = float(np.sqrt(252.0))
_GARCH_SIMULATIONS = 2000
_GARCH_SEED = 23
_DATA_PATH = Path(__file__).resolve().parent / "data" / "sp500_daily.csv"
_IMG_DIR = _ROOT / "docs" / "img"
_CONTEXT_DAYS = 250


def load_closes(path: Path = _DATA_PATH) -> dict[str, object]:
    """Read the committed close snapshot.

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
    if len(closes) < VOL_WINDOW + HOLDOUT_DAYS + 2:
        raise ValueError(f"{path} does not have enough rows for the holdout.")
    return {
        "dates": dates,
        "closes": np.asarray(closes, dtype=np.float64),
        "source": source,
        "downloaded": downloaded,
    }


def rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    """Root mean square of every trailing window.

    Args:
        values: One-dimensional series.
        window: Positive window length.

    Returns:
        RMS ending at each index ``window - 1`` through the last index.
    """

    squared = np.asarray(values, dtype=np.float64) ** 2
    cumulative = np.cumsum(squared)
    window_sum = cumulative[window - 1 :] - np.concatenate(([0.0], cumulative[:-window]))
    return np.sqrt(window_sum / float(window))


def series_from_prices(
    dates: list[datetime],
    closes: np.ndarray,
    holdout: int = HOLDOUT_DAYS,
    window: int = VOL_WINDOW,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Build standardized return and log-volatility observations.

    ``r_t = log(P_t / P_{t-1})``. The volatility channel at ``t`` is
    ``log`` of the root mean square of ``r`` over ``t`` and the previous
    ``window - 1`` days. Standardization uses the training rows only.

    Args:
        dates: One date per close, the trading-day stamp of ``P_t``.
        closes: Positive close levels.
        holdout: Number of trailing observation days kept out of the fit.
        window: Realized-volatility window.
        seed: Seed recorded for the Gibbs sampler. The transform itself
            does not draw random numbers.

    Returns:
        Training and holdout arrays of shape ``(1, steps, 2)``, the index
        level on those dates, and the training center and scale.
    """

    closes = np.asarray(closes, dtype=np.float64)
    if closes.ndim != 1 or closes.size < window + holdout + 1:
        raise ValueError("Need a longer close series for this holdout.")
    if np.any(closes <= 0.0):
        raise ValueError("Closes must be positive before a log return.")
    returns = np.log(closes[1:] / closes[:-1])
    return_dates = dates[1:]
    rms = rolling_rms(returns, window)
    log_vol = np.log(np.maximum(rms, 1e-12))
    raw = np.column_stack((returns[window - 1 :], log_vol))
    obs_dates = return_dates[window - 1 :]
    # r_t is dated with P_t, which is closes[t_index + 1]. The first
    # complete window ends on return index ``window - 1``, so the price
    # index is ``window``.
    index_level = closes[window:]
    if index_level.shape[0] != raw.shape[0]:
        raise RuntimeError("Index level and observations are misaligned.")
    train_raw = raw[:-holdout]
    hold_raw = raw[-holdout:]
    center = train_raw.mean(axis=0)
    scale = np.maximum(train_raw.std(axis=0), 1e-8)
    train = ((train_raw - center) / scale)[None, :, :]
    hold = ((hold_raw - center) / scale)[None, :, :]
    return {
        "observations_train": train,
        "observations_hold": hold,
        "dates_train": np.asarray(obs_dates[:-holdout]),
        "dates_hold": np.asarray(obs_dates[-holdout:]),
        "index_train": index_level[:-holdout],
        "index_hold": index_level[-holdout:],
        "center": center,
        "scale": scale,
        "seed": np.array(seed),
        "returns": returns,
        "window": np.array(window),
    }


def build_model(observations: np.ndarray) -> StickyHDPSLDSHMMModel:
    """Construct a 5-mode sticky HDP-SLDS for the two standardized channels.

    The matrix-normal mean says that a return is nearly unpredictable and
    that log volatility is persistent. Column precision is ``I``, one
    pseudo-observation, so a few thousand trading days dominate the prior.
    Nothing here was chosen by looking at the holdout.

    Args:
        observations: Shape ``(traces, steps, 2)``.

    Returns:
        A model whose first Gibbs sweep resamples parameters and states.
    """

    latent_dim = 2
    mean_dynamics = np.array([[0.0, 0.0], [0.0, 0.8]])
    dynamics_prior = MNIWPrior(
        mean_dynamics,
        np.eye(latent_dim),
        6,
        0.2 * np.eye(latent_dim),
    )
    # Mean alpha+kappa = 8, mean rho = 0.8, mean gamma = 2.
    hdp_prior = StickyHDPPrior(
        TRUNCATION,
        [8.0, 1.0],
        [12.0, 3.0],
        [2.0, 1.0],
        "uniform",
    )
    emission_scale = 0.3 * np.eye(latent_dim)
    model = StickyHDPSLDSHMMModel(
        latent_dim,
        latent_dim,
        hdp_prior,
        dynamics_prior,
        np.eye(latent_dim),
        [6.0, emission_scale],
        np.eye(latent_dim),
    )
    model.init_params()
    model.y = np.array(observations, dtype=np.float64, copy=True)
    model.x = np.array(observations, dtype=np.float64, copy=True)
    model.slds_init_z()
    return model


def _relabel_by_volatility(modes: np.ndarray, log_vol: np.ndarray) -> np.ndarray:
    """Order one mode sequence by mean volatility so a figure is readable.

    Labels permute across sweeps, so this map is for that one sequence only.

    Args:
        modes: Shape ``(traces, steps)``.
        log_vol: Volatility channel with the same time axis.

    Returns:
        A copy of ``modes`` sorted by increasing mean volatility.
    """

    present = np.unique(modes)
    centers = [float(np.mean(log_vol[modes == mode])) for mode in present]
    order = present[np.argsort(centers)]
    relabeled = np.empty_like(modes)
    for new_id, old_id in enumerate(order):
        relabeled[modes == old_id] = new_id
    return relabeled


def fit_and_forecast(series: dict[str, np.ndarray], sweeps: int) -> dict[str, np.ndarray]:
    """Fit the training window, then forecast the holdout from the second half.

    Forecast draws run after the Gibbs loop so they do not consume the
    sampler's random stream.

    Args:
        series: Output of :func:`series_from_prices`.
        sweeps: Number of Gibbs sweeps. Forecasts use the second half.

    Returns:
        Log joint, occupied modes, the last mode sequence, and predictive
        futures of shape ``(samples, 1, horizon, 2)``.
    """

    train = series["observations_train"]
    hold = series["observations_hold"]
    horizon = int(hold.shape[1])
    np.random.seed(int(series["seed"]))
    model = build_model(train)
    half = sweeps // 2
    log_joint = np.empty(sweeps)
    used_modes = np.empty(sweeps, dtype=np.int64)
    snapshots: list[dict[str, np.ndarray]] = []
    started = time.perf_counter()
    for sweep in range(sweeps):
        model.gibbs_step()
        log_joint[sweep] = model.calc_log_likelihood()
        used_modes[sweep] = np.unique(model.z).size
        if sweep >= half:
            snapshots.append(
                {
                    "x_last": np.array(model.x[:, -1], copy=True),
                    "z_last": np.array(model.z[:, -1], copy=True),
                    "A": np.array(model.dynamics.A, copy=True),
                    "Sigma": np.array(model.dynamics.Sigma, copy=True),
                    "pi": np.array(model.hdp.pi, copy=True),
                    "R": np.array(model.R, copy=True),
                    "C": np.array(model.C, copy=True),
                }
            )
        if sweep == 0 or (sweep + 1) % 25 == 0 or sweep + 1 == sweeps:
            print(
                f"sweep {sweep + 1:4d}/{sweeps}  "
                f"log_joint {log_joint[sweep]:12.2f}  "
                f"modes {used_modes[sweep]}  "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    failures = [0]
    futures = np.empty((len(snapshots), train.shape[0], horizon, model.od))
    for index, snapshot in enumerate(snapshots):
        futures[index] = roll_forward(snapshot, horizon, failures)
    final_modes = _relabel_by_volatility(model.z, train[:, :, 1])
    elapsed = time.perf_counter() - started
    return {
        "observations_train": train,
        "observations_hold": hold,
        "dates_train": series["dates_train"],
        "dates_hold": series["dates_hold"],
        "index_train": series["index_train"],
        "futures": futures,
        "log_joint": log_joint,
        "used_modes": used_modes,
        "modes_last": final_modes,
        "elapsed_sec": np.array(elapsed),
        "gaussian_jitter_draws": np.array(failures[0]),
        "center": series["center"],
        "scale": series["scale"],
        "window": series["window"],
    }


def ewma_variance_forecast(returns: np.ndarray, last_known: int, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics variance forecast that does not read past ``last_known``.

    ``sigma2[0]`` is the mean square of the returns through ``last_known``.
    After that, ``sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2``
    while ``t - 1`` is still known. Later days stay at the last forecast,
    which is the EWMA multi-step forecast because the weights sum to one.

    Args:
        returns: Daily log returns.
        last_known: Last return index the forecast may use.
        lam: Decay. 0.94 is the RiskMetrics daily value.

    Returns:
        A variance forecast for every return index.
    """

    sigma2 = np.empty(returns.shape[0], dtype=np.float64)
    history = returns[: last_known + 1]
    sigma2[0] = float(np.mean(history**2))
    limit = min(last_known + 1, returns.size - 1)
    for index in range(1, limit + 1):
        sigma2[index] = lam * sigma2[index - 1] + (1.0 - lam) * returns[index - 1] ** 2
    if limit < returns.size:
        sigma2[limit:] = sigma2[limit]
    return sigma2


def _fit_arch_garch(returns: np.ndarray):
    """Fit a zero-mean GARCH(1,1) Student-t model with ``arch``.

    ``rescale=True`` multiplies the returns by ``result.scale`` (100 for
    these daily log returns) so the optimizer sees order-1 data. ``omega``
    below is divided back by ``scale**2``. ``alpha``, ``beta``, and ``nu``
    are invariant to that scale. ``arch`` constrains ``alpha + beta < 1``.

    Args:
        returns: Log returns the fit is allowed to see.

    Returns:
        The ``arch`` result. Parameters in ``result.params`` are still in
        the rescaled units.
    """

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
    # arch keeps alpha + beta below 1, but the sum can land a rounding
    # step above 1. A slack of 1e-6 still rejects a genuinely explosive fit.
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

    The estimates are in the original return units. ``sigma2_last`` is the
    filtered variance of the last return. ``success`` is 1 when ``arch``
    reports convergence, the variance is finite, and ``nu > 2``.

    Args:
        returns: Training log returns.

    Returns:
        Point estimates. A failed fit still returns the numbers ``arch``
        produced, with ``success`` 0.
    """

    return _garch_params(_fit_arch_garch(returns))


def garch_variance_leads(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Analytic GARCH variance of each future return, in original units.

    Lead ``h`` (1-based) is the variance of the return ``h`` days after the
    last entry of ``returns``:

    ``v_1 = omega + alpha * r_T^2 + beta * sigma_T^2``

    ``v_h = omega / (1 - alpha - beta) + (alpha + beta)^(h - 1) * (v_1 - omega / (1 - alpha - beta))``

    ``arch`` returns those variances in rescaled units. This divides by
    ``scale**2``.

    Args:
        returns: Returns the fit may use. The forecast origin is the last one.
        horizon: Number of leads.

    Returns:
        Array of length ``horizon``. Entry ``h - 1`` is the lead-``h`` variance.
    """

    fitted = _fit_arch_garch(returns)
    scale = float(fitted.scale)
    forecast = fitted.forecast(horizon=horizon, method="analytic", reindex=False)
    leads = np.asarray(forecast.variance, dtype=np.float64)[-1] / scale**2
    return np.maximum(leads, 1e-18)


def _predicted_log_rv(
    returns: np.ndarray,
    variance: np.ndarray,
    window: int,
    first_holdout: int,
    horizon: int,
    last_known: int,
) -> np.ndarray:
    """Log 20-day RMS, mixing known squares with a variance forecast.

    Holdout step ``s`` is the log root mean square of returns
    ``first_holdout + s`` through ``first_holdout + s + window - 1``.
    That window is the same trailing window as the SLDS volatility target.
    An index at or before ``last_known`` contributes the realized square.
    A later index contributes ``variance`` at that index, which must already
    be the forecast of that return's variance.

    Args:
        returns: Full return series, including the holdout.
        variance: Forecast variance aligned to ``returns``.
        window: Realized-volatility window.
        first_holdout: Return index where the first holdout window starts.
        horizon: Number of holdout steps.
        last_known: Last return index the forecast may read.

    Returns:
        Log realized-volatility forecast, length ``horizon``.
    """

    predicted = np.empty(horizon, dtype=np.float64)
    for step in range(horizon):
        start = first_holdout + step
        squares = np.empty(window, dtype=np.float64)
        for offset in range(window):
            index = start + offset
            if index <= last_known:
                squares[offset] = returns[index] ** 2
            else:
                squares[offset] = variance[index]
        predicted[step] = np.log(max(float(np.sqrt(np.mean(squares))), 1e-12))
    return predicted


def _simulated_log_rv(
    returns: np.ndarray,
    future_returns: np.ndarray,
    window: int,
    first_holdout: int,
    horizon: int,
    last_known: int,
) -> np.ndarray:
    """Log 20-day RMS on each simulated path.

    ``future_returns`` has shape ``(simulations, leads)``. Column ``h - 1``
    is a draw of the return at index ``last_known + h``. Known returns inside
    a window stay at their realized values, so the band does not redraw the
    training sample.

    Args:
        returns: Full return series.
        future_returns: Simulated returns after ``last_known``.
        window: Realized-volatility window.
        first_holdout: Return index where the first holdout window starts.
        horizon: Number of holdout steps.
        last_known: Last return index treated as known.

    Returns:
        Array of shape ``(simulations, horizon)``.
    """

    draws = future_returns.shape[0]
    log_paths = np.empty((draws, horizon), dtype=np.float64)
    for step in range(horizon):
        start = first_holdout + step
        squares = np.empty((draws, window), dtype=np.float64)
        for offset in range(window):
            index = start + offset
            if index <= last_known:
                squares[:, offset] = returns[index] ** 2
            else:
                squares[:, offset] = future_returns[:, index - last_known - 1] ** 2
        rms = np.sqrt(np.mean(squares, axis=1))
        log_paths[:, step] = np.log(np.maximum(rms, 1e-12))
    return log_paths


def _garch_log_rv_forecast(
    returns: np.ndarray,
    last_known: int,
    window: int,
    first_holdout: int,
    horizon: int,
    simulations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Point forecast and 90% band of log realized volatility from one GARCH fit.

    The point forecast plugs the analytic variance leads into
    :func:`_predicted_log_rv`. The band simulates Student-t return paths
    from the same fit and takes the 5th and 95th percentiles of the log
    root mean square. Both use only ``returns[: last_known + 1]``.

    Args:
        returns: Full return series. Entries after ``last_known`` are ignored.
        last_known: Last return index the fit may use.
        window: Realized-volatility window.
        first_holdout: Return index where the first holdout window starts.
        horizon: Number of holdout steps.
        simulations: Simulated paths in the band.
        seed: Seed for those paths.

    Returns:
        Point forecast, lower band, upper band, and the parameter dict.
        The three arrays are raw log realized volatility.
    """

    history = returns[: last_known + 1]
    fitted = _fit_arch_garch(history)
    params = _garch_params(fitted)
    empty = np.full(horizon, np.nan)
    if params["success"] != 1.0:
        return empty, empty.copy(), empty.copy(), params
    scale = float(fitted.scale)
    analytic = fitted.forecast(horizon=horizon, method="analytic", reindex=False)
    leads = np.asarray(analytic.variance, dtype=np.float64)[-1] / scale**2
    needed = returns.shape[0] - (last_known + 1)
    if int(leads.shape[0]) < needed:
        raise RuntimeError("GARCH variance forecast is shorter than the holdout windows.")
    variance = np.empty(returns.shape[0], dtype=np.float64)
    for index in range(last_known + 1, returns.shape[0]):
        variance[index] = leads[index - last_known - 1]
    point = _predicted_log_rv(returns, variance, window, first_holdout, horizon, last_known)
    # arch uses ``rng(size)`` for simulation paths. ``random_state`` only
    # seeds the bootstrap method, so the band draws standardized Student-t
    # shocks here. Dividing by sqrt(nu / (nu - 2)) gives variance 1, which
    # is the scale arch's volatility recursion expects.
    nu = float(params["nu"])
    draws = np.random.RandomState(seed)

    def student_rng(size: tuple[int, ...]) -> np.ndarray:
        shocks = draws.standard_t(nu, size=size)
        return shocks / np.sqrt(nu / (nu - 2.0))

    simulated = fitted.forecast(
        horizon=horizon,
        method="simulation",
        simulations=simulations,
        rng=student_rng,
        reindex=False,
    )
    future = np.asarray(simulated.simulations.values, dtype=np.float64)[0] / scale
    log_paths = _simulated_log_rv(returns, future, window, first_holdout, horizon, last_known)
    lower = np.percentile(log_paths, 5.0, axis=0)
    upper = np.percentile(log_paths, 95.0, axis=0)
    return point, lower, upper, params


def volatility_baseline_paths(
    series: dict[str, np.ndarray],
    simulations: int = _GARCH_SIMULATIONS,
    seed: int = _GARCH_SEED,
) -> dict[str, np.ndarray]:
    """Trailing, EWMA(0.94), and GARCH(1,1)-t forecasts of log realized volatility.

    All three point forecasts are standardized with the training center and
    scale of the volatility channel. The trailing forecast holds the last
    training 20-day value flat. EWMA and GARCH see training returns only.
    The GARCH point forecast is the log root mean square of the analytic
    variance forecast. ``garch_lower`` and ``garch_upper`` are the 5th and
    95th percentiles of simulated log root mean squares, standardized the
    same way. The first ``window - 1`` holdout windows still contain known
    returns.

    Args:
        series: Output of :func:`series_from_prices`.
        simulations: Paths in the GARCH band.
        seed: Seed for those paths.

    Returns:
        Holdout paths and the GARCH point estimates.
    """

    returns = np.asarray(series["returns"], dtype=np.float64)
    window = int(series["window"])
    train_steps = int(series["observations_train"].shape[1])
    horizon = int(series["observations_hold"].shape[1])
    last_known = train_steps + window - 2
    center = float(series["center"][1])
    vol_scale = float(series["scale"][1])
    ewma = ewma_variance_forecast(returns, last_known, 0.94)
    ewma_log = _predicted_log_rv(returns, ewma, window, train_steps, horizon, last_known)
    garch_log, garch_lower, garch_upper, garch = _garch_log_rv_forecast(
        returns, last_known, window, train_steps, horizon, simulations, seed
    )
    flat = np.full(horizon, float(series["observations_train"][0, -1, 1]))

    def standardize(log_rv: np.ndarray) -> np.ndarray:
        return (log_rv - center) / vol_scale

    return {
        "flat": flat,
        "ewma": standardize(ewma_log),
        "garch": standardize(garch_log),
        "garch_lower": standardize(garch_lower),
        "garch_upper": standardize(garch_upper),
        "garch_omega": np.array(garch["omega"]),
        "garch_alpha": np.array(garch["alpha"]),
        "garch_beta": np.array(garch["beta"]),
        "garch_nu": np.array(garch["nu"]),
        "garch_persistence": np.array(garch["persistence"]),
        "garch_success": np.array(garch["success"]),
    }


def log_vol_winners(metrics: dict[str, float]) -> dict[int, str]:
    """Name the lowest log-volatility RMSE at each reported lead.

    A tie keeps every method that matches the minimum. The names are
    ``SLDS``, ``trailing``, ``EWMA``, and ``GARCH``.

    Args:
        metrics: Output of :func:`forecast_metrics`.

    Returns:
        Map from lead time to the winning name or names.
    """

    labels = (("model", "SLDS"), ("flat", "trailing"), ("ewma", "EWMA"), ("garch", "GARCH"))
    winners: dict[int, str] = {}
    for horizon in _LOG_VOL_HORIZONS:
        scores = []
        for name, label in labels:
            key = f"rmse_log_vol_{name}_h{horizon}"
            if key not in metrics:
                continue
            scores.append((label, metrics[key]))
        if not scores:
            continue
        best = min(score for _, score in scores)
        tied = [label for label, score in scores if np.isclose(score, best)]
        winners[horizon] = " and ".join(tied)
    return winners


def raw_log_rv(standardized: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Undo training standardization of the log realized-volatility channel."""

    return np.asarray(standardized, dtype=np.float64) * scale + center


def annualized_vol_percent(log_rv: np.ndarray) -> np.ndarray:
    """Map log 20-day RMS volatility to an annualized percent.

    The 20-day root mean square is a daily volatility. Multiplying by
    ``sqrt(252)`` is the usual trading-day annualization.

    Args:
        log_rv: Log of the 20-day root mean square of daily log returns.

    Returns:
        Annualized volatility in percent.
    """

    return 100.0 * np.exp(np.asarray(log_rv, dtype=np.float64)) * _ANNUALIZATION


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Root mean square error over every element."""

    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def forecast_metrics(
    result: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Score log realized volatility, and record the return forecast in passing.

    Volatility RMSE is on the raw log of 20-day realized volatility, not the
    standardized units. Horizons 1, 5, 20, and 60 are the RMSE of that path
    through the first that many holdout days. One series means the 1-day
    value is a single absolute error. The flat path holds the last training
    20-day value fixed. EWMA uses decay 0.94. The return numbers are kept
    so a reader can see that channel is close to unpredictable.

    Args:
        result: Output of :func:`fit_and_forecast`.
        baselines: Output of :func:`volatility_baseline_paths`, if available.

    Returns:
        RMSE, empirical 90% coverage, and the second-half log-joint summary.
    """

    futures = result["futures"]
    hold = result["observations_hold"]
    predictive = np.mean(futures, axis=0)
    lower = np.percentile(futures, 5.0, axis=0)
    upper = np.percentile(futures, 95.0, axis=0)
    inside = (hold >= lower) & (hold <= upper)
    last = result["observations_train"][:, -1, :]
    ret_hold = hold[0, :, 0]
    ret_hat = predictive[0, :, 0]
    vol_hold = hold[0, :, 1]
    vol_hat = predictive[0, :, 1]
    zero = np.zeros_like(ret_hold)
    last_return = np.full_like(ret_hold, last[0, 0])
    flat_vol = np.full_like(vol_hold, last[0, 1])
    half = max(result["log_joint"].size // 2, 1)
    second = result["log_joint"][half:]
    centered = np.arange(second.size, dtype=np.float64)
    centered -= centered.mean()
    denom = float(np.dot(centered, centered))
    slope = float(np.dot(centered, second - second.mean()) / denom) if denom > 0.0 else 0.0
    metrics = {
        "retained_sweeps": float(futures.shape[0]),
        "rmse_return_model": _rmse(ret_hat, ret_hold),
        "rmse_return_zero": _rmse(zero, ret_hold),
        "rmse_return_last": _rmse(last_return, ret_hold),
        "coverage_return": float(np.mean(inside[0, :, 0])),
        "rmse_vol_model": _rmse(vol_hat, vol_hold),
        "rmse_vol_flat": _rmse(flat_vol, vol_hold),
        "coverage_vol": float(np.mean(inside[0, :, 1])),
        "coverage_pooled": float(np.mean(inside)),
        "gaussian_jitter_draws": float(result["gaussian_jitter_draws"]),
        "log_joint_start": float(result["log_joint"][0]),
        "log_joint_end": float(result["log_joint"][-1]),
        "log_joint_second_mean": float(np.mean(second)),
        "log_joint_second_std": float(np.std(second)),
        "log_joint_second_slope": slope,
        "modes_second_mean": float(np.mean(result["used_modes"][half:])),
        "modes_last": float(result["used_modes"][-1]),
    }
    # The first ``window - 1`` holdout days still contain known returns.
    overlap = min(int(result["window"]) - 1, vol_hold.size - 1)
    metrics["rmse_vol_model_after_overlap"] = _rmse(vol_hat[overlap:], vol_hold[overlap:])
    metrics["rmse_vol_flat_after_overlap"] = _rmse(flat_vol[overlap:], vol_hold[overlap:])
    center = float(result["center"][1])
    scale = float(result["scale"][1])
    log_hold = raw_log_rv(vol_hold, center, scale)
    log_paths = {
        "model": raw_log_rv(vol_hat, center, scale),
        "flat": raw_log_rv(flat_vol, center, scale),
    }
    if baselines is not None:
        for name in ("ewma", "garch"):
            path = np.asarray(baselines[name], dtype=np.float64)
            metrics[f"rmse_vol_{name}"] = _rmse(path, vol_hold)
            metrics[f"rmse_vol_{name}_after_overlap"] = _rmse(path[overlap:], vol_hold[overlap:])
            log_paths[name] = raw_log_rv(path, center, scale)
        metrics["garch_omega"] = float(baselines["garch_omega"])
        metrics["garch_alpha"] = float(baselines["garch_alpha"])
        metrics["garch_beta"] = float(baselines["garch_beta"])
        metrics["garch_nu"] = float(baselines["garch_nu"])
        metrics["garch_persistence"] = float(baselines["garch_persistence"])
        metrics["garch_success"] = float(baselines["garch_success"])
        if "garch_lower" in baselines:
            lower_band = np.asarray(baselines["garch_lower"], dtype=np.float64)
            upper_band = np.asarray(baselines["garch_upper"], dtype=np.float64)
            inside_garch = (vol_hold >= lower_band) & (vol_hold <= upper_band)
            metrics["coverage_garch"] = float(np.mean(inside_garch))
            for horizon in _LOG_VOL_HORIZONS:
                if horizon > vol_hold.size:
                    continue
                metrics[f"coverage_garch_h{horizon}"] = float(np.mean(inside_garch[:horizon]))
    for name, path in log_paths.items():
        metrics[f"rmse_log_vol_{name}"] = _rmse(path, log_hold)
        for horizon in _LOG_VOL_HORIZONS:
            if horizon > log_hold.size:
                continue
            metrics[f"rmse_log_vol_{name}_h{horizon}"] = _rmse(path[:horizon], log_hold[:horizon])
    for horizon in _LOG_VOL_HORIZONS:
        if horizon > log_hold.size:
            continue
        metrics[f"coverage_vol_h{horizon}"] = float(np.mean(inside[0, :horizon, 1]))
    # Labels are sorted by mean volatility, so the order is for display.
    # The spread of these means is whether that draw separates regimes.
    modes = result["modes_last"][0]
    train_ann = annualized_vol_percent(
        raw_log_rv(result["observations_train"][0, :, 1], center, scale)
    )
    for mode in np.unique(modes):
        selected = modes == mode
        metrics[f"mode_{int(mode)}_ann_vol_pct"] = float(np.mean(train_ann[selected]))
        metrics[f"mode_{int(mode)}_share"] = float(np.mean(selected))
    return metrics


def save_plots(
    result: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    destination: Path,
    prefix: str,
    asset_name: str,
) -> None:
    """Write convergence, mode, forecast, and lead-time figures.

    Args:
        result: Output of :func:`fit_and_forecast`.
        baselines: Output of :func:`volatility_baseline_paths`.
        destination: Directory for the PNG files.
        prefix: Filename stem, such as ``sp500`` or ``btc``.
        asset_name: Label used in the figure titles.
    """

    destination.mkdir(parents=True, exist_ok=True)
    sweeps = np.arange(1, result["log_joint"].size + 1)
    half = result["log_joint"].size // 2
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axes[0].plot(sweeps, result["log_joint"], color="#1f4e79", lw=1.2)
    axes[0].axvline(half + 1, color="#888888", ls="--", lw=0.8)
    axes[0].set_ylabel("complete-data log joint")
    axes[1].plot(sweeps, result["used_modes"], color="#b85c38", lw=1.2)
    axes[1].set_ylabel("occupied modes")
    axes[1].set_xlabel("Gibbs sweep")
    axes[1].set_ylim(0.5, TRUNCATION + 0.5)
    fig.suptitle(f"Sticky HDP-SLDS on {asset_name} returns and log volatility")
    fig.tight_layout()
    fig.savefig(destination / f"{prefix}_convergence.png", dpi=140)
    plt.close(fig)

    dates = result["dates_train"]
    modes = result["modes_last"][0]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True)
    axes[0].plot(dates, result["index_train"], color="#d0d0d0", lw=0.6)
    axes[0].scatter(dates, result["index_train"], c=modes, cmap="viridis", s=3, linewidths=0)
    axes[0].set_ylabel(f"{asset_name} close")
    axes[1].step(dates, modes, where="post", color="#1f4e79", lw=0.8)
    axes[1].set_ylabel("mode, last sweep")
    axes[1].set_xlabel("date")
    axes[1].set_ylim(-0.2, TRUNCATION - 0.8)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Last sweep, modes ordered from calm to turbulent")
    fig.tight_layout()
    fig.savefig(destination / f"{prefix}_modes.png", dpi=140)
    plt.close(fig)

    futures = result["futures"]
    predictive = np.mean(futures, axis=0)[0]
    lower = np.percentile(futures, 5.0, axis=0)[0]
    upper = np.percentile(futures, 95.0, axis=0)[0]
    context = min(_CONTEXT_DAYS, result["dates_train"].shape[0])
    train_dates = result["dates_train"][-context:]
    hold_dates = result["dates_hold"]
    center = float(result["center"][1])
    scale = float(result["scale"][1])
    train_log = raw_log_rv(result["observations_train"][0, -context:, 1], center, scale)
    hold_log = raw_log_rv(result["observations_hold"][0, :, 1], center, scale)
    mean_log = raw_log_rv(predictive[:, 1], center, scale)
    lower_log = raw_log_rv(lower[:, 1], center, scale)
    upper_log = raw_log_rv(upper[:, 1], center, scale)
    flat_log = raw_log_rv(baselines["flat"], center, scale)
    ewma_log = raw_log_rv(baselines["ewma"], center, scale)
    garch_log = raw_log_rv(baselines["garch"], center, scale)
    garch_lo = raw_log_rv(baselines["garch_lower"], center, scale)
    garch_hi = raw_log_rv(baselines["garch_upper"], center, scale)
    panels = (
        (
            "annualized volatility (%)",
            annualized_vol_percent(train_log),
            annualized_vol_percent(hold_log),
            annualized_vol_percent(mean_log),
            annualized_vol_percent(lower_log),
            annualized_vol_percent(upper_log),
            annualized_vol_percent(flat_log),
            annualized_vol_percent(ewma_log),
            annualized_vol_percent(garch_log),
            annualized_vol_percent(garch_lo),
            annualized_vol_percent(garch_hi),
        ),
        (
            "log 20-day realized volatility",
            train_log,
            hold_log,
            mean_log,
            lower_log,
            upper_log,
            flat_log,
            ewma_log,
            garch_log,
            garch_lo,
            garch_hi,
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True)
    for index, panel in enumerate(panels):
        axis = axes[index]
        (
            name,
            train,
            hold,
            mean,
            band_low,
            band_high,
            flat,
            ewma,
            garch,
            garch_low,
            garch_high,
        ) = panel
        axis.plot(train_dates, train, color="#9aa0a6", lw=0.8, label="fit window")
        axis.plot(hold_dates, hold, color="#333333", lw=1.1, label="realized holdout")
        axis.fill_between(
            hold_dates,
            band_low,
            band_high,
            color="#1f4e79",
            alpha=0.18,
            label="SLDS 90% band",
        )
        axis.plot(hold_dates, mean, color="#1f4e79", lw=1.3, label="SLDS mean")
        axis.fill_between(
            hold_dates,
            garch_low,
            garch_high,
            color="#E45756",
            alpha=0.12,
            label="GARCH 90% band",
        )
        axis.plot(hold_dates, garch, color="#E45756", lw=1.1, ls="-.", label="GARCH-t")
        axis.plot(hold_dates, flat, color="#666666", lw=1.0, ls=":", label="trailing 20-day")
        axis.plot(hold_dates, ewma, color="#F58518", lw=1.1, ls="--", label="EWMA 0.94")
        axis.axvline(hold_dates[0], color="#888888", lw=0.8)
        axis.set_ylabel(name)
    axes[0].legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    axes[1].set_xlabel("date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"{asset_name} volatility forecast, 250-day holdout")
    fig.tight_layout()
    fig.savefig(destination / f"{prefix}_forecast.png", dpi=140)
    plt.close(fig)

    lead = np.arange(1, hold_log.size + 1)
    count = lead.astype(np.float64)

    def expanding_rmse(error: np.ndarray) -> np.ndarray:
        return np.sqrt(np.cumsum(error**2) / count)

    inside_vol = (hold_log >= lower_log) & (hold_log <= upper_log)
    inside_garch = (hold_log >= garch_lo) & (hold_log <= garch_hi)
    # One series makes a single-day coverage bit a barcode. A 20-day
    # trailing fraction is the same indicator, smoothed. The prefix is
    # the expanding fraction until the window is full.
    window = min(20, hold_log.size)
    vol_cover = np.cumsum(inside_vol.astype(np.float64)) / count
    garch_cover = np.cumsum(inside_garch.astype(np.float64)) / count
    if hold_log.size >= window:
        kernel = np.ones(window) / float(window)
        vol_cover[window - 1 :] = np.convolve(inside_vol.astype(np.float64), kernel, mode="valid")
        garch_cover[window - 1 :] = np.convolve(
            inside_garch.astype(np.float64), kernel, mode="valid"
        )
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    axes[0].plot(
        lead,
        expanding_rmse(mean_log - hold_log),
        color="#1f4e79",
        lw=1.4,
        label="model",
    )
    axes[0].plot(
        lead,
        expanding_rmse(flat_log - hold_log),
        color="#666666",
        lw=1.1,
        ls=":",
        label="trailing 20-day",
    )
    axes[0].plot(
        lead,
        expanding_rmse(ewma_log - hold_log),
        color="#F58518",
        lw=1.1,
        ls="--",
        label="EWMA 0.94",
    )
    axes[0].plot(
        lead,
        expanding_rmse(garch_log - hold_log),
        color="#E45756",
        lw=1.1,
        ls="-.",
        label="GARCH-t",
    )
    for horizon in _LOG_VOL_HORIZONS:
        if horizon <= hold_log.size:
            axes[0].axvline(horizon, color="#cccccc", lw=0.7)
    axes[0].set_ylabel("RMSE of log volatility")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(lead, vol_cover, color="#1f4e79", lw=1.2, label="SLDS, 20-day coverage")
    axes[1].plot(
        lead, garch_cover, color="#E45756", lw=1.1, ls="-.", label="GARCH, 20-day coverage"
    )
    axes[1].axhline(0.90, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel("fraction in 90% band")
    axes[1].set_xlabel("lead time (days)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Log-volatility RMSE through each lead, one series")
    fig.tight_layout()
    fig.savefig(destination / f"{prefix}_forecast_horizon.png", dpi=140)
    plt.close(fig)


def run_forecast(path: Path, prefix: str, asset_name: str, sweeps: int) -> None:
    """Fit one committed close snapshot and write its figures and scores.

    Args:
        path: CSV snapshot.
        prefix: Filename stem for the figures.
        asset_name: Name used in the log and figure titles.
        sweeps: Gibbs sweeps. Forecasts use the second half.
    """

    snapshot = load_closes(path)
    series = series_from_prices(snapshot["dates"], snapshot["closes"])
    train_dates = series["dates_train"]
    hold_dates = series["dates_hold"]
    print(
        f"{prefix} forecast  seed {int(series['seed'])}  "
        f"downloaded {snapshot['downloaded']}  "
        f"train {train_dates[0]:%Y-%m-%d}..{train_dates[-1]:%Y-%m-%d} "
        f"n {train_dates.size}  "
        f"holdout {hold_dates[0]:%Y-%m-%d}..{hold_dates[-1]:%Y-%m-%d} "
        f"n {hold_dates.size}  sweeps {sweeps}",
        flush=True,
    )
    print(f"  source {snapshot['source']}", flush=True)
    baselines = volatility_baseline_paths(series)
    result = fit_and_forecast(series, sweeps)
    metrics = forecast_metrics(result, baselines)
    save_plots(result, baselines, _IMG_DIR, prefix, asset_name)
    print("forecast")
    for key, value in metrics.items():
        print(f"  {key:28s} {value:.6g}" if isinstance(value, float) else f"  {key:28s} {value}")
    print(f"  elapsed_sec                  {float(result['elapsed_sec']):.1f}")
    print(f"  figures                      {_IMG_DIR}")
    print("winner, lowest RMSE of log volatility")
    for horizon, name in log_vol_winners(metrics).items():
        print(f"  lead {horizon:<4d} {name}")


def main() -> None:
    """Fit the committed S&P 500 snapshot and score the 250-day holdout."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEPS
    run_forecast(_DATA_PATH, "sp500", "S&P 500", sweeps)


if __name__ == "__main__":
    main()
