"""Forecast the pendulum past the window used to fit the sticky HDP-SLDS.

The training series and the Gibbs seed are the ones in ``pendulum_demo``.
Eighty further steps are integrated with a fixed continuation seed and are
never passed to the sampler. Hyperparameters are unchanged. The holdout is
not used to choose them.

For every sweep in the second half of the chain, one future is drawn from
that sweep's sampled terminal state, terminal mode, dynamics, sticky
transition matrix, and emission covariance. The draws are aggregated into a
predictive mean and a central 90% band.

    python examples/pendulum_forecast.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.pendulum_demo import (
    DEFAULT_SWEEPS,
    DT,
    RELEASE_ANGLES,
    SEED,
    STEP_COUNT,
    build_model,
    continue_pendulum,
    gibbs_sweep,
    simulate_pendulum,
)

HOLDOUT_SEED = SEED + 1
HORIZON = 80
SHORT_HORIZON = 10
_IMG_DIR = _ROOT / "docs" / "img"
_GAUSSIAN_JITTER = 1e-10


def _sample_gaussian(mean: np.ndarray, covariance: np.ndarray, failures: list[int]) -> np.ndarray:
    """Draw one Gaussian vector, symmetrizing the covariance first.

    Args:
        mean: Mean vector.
        covariance: Covariance matrix, possibly a hair away from symmetric.
        failures: Mutable counter of draws that needed a diagonal jitter.

    Returns:
        One sample with the same shape as ``mean``.
    """

    symmetric = 0.5 * (covariance + covariance.T)
    try:
        return np.random.multivariate_normal(mean, symmetric)
    except np.linalg.LinAlgError:
        failures[0] += 1
        jittered = symmetric + _GAUSSIAN_JITTER * np.eye(symmetric.shape[0])
        return np.random.multivariate_normal(mean, jittered)


def roll_forward(snapshot: dict[str, np.ndarray], horizon: int, failures: list[int]) -> np.ndarray:
    """Sample one future observation path for every trace.

    The terminal latent state and mode are that sweep's Gibbs draw. The next
    mode is drawn from that mode's row of ``pi``, and the latent state follows
    ``x <- A_z x + noise``. Observations add a fresh emission from ``R``.

    Args:
        snapshot: Copies of ``x_last``, ``z_last``, ``A``, ``Sigma``, ``pi``,
            ``R``, and ``C`` taken after a finished sweep.
        horizon: Number of future steps.
        failures: Counter of covariance draws that needed jitter.

    Returns:
        Future observations, shape ``(traces, horizon, observation_dim)``.
    """

    dynamics = snapshot["A"]
    process_cov = snapshot["Sigma"]
    transition = np.asarray(snapshot["pi"], dtype=np.float64)
    emission_cov = np.asarray(snapshot["R"], dtype=np.float64)
    projection = np.asarray(snapshot["C"], dtype=np.float64)
    latent = np.array(snapshot["x_last"], dtype=np.float64, copy=True)
    mode = np.array(snapshot["z_last"], dtype=np.int64, copy=True)
    trace_count = latent.shape[0]
    mode_count = transition.shape[0]
    future = np.empty((trace_count, horizon, projection.shape[0]))
    for step in range(horizon):
        for trace in range(trace_count):
            probs = np.clip(transition[mode[trace]], 0.0, None)
            total = float(probs.sum())
            if total <= 0.0:
                probs = np.full(mode_count, 1.0 / mode_count)
            else:
                probs /= total
            mode[trace] = int(np.random.choice(mode_count, p=probs))
            mean_latent = dynamics[mode[trace]] @ latent[trace]
            latent[trace] = _sample_gaussian(mean_latent, process_cov[mode[trace]], failures)
            mean_obs = projection @ latent[trace]
            future[trace, step] = _sample_gaussian(mean_obs, emission_cov, failures)
    return future


def fit_and_forecast(sweeps: int, horizon: int) -> dict[str, np.ndarray]:
    """Fit the demo series, then forecast ``horizon`` steps from the second half.

    Args:
        sweeps: Number of Gibbs sweeps. Forecasts use the second half.
        horizon: Held-out length.

    Returns:
        Training history, posterior-predictive futures, and the held-out series.
    """

    training = simulate_pendulum(SEED)
    held_out = continue_pendulum(training["state"], horizon, HOLDOUT_SEED)
    np.random.seed(SEED)
    model = build_model(training["observations"])
    half = sweeps // 2
    log_joint = np.empty(sweeps)
    used_modes = np.empty(sweeps, dtype=np.int64)
    latent_sum = np.zeros_like(model.x)
    snapshots: list[dict[str, np.ndarray]] = []
    started = time.perf_counter()
    for sweep in range(sweeps):
        gibbs_sweep(model)
        log_joint[sweep] = model.calc_log_likelihood()
        used_modes[sweep] = np.unique(model.z).size
        if sweep >= half:
            # Copies only. Forecast draws happen after the chain so they do
            # not consume the sampler's random stream.
            latent_sum += model.x
            snapshots.append(
                {
                    "x_last": np.array(model.x[:, -1], copy=True),
                    "z_last": np.array(model.z[:, -1], copy=True),
                    "A": np.array(model.slds_params[0], copy=True),
                    "Sigma": np.array(model.slds_params[1], copy=True),
                    "pi": np.array(model.hdp_params[4], copy=True),
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
    futures = np.empty((len(snapshots), training["observations"].shape[0], horizon, model.od))
    for index, snapshot in enumerate(snapshots):
        futures[index] = roll_forward(snapshot, horizon, failures)
    elapsed = time.perf_counter() - started
    return {
        "time_train": training["time"],
        "state_train": training["state"],
        "observations_train": training["observations"],
        "state_hold": held_out["state"],
        "observations_hold": held_out["observations"],
        "posterior_mean": latent_sum / float(sweeps - half),
        "futures": futures,
        "log_joint": log_joint,
        "used_modes": used_modes,
        "elapsed_sec": np.array(elapsed),
        "gaussian_jitter_draws": np.array(failures[0]),
    }


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Root mean square error, pooled over every element of the arrays."""

    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def forecast_metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    """Score the predictive mean and the central 90% band on the holdout.

    Args:
        result: Output of :func:`fit_and_forecast`.

    Returns:
        RMSE against the noisy holdout, the same RMSE for a last-observation
        baseline, and the fraction of holdout entries inside the band.
    """

    futures = result["futures"]
    holdout = result["observations_hold"]
    predictive_mean = np.mean(futures, axis=0)
    lower = np.percentile(futures, 5.0, axis=0)
    upper = np.percentile(futures, 95.0, axis=0)
    inside = (holdout >= lower) & (holdout <= upper)
    last = result["observations_train"][:, -1:, :]
    baseline = np.repeat(last, holdout.shape[1], axis=1)
    names = ("angle", "velocity")
    metrics = {
        "retained_sweeps": float(futures.shape[0]),
        "rmse_model_full": _rmse(predictive_mean, holdout),
        "rmse_model_10": _rmse(predictive_mean[:, :SHORT_HORIZON], holdout[:, :SHORT_HORIZON]),
        "rmse_baseline_full": _rmse(baseline, holdout),
        "rmse_baseline_10": _rmse(baseline[:, :SHORT_HORIZON], holdout[:, :SHORT_HORIZON]),
        "coverage_full": float(np.mean(inside)),
        "coverage_10": float(np.mean(inside[:, :SHORT_HORIZON])),
        "gaussian_jitter_draws": float(result["gaussian_jitter_draws"]),
    }
    for channel, name in enumerate(names):
        metrics[f"rmse_model_{name}"] = _rmse(predictive_mean[:, :, channel], holdout[:, :, channel])
        metrics[f"rmse_model_{name}_10"] = _rmse(
            predictive_mean[:, :SHORT_HORIZON, channel],
            holdout[:, :SHORT_HORIZON, channel],
        )
        metrics[f"coverage_{name}"] = float(np.mean(inside[:, :, channel]))
        metrics[f"rmse_baseline_{name}"] = _rmse(baseline[:, :, channel], holdout[:, :, channel])
    trace = int(np.argmax(RELEASE_ANGLES))
    metrics["rmse_model_largest_trace"] = _rmse(predictive_mean[trace], holdout[trace])
    metrics["rmse_model_largest_10"] = _rmse(
        predictive_mean[trace, :SHORT_HORIZON],
        holdout[trace, :SHORT_HORIZON],
    )
    metrics["coverage_largest_trace"] = float(np.mean(inside[trace]))
    metrics["coverage_largest_10"] = float(np.mean(inside[trace, :SHORT_HORIZON]))
    for start in (0, 20, 40, 60):
        window = slice(start, start + 20)
        metrics[f"rmse_largest_{start + 1}_{start + 20}"] = _rmse(
            predictive_mean[trace, window],
            holdout[trace, window],
        )
        metrics[f"coverage_largest_{start + 1}_{start + 20}"] = float(np.mean(inside[trace, window]))
    for trace_id in range(holdout.shape[0]):
        metrics[f"rmse_model_trace_{trace_id}"] = _rmse(predictive_mean[trace_id], holdout[trace_id])
        metrics[f"coverage_trace_{trace_id}"] = float(np.mean(inside[trace_id]))
    return metrics


def _lead_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RMSE at each lead time, pooled over traces and channels."""

    squared = np.mean((prediction - target) ** 2, axis=(0, 2))
    return np.sqrt(squared)


def save_forecast_plots(result: dict[str, np.ndarray], destination: Path) -> None:
    """Write the forecast ribbon and the lead-time score curves.

    Args:
        result: Output of :func:`fit_and_forecast`.
        destination: Directory for the PNG files.
    """

    destination.mkdir(parents=True, exist_ok=True)
    futures = result["futures"]
    predictive_mean = np.mean(futures, axis=0)
    lower = np.percentile(futures, 5.0, axis=0)
    upper = np.percentile(futures, 95.0, axis=0)
    trace = int(np.argmax(RELEASE_ANGLES))
    train_time = result["time_train"]
    hold_time = train_time[-1] + DT * np.arange(1, futures.shape[2] + 1)
    split = train_time[-1]
    last = result["observations_train"][trace, -1]
    channel_names = ("angle (rad)", "angular velocity (rad/s)")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
    for channel, axis in enumerate(axes):
        axis.axvspan(split, hold_time[-1] + 0.5 * DT, color="#f4e8d0", zorder=0)
        axis.plot(
            train_time,
            result["observations_train"][trace, :, channel],
            color="#9aa0a6",
            lw=0.8,
            label="fitted observations",
        )
        axis.plot(
            train_time,
            result["posterior_mean"][trace, :, channel],
            color="#1f4e79",
            lw=1.3,
            label="posterior mean",
        )
        axis.fill_between(
            hold_time,
            lower[trace, :, channel],
            upper[trace, :, channel],
            color="#1f4e79",
            alpha=0.22,
            label="90% predictive band",
        )
        axis.plot(hold_time, predictive_mean[trace, :, channel], color="#1f4e79", lw=1.5, label="predictive mean")
        axis.plot(
            hold_time,
            result["observations_hold"][trace, :, channel],
            color="#b85c38",
            lw=1.1,
            label="held-out observations",
        )
        axis.plot(
            hold_time,
            np.full(hold_time.shape, last[channel]),
            color="#666666",
            lw=1.0,
            ls=":",
            label="last observation",
        )
        axis.axvline(split, color="#888888", lw=0.8)
        axis.set_ylabel(channel_names[channel])
    axes[0].text(0.74, 0.08, "held out", transform=axes[0].transAxes, color="#8a6a32", fontsize=8)
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Forecast from the fitted window, release {RELEASE_ANGLES[trace]:.2f} rad")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_forecast.png", dpi=140)
    plt.close(fig)

    holdout = result["observations_hold"]
    baseline = np.repeat(result["observations_train"][:, -1:, :], holdout.shape[1], axis=1)
    model_lead = _lead_rmse(predictive_mean, holdout)
    baseline_lead = _lead_rmse(baseline, holdout)
    inside = (holdout >= lower) & (holdout <= upper)
    coverage_lead = np.mean(inside, axis=(0, 2))
    largest_error = predictive_mean[trace, :, :] - holdout[trace, :, :]
    largest_lead = np.sqrt(np.mean(largest_error ** 2, axis=1))
    largest_coverage = np.mean(inside[trace], axis=1)
    lead = np.arange(1, holdout.shape[1] + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    axes[0].plot(lead, model_lead, color="#1f4e79", lw=1.4, label="all traces")
    axes[0].plot(lead, largest_lead, color="#b85c38", lw=1.3, label="largest release")
    axes[0].plot(lead, baseline_lead, color="#666666", lw=1.2, ls=":", label="last observation")
    axes[0].axvline(SHORT_HORIZON, color="#888888", lw=0.8, ls="--")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)
    axes[1].plot(lead, coverage_lead, color="#1f4e79", lw=1.4, label="all traces")
    axes[1].plot(lead, largest_coverage, color="#b85c38", lw=1.3, label="largest release")
    axes[1].axhline(0.90, color="#888888", lw=0.8, ls="--")
    axes[1].axvline(SHORT_HORIZON, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel("90% band coverage")
    axes[1].set_xlabel("lead time (steps)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False)
    fig.suptitle("Holdout score by lead time, pooled over traces and channels")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_forecast_horizon.png", dpi=140)
    plt.close(fig)


def main() -> None:
    """Fit the published pendulum series and score an 80-step forecast."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEPS
    print(
        f"pendulum forecast  seed {SEED}  holdout_seed {HOLDOUT_SEED}  "
        f"fit_steps {STEP_COUNT}  horizon {HORIZON}  sweeps {sweeps}",
        flush=True,
    )
    result = fit_and_forecast(sweeps, HORIZON)
    metrics = forecast_metrics(result)
    save_forecast_plots(result, _IMG_DIR)
    print("forecast")
    for key, value in metrics.items():
        print(f"  {key:28s} {value:.6g}" if isinstance(value, float) else f"  {key:28s} {value}")
    print(f"  elapsed_sec                  {float(result['elapsed_sec']):.1f}")
    print(f"  log_joint_start              {float(result['log_joint'][0]):.6f}")
    print(f"  log_joint_end                {float(result['log_joint'][-1]):.6f}")
    print(f"  figures                      {_IMG_DIR}")


if __name__ == "__main__":
    main()
