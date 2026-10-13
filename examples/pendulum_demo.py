"""Fit the sticky HDP-SLDS to a noisy nonlinear pendulum.

The pendulum is integrated with the symplectic Euler method. Its one-step
map is homogeneous and linear in ``(angle, angular velocity)`` only inside
a fixed angle regime, because ``sin(theta) = sinc(theta) * theta``. Piecewise
linear modes are the approximation the sampler is asked to find.

Observations are the angle and the angular velocity, plus Gaussian noise.
The emission matrix is the identity. Hyperparameters below are for this
series; a different length or noise level needs its own prior scales.

    python examples/pendulum_demo.py
    python examples/pendulum_demo.py 40
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

# The series uses a Generator. The Gibbs sampler uses a JAX key with the same seed.
SEED = 11
TRACE_COUNT = 4
STEP_COUNT = 160
DT = 0.05
GRAVITY_OVER_LENGTH = 9.81
RELEASE_ANGLES = (0.45, 1.05, 1.70, 2.35)
OBSERVATION_STD = np.array([0.04, 0.12])
PROCESS_STD = 0.01
TRUNCATION = 5
DEFAULT_SWEEPS = 300

_IMG_DIR = _ROOT / "docs" / "img"
_MODE_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2")


def simulate_pendulum(seed: int = SEED) -> dict[str, np.ndarray]:
    """Integrate several releases and add observation noise.

    Args:
        seed: Seed for the observation noise and the small process noise.

    Returns:
        Time, the noise-free state, and the noisy observations. Both series
        have shape ``(traces, steps, 2)`` in ``(angle, angular velocity)``.
    """

    rng = np.random.default_rng(seed)
    state = np.empty((TRACE_COUNT, STEP_COUNT, 2))
    stiffness = GRAVITY_OVER_LENGTH
    for trace, release in enumerate(RELEASE_ANGLES):
        angle = float(release)
        velocity = 0.0
        for step in range(STEP_COUNT):
            state[trace, step, 0] = angle
            state[trace, step, 1] = velocity
            velocity = velocity - DT * stiffness * np.sin(angle)
            velocity += float(rng.normal(0.0, PROCESS_STD))
            angle = angle + DT * velocity
    noise = rng.normal(0.0, 1.0, size=state.shape) * OBSERVATION_STD
    time = np.arange(STEP_COUNT) * DT
    return {"time": time, "state": state, "observations": state + noise}


def continue_pendulum(state: np.ndarray, extra_steps: int, seed: int) -> dict[str, np.ndarray]:
    """Integrate ``extra_steps`` past the last stored state.

    The step is the same symplectic Euler update as :func:`simulate_pendulum`.
    The stored state is the pre-update angle and velocity, so the first
    returned frame is one step after ``state[:, -1]``.

    Args:
        state: Noise-free series, shape ``(traces, steps, 2)``.
        extra_steps: Number of new frames.
        seed: Seed for the process disturbance and the observation noise
            on the continuation.

    Returns:
        Noise-free continuation and noisy observations, each of shape
        ``(traces, extra_steps, 2)``.
    """

    rng = np.random.default_rng(seed)
    trace_count = state.shape[0]
    continued = np.empty((trace_count, extra_steps, 2))
    stiffness = GRAVITY_OVER_LENGTH
    for trace in range(trace_count):
        angle = float(state[trace, -1, 0])
        velocity = float(state[trace, -1, 1])
        for step in range(extra_steps):
            velocity = velocity - DT * stiffness * np.sin(angle)
            velocity += float(rng.normal(0.0, PROCESS_STD))
            angle = angle + DT * velocity
            continued[trace, step, 0] = angle
            continued[trace, step, 1] = velocity
    noise = rng.normal(0.0, 1.0, size=continued.shape) * OBSERVATION_STD
    return {"state": continued, "observations": continued + noise}


def small_angle_dynamics() -> np.ndarray:
    """Linear map of the symplectic Euler step at ``theta = 0``."""

    stiffness = DT * GRAVITY_OVER_LENGTH
    return np.array(
        [
            [1.0 - DT * stiffness, DT],
            [-stiffness, 1.0],
        ]
    )


def _slds_prior_and_state(observations: np.ndarray):
    """Prior means and an observation-initialized state for the blocked sampler."""

    import jax.numpy as jnp
    from slds.types import SLDSData, SLDSPrior, SLDSState

    latent_dim = 2
    mean_dynamics = jnp.asarray(small_angle_dynamics(), dtype=jnp.float64)
    dynamics = jnp.broadcast_to(mean_dynamics, (TRUNCATION, latent_dim, latent_dim))
    beta = jnp.full(TRUNCATION, 1.0 / TRUNCATION)
    stay = 0.92
    transition = (1.0 - stay) * beta + stay * jnp.eye(TRUNCATION)
    observations_j = jnp.asarray(observations, dtype=jnp.float64)
    state = SLDSState(
        modes=jnp.zeros(observations_j.shape[:2], dtype=jnp.int32),
        latents=observations_j,
        dynamics=dynamics,
        process_cov=jnp.broadcast_to(
            0.15 * jnp.eye(latent_dim), (TRUNCATION, latent_dim, latent_dim)
        ),
        emission_cov=jnp.diag(jnp.asarray(OBSERVATION_STD, dtype=jnp.float64) ** 2),
        beta=beta,
        transition=transition,
        alpha_plus_kappa=jnp.asarray(12.0),
        rho=jnp.asarray(stay),
        gamma=jnp.asarray(1.5),
        column_precision=jnp.full((TRUNCATION, latent_dim), 40.0),
    )
    prior = SLDSPrior(
        alpha_kappa=jnp.asarray([12.0, 1.0]),
        rho=jnp.asarray([24.0, 2.0]),
        gamma=jnp.asarray([3.0, 2.0]),
        dynamics_mean=mean_dynamics,
        dynamics_column_precision=40.0 * jnp.eye(latent_dim),
        dynamics_df=jnp.asarray(8.0),
        dynamics_scale=0.15 * jnp.eye(latent_dim),
        latent_prior=jnp.diag(jnp.asarray([2.0, 16.0])),
        emission_df=jnp.asarray(6.0),
        emission_scale=jnp.diag(jnp.asarray(OBSERVATION_STD, dtype=jnp.float64) ** 2) * 3.0,
        emission=jnp.eye(latent_dim),
        ard=jnp.ones(latent_dim),
    )
    return state, SLDSData(observations=observations_j), prior


def fit(observations: np.ndarray, sweeps: int) -> dict[str, np.ndarray]:
    """Run ``sweeps`` blocked Gibbs iterations and keep the monitored traces.

    Args:
        observations: Noisy pendulum series, shape ``(traces, steps, 2)``.
        sweeps: Number of complete Gibbs sweeps.

    Returns:
        Log joint, occupied-mode counts, ``rho``, latents, modes, and dynamics.
    """

    from jax import random
    from slds.slds import sample_slds

    state, data, prior = _slds_prior_and_state(observations)
    started = time.perf_counter()
    trace, loglik = sample_slds(
        random.PRNGKey(SEED),
        state,
        data,
        prior,
        sweeps,
        sequential_modes=False,
        ard=False,
    )
    loglik.block_until_ready()
    elapsed = time.perf_counter() - started
    log_joint = np.asarray(loglik)
    modes = np.asarray(trace.modes)
    used_modes = np.array([np.unique(modes[sweep]).size for sweep in range(sweeps)])
    print(
        f"blocked sweeps {sweeps} in {elapsed:.1f}s  "
        f"log_joint[0] {log_joint[0]:.2f}  log_joint[-1] {log_joint[-1]:.2f}",
        flush=True,
    )
    return {
        "log_joint": log_joint,
        "used_modes": used_modes,
        "rho": np.asarray(trace.rho),
        "latents": np.asarray(trace.latents),
        "modes": modes,
        "dynamics": np.asarray(trace.dynamics),
        "elapsed_sec": np.array(elapsed),
        "prior_emission": np.asarray(prior.emission),
        "process_cov": np.asarray(trace.process_cov),
        "transition": np.asarray(trace.transition),
        "emission_cov": np.asarray(trace.emission_cov),
    }


def convergence_summary(log_joint: np.ndarray, used_modes: np.ndarray) -> dict[str, float]:
    """Summarize whether the second half of the chain is still trending.

    Args:
        log_joint: Complete-data log joint after each sweep.
        used_modes: Number of distinct modes after each sweep.

    Returns:
        Levels, a per-sweep slope on the second half, and that slope measured
        in units of the second-half standard deviation.
    """

    half = max(len(log_joint) // 2, 1)
    second = np.asarray(log_joint[half:], dtype=np.float64)
    centered = np.arange(second.size, dtype=np.float64)
    centered -= centered.mean()
    denom = float(np.dot(centered, centered))
    slope = float(np.dot(centered, second - second.mean()) / denom) if denom > 0.0 else 0.0
    scale = float(np.std(second))
    return {
        "start": float(log_joint[0]),
        "mid": float(log_joint[half]),
        "end": float(log_joint[-1]),
        "second_half_mean": float(np.mean(second)),
        "second_half_std": scale,
        "second_half_slope_per_sweep": slope,
        "slope_over_std": slope / scale if scale > 0.0 else 0.0,
        "modes_min": int(used_modes.min()),
        "modes_max": int(used_modes.max()),
        "modes_last": int(used_modes[-1]),
        "modes_second_half_mean": float(np.mean(used_modes[half:])),
    }


def _relabel_by_angle(modes: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Order mode ids by mean absolute angle so a figure is readable.

    Args:
        modes: One mode sequence, shape ``(traces, steps)``.
        angle: Angle series with the same shape prefix.

    Returns:
        A copy of ``modes`` with labels sorted by increasing ``|angle|``.
    """

    present = np.unique(modes)
    centers = []
    for mode in present:
        centers.append(float(np.mean(np.abs(angle[modes == mode]))))
    order = present[np.argsort(centers)]
    relabeled = np.empty_like(modes)
    for new_id, old_id in enumerate(order):
        relabeled[modes == old_id] = new_id
    return relabeled


def save_plots(
    series: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    destination: Path,
) -> None:
    """Write the convergence, reconstruction, and mode figures.

    Args:
        series: Output of :func:`simulate_pendulum`.
        history: Output of :func:`fit`.
        destination: Directory for the PNG files.
    """

    destination.mkdir(parents=True, exist_ok=True)
    sweeps = np.arange(1, history["log_joint"].size + 1)
    half = history["log_joint"].size // 2
    posterior_mean = np.mean(history["latents"][half:], axis=0)
    final_modes = _relabel_by_angle(history["modes"][-1], series["state"][:, :, 0])
    trace = int(np.argmax(RELEASE_ANGLES))
    time_axis = series["time"]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axes[0].plot(sweeps, history["log_joint"], color="#1f4e79", lw=1.4)
    axes[0].axvline(half + 1, color="#888888", ls="--", lw=0.8)
    axes[0].set_ylabel("complete-data log joint")
    axes[1].plot(sweeps, history["used_modes"], color="#b85c38", lw=1.4)
    axes[1].set_ylabel("occupied modes")
    axes[1].set_xlabel("Gibbs sweep")
    axes[1].set_ylim(0.5, TRUNCATION + 0.5)
    fig.suptitle("Sticky HDP-SLDS on the noisy pendulum")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_convergence.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    names = ("angle (rad)", "angular velocity (rad/s)")
    for axis, name in enumerate(names):
        axes[axis].plot(
            time_axis,
            series["observations"][trace, :, axis],
            color="#9aa0a6",
            lw=0.9,
            label="observed",
        )
        axes[axis].plot(
            time_axis,
            series["state"][trace, :, axis],
            color="#333333",
            lw=1.0,
            ls="--",
            label="noise-free",
        )
        axes[axis].plot(
            time_axis,
            posterior_mean[trace, :, axis],
            color="#1f4e79",
            lw=1.5,
            label="posterior mean",
        )
        axes[axis].set_ylabel(name)
    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Largest release ({RELEASE_ANGLES[trace]:.2f} rad), second-half mean state")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(TRACE_COUNT, 1, figsize=(8.2, 7.2), sharex=True)
    for trace_id, axis in enumerate(axes):
        modes = final_modes[trace_id]
        change = np.flatnonzero(np.diff(modes)) + 1
        bounds = np.concatenate(([0], change, [modes.size]))
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            axis.axvspan(
                time_axis[start] - 0.5 * DT,
                time_axis[end - 1] + 0.5 * DT,
                color=_MODE_COLORS[int(modes[start]) % len(_MODE_COLORS)],
                alpha=0.45,
                lw=0,
            )
        axis.plot(time_axis, series["state"][trace_id, :, 0], color="#222222", lw=1.1)
        axis.set_ylabel(f"trace {trace_id + 1}")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Final mode sample, labels ordered by mean |angle|")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_modes.png", dpi=140)
    plt.close(fig)


def _print_regime_diagnostic(
    series: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
) -> None:
    """Print whether the final modes split the swing by absolute angle."""

    final = history["modes"][-1]
    angle = np.abs(series["state"][:, :, 0])
    half = history["log_joint"].size // 2
    quarters = np.array_split(history["log_joint"], 4)
    print("log_joint_quarter_means", " ".join(f"{float(np.mean(part)):.1f}" for part in quarters))
    posterior_mean = np.mean(history["latents"][half:], axis=0)
    truth = series["state"]
    observed = series["observations"]
    print(
        "rmse_posterior_mean",
        float(np.sqrt(np.mean((posterior_mean - truth) ** 2))),
        "rmse_observations",
        float(np.sqrt(np.mean((observed - truth) ** 2))),
    )
    print("final_A")
    for mode in np.unique(final):
        mask = final == mode
        dwells = []
        for trace in final:
            run = 0
            for value in trace:
                if value == mode:
                    run += 1
                elif run:
                    dwells.append(run)
                    run = 0
            if run:
                dwells.append(run)
        print(
            f"  mode {int(mode)}  count {int(mask.sum())}  "
            f"mean_abs_angle {float(np.mean(angle[mask])):.3f}  "
            f"mean_dwell {float(np.mean(dwells)):.2f}"
        )
        print("   ", np.array2string(history["dynamics"][-1, mode], precision=3))


def main() -> None:
    """Simulate the pendulum, run the sampler, and write the three figures."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEPS
    series = simulate_pendulum(SEED)
    print(
        f"pendulum demo  seed {SEED}  traces {TRACE_COUNT}  "
        f"steps {STEP_COUNT}  dt {DT}  truncation {TRUNCATION}  sweeps {sweeps}",
        flush=True,
    )
    history = fit(series["observations"], sweeps)
    summary = convergence_summary(history["log_joint"], history["used_modes"])
    save_plots(series, history, _IMG_DIR)
    _print_regime_diagnostic(series, history)
    print("convergence")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    print(f"  elapsed_sec                  {float(history['elapsed_sec']):.1f}")
    print(f"  rho_last                     {float(history['rho'][-1]):.3f}")
    print(f"  figures                      {_IMG_DIR}")


if __name__ == "__main__":
    main()
