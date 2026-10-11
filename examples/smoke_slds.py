"""Run a few Gibbs sweeps of the sticky HDP-SLDS on a small synthetic series.

The script is the short path from install to a finite posterior sample:

    python examples/smoke_slds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.mniw_prior import MNIWPrior
from model.sticky_hdp_prior import StickyHDPPrior
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel


def run_smoke(steps: int = 3, seed: int = 0) -> dict[str, object]:
    """Simulate a two-mode system and run ``steps`` full Gibbs sweeps.

    Args:
        steps: Number of Gibbs sweeps.
        seed: Seed for NumPy's legacy random generator, which the sampler uses.

    Returns:
        Shapes and a few posterior summaries from the last sweep.
    """

    np.random.seed(seed)
    latent_dim = 2
    mode_count = 2
    trace_count = 2
    step_count = 12
    projection = np.eye(latent_dim)
    hdp_prior = StickyHDPPrior(mode_count, [2.0, 0.5], [8.0, 2.0], [2.0, 0.5], "uniform")
    dynamics_prior = MNIWPrior(
        0.6 * np.eye(latent_dim),
        20.0 * np.eye(latent_dim),
        8,
        0.05 * np.eye(latent_dim),
    )
    model = StickyHDPSLDSHMMModel(
        latent_dim,
        latent_dim,
        hdp_prior,
        dynamics_prior,
        0.05 * np.eye(latent_dim),
        [6.0, 0.05 * np.eye(latent_dim)],
        projection,
    )
    model.init_params()
    model.slds_params[0][:] = 0.6 * np.eye(latent_dim)
    model.slds_params[1][:] = 0.05 * np.eye(latent_dim)
    model.R = 0.05 * np.eye(latent_dim)
    model.hdp_params[3] = np.full(mode_count, 1.0 / mode_count)
    model.hdp_params[4] = np.array([[0.9, 0.1], [0.15, 0.85]])
    model.simulate(trace_count, step_count, init=False)

    for _ in range(steps):
        model.slds_post_block_sample_z()
        backward = model.slds_backward_kalman_filter_gains()
        forward = model.slds_forward_kalman_filter_gains()
        model.slds_post_seq_sample_z(forward[0], forward[1], backward[0], backward[1])
        backward = model.slds_backward_kalman_filter_gains()
        model.slds_post_block_sample_x(backward[0], backward[1])
        model.calc_transitions()
        model.sample_post_auxiliary_vars()
        model.sample_post_hdp_alpha_kappa()
        model.sample_post_hdp_rho()
        model.sample_post_hdp_gamma()
        model.sample_post_hdp_beta()
        model.sample_post_hdp_pi()
        model.slds_post_mniw_dynamics_param_sampling()
        model.slds_post_sample_r()
    log_likelihood = float(model.calc_log_likelihood())
    rho = float(model.hdp_params[1])

    # A separate sticky check: a run of one mode must put mass on its diagonal.
    model.z = np.zeros((trace_count, step_count), dtype=np.int64)
    model.hdp_params[0] = 2.0
    model.hdp_params[1] = 0.5
    model.hdp_params[3] = np.full(mode_count, 0.5)
    model.calc_transitions()
    model.sample_post_hdp_pi()

    return {
        "log_likelihood": log_likelihood,
        "x_shape": tuple(model.x.shape),
        "y_shape": tuple(model.y.shape),
        "w_shape": tuple(model.w.shape),
        "pi_row_sums": model.hdp_params[4].sum(axis=1).copy(),
        "rho": rho,
        "sticky_diagonal": float(model.hdp_params[4][0, 0]),
    }


def main() -> None:
    """Print a one-screen summary of a smoke run."""

    report = run_smoke()
    print("sticky HDP-SLDS smoke run")
    print(f"  log likelihood     {report['log_likelihood']:.3f}")
    print(f"  latent shape       {report['x_shape']}")
    print(f"  observation shape  {report['y_shape']}")
    print(f"  emission noise     {report['w_shape']}")
    print(f"  pi row sums        {np.array2string(report['pi_row_sums'], precision=3)}")
    print(f"  rho                {report['rho']:.3f}")
    print(f"  sticky diagonal    {report['sticky_diagonal']:.3f}")


if __name__ == "__main__":
    main()
