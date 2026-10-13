"""Three blocked Gibbs sweeps on a tiny two-mode system.

uv run python -m examples.smoke_slds
"""

from __future__ import annotations

import numpy as np
from jax import random

from sticky_hdp_slds import SLDSData, SLDSPrior, SLDSState, sample_slds


def run_smoke(sweeps: int = 3) -> dict[str, np.ndarray]:
    """Run a short blocked chain and return the final state as NumPy arrays."""

    import jax.numpy as jnp

    observations = jnp.asarray(np.random.default_rng(0).normal(size=(2, 12, 2)))
    eye = jnp.eye(2)
    state = SLDSState(
        modes=jnp.zeros((2, 12), dtype=jnp.int32),
        latents=observations,
        dynamics=jnp.stack([0.5 * eye, 0.2 * eye]),
        process_cov=jnp.stack([0.2 * eye, 0.3 * eye]),
        emission_cov=0.1 * eye,
        beta=jnp.array([0.6, 0.4]),
        transition=jnp.array([[0.9, 0.1], [0.2, 0.8]]),
        alpha_plus_kappa=jnp.asarray(8.0),
        rho=jnp.asarray(0.75),
        gamma=jnp.asarray(1.5),
        column_precision=jnp.ones((2, 2)),
    )
    prior = SLDSPrior(
        alpha_kappa=jnp.array([2.0, 1.0]),
        rho=jnp.array([8.0, 2.0]),
        gamma=jnp.array([2.0, 1.0]),
        dynamics_mean=0.4 * eye,
        dynamics_column_precision=eye,
        dynamics_df=jnp.asarray(8.0),
        dynamics_scale=0.2 * eye,
        latent_prior=eye,
        emission_df=jnp.asarray(6.0),
        emission_scale=0.2 * eye,
        emission=eye,
        ard=jnp.array([1.0, 1.0]),
    )
    trace, loglik = sample_slds(
        random.PRNGKey(1), state, SLDSData(observations=observations), prior, sweeps
    )
    loglik.block_until_ready()
    final = jax_tree_last(trace)
    final["log_joint"] = np.asarray(loglik)
    return final


def jax_tree_last(trace) -> dict[str, np.ndarray]:
    """Last sweep of a stacked state, as NumPy arrays."""

    return {
        "modes": np.asarray(trace.modes[-1]),
        "latents": np.asarray(trace.latents[-1]),
        "dynamics": np.asarray(trace.dynamics[-1]),
        "transition": np.asarray(trace.transition[-1]),
        "rho": np.asarray(trace.rho[-1]),
        "emission_cov": np.asarray(trace.emission_cov[-1]),
    }


def main() -> None:
    """Print a finite log joint and a sticky transition row."""

    result = run_smoke()
    rows = result["transition"].sum(axis=1)
    print(f"log_joint {result['log_joint'][-1]:.2f}")
    print(f"latents {result['latents'].shape} observations-aligned")
    print(f"transition_rows {rows}")
    print(f"rho {float(result['rho']):.3f}")
    assert np.all(np.isfinite(result["log_joint"]))
    assert np.allclose(rows, 1.0, atol=1e-6)
    assert 0.0 < float(result["rho"]) < 1.0


if __name__ == "__main__":
    main()
