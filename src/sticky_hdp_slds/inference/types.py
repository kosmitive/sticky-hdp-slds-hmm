"""Immutable pytree state for the JAX samplers.

``NamedTuple`` fields are JAX pytrees, so ``jit``, ``vmap``, and ``scan``
can pass a whole Gibbs state through a pure function.
"""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array, Bool, Float, Int


class RegimeData(NamedTuple):
    """Daily log-variance regression data.

    Attributes:
        series: Response, shape ``(T,)``.
        features: HAR lags, shape ``(T, F)``.
    """

    series: Float[Array, "T"]
    features: Float[Array, "T F"]


class RegimePrior(NamedTuple):
    """Fixed hyperparameters for one regime-HAR chain.

    Attributes:
        alpha: Off-diagonal Dirichlet concentration.
        kappa: Extra diagonal Dirichlet concentration.
        variance_a: Inverse-gamma shape for each regime scale.
        variance_b: Inverse-gamma scale for each regime scale.
        prior_mean: Gaussian prior mean of intercepts and slopes.
        prior_var: Gaussian prior variance of intercepts and slopes.
        radius_limit: Companion-root bound used by the Schur-Cohn check.
    """

    alpha: Float[Array, ""]
    kappa: Float[Array, ""]
    variance_a: Float[Array, ""]
    variance_b: Float[Array, ""]
    prior_mean: Float[Array, "P"]
    prior_var: Float[Array, "P"]
    radius_limit: Float[Array, ""]


class RegimeState(NamedTuple):
    """One regime-HAR Gibbs state.

    Attributes:
        coef: Intercepts followed by the shared HAR slopes.
        sigma: Innovation scale of each regime.
        transition: Row-stochastic transition matrix.
        modes: Latent regime at each time.
    """

    coef: Float[Array, "P"]
    sigma: Float[Array, "K"]
    transition: Float[Array, "K K"]
    modes: Int[Array, "T"]


class RegimeDraw(NamedTuple):
    """Quantities recorded after one sweep.

    Attributes:
        coef: Intercepts and shared slopes.
        sigma: Regime scales.
        transition: Transition matrix after intercept ordering.
        occupancy: Fraction of times assigned to each ordered regime.
        accepted: Whether the coefficient proposal passed the Schur-Cohn test.
        loglik: Complete-data log likelihood.
    """

    coef: Float[Array, "P"]
    sigma: Float[Array, "K"]
    transition: Float[Array, "K K"]
    occupancy: Float[Array, "K"]
    accepted: Bool[Array, ""]
    loglik: Float[Array, ""]


class SLDSData(NamedTuple):
    """Observations for the sticky HDP-SLDS.

    Attributes:
        observations: Shape ``(traces, steps, obs_dim)``.
    """

    observations: Float[Array, "N T O"]


class SLDSPrior(NamedTuple):
    """Weak-limit sticky HDP-SLDS prior used by one sweep.

    Attributes:
        alpha_kappa: Gamma prior on ``alpha + kappa``, shape and rate.
        rho: Beta prior on the sticky weight.
        gamma: Gamma prior on the top-level concentration, shape and rate.
        dynamics_mean: Matrix-normal mean, shape ``(latent, latent)``.
        dynamics_column_precision: Column precision ``K`` of the matrix normal.
        dynamics_df: Inverse-Wishart degrees of freedom.
        dynamics_scale: Inverse-Wishart scale.
        latent_prior: Covariance of the latent state at the first time.
        emission_df: Inverse-Wishart degrees of freedom for ``R``.
        emission_scale: Inverse-Wishart scale for ``R``.
        emission: Observation matrix ``C``.
        ard: Gamma prior on the ARD column precisions, shape and rate.
            The matrix-normal sweep does not read this field.
    """

    alpha_kappa: Float[Array, "2"]
    rho: Float[Array, "2"]
    gamma: Float[Array, "2"]
    dynamics_mean: Float[Array, "D D"]
    dynamics_column_precision: Float[Array, "D D"]
    dynamics_df: Float[Array, ""]
    dynamics_scale: Float[Array, "D D"]
    latent_prior: Float[Array, "D D"]
    emission_df: Float[Array, ""]
    emission_scale: Float[Array, "O O"]
    emission: Float[Array, "O D"]
    ard: Float[Array, "2"]


class SLDSState(NamedTuple):
    """Latent path and parameters of one sticky HDP-SLDS sweep.

    Attributes:
        modes: Mode index, shape ``(traces, steps)``.
        latents: Continuous state, shape ``(traces, steps, latent)``.
        dynamics: Mode matrices ``A``, shape ``(modes, latent, latent)``.
        process_cov: Mode process covariances.
        emission_cov: Shared emission covariance.
        beta: Global weights.
        transition: Mode transition matrix.
        alpha_plus_kappa: Sum of the HDP concentration and the sticky mass.
        rho: Sticky weight.
        gamma: Top-level concentration.
        column_precision: ARD precision of each column of ``A``. The
            matrix-normal sweep leaves this field unchanged.
    """

    modes: Int[Array, "N T"]
    latents: Float[Array, "N T D"]
    dynamics: Float[Array, "K D D"]
    process_cov: Float[Array, "K D D"]
    emission_cov: Float[Array, "O O"]
    beta: Float[Array, "K"]
    transition: Float[Array, "K K"]
    alpha_plus_kappa: Float[Array, ""]
    rho: Float[Array, ""]
    gamma: Float[Array, ""]
    column_precision: Float[Array, "K D"]
