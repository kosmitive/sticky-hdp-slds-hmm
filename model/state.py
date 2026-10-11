"""Typed parameter state for the sticky HDP-SLDS.

The sampler used to keep these values in positional lists. ``StickyHDPState``
is the transition side and ``DynamicsState`` is the linear-Gaussian side.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class StickyHDPState:
    """Truncated sticky HDP-HMM parameters.

    Attributes:
        alpha_plus_kappa: Concentration ``alpha + kappa``.
        rho: Sticky self-transition weight ``kappa / (alpha + kappa)``.
        gamma: Top-level concentration of the global weights.
        beta: Global weights, shape ``(modes,)``.
        pi: Mode transition matrix, shape ``(modes, modes)``.
    """

    alpha_plus_kappa: float
    rho: float
    gamma: float
    beta: np.ndarray
    pi: np.ndarray

    @classmethod
    def from_sequence(cls, values: Sequence[object]) -> StickyHDPState:
        """Build a state from the historical five-element list.

        Args:
            values: ``[alpha_plus_kappa, rho, gamma, beta, pi]``.

        Returns:
            A sticky HDP state.
        """

        alpha_plus_kappa, rho, gamma, beta, pi = values
        return cls(
            float(alpha_plus_kappa),
            float(rho),
            float(gamma),
            np.asarray(beta, dtype=np.float64),
            np.asarray(pi, dtype=np.float64),
        )


@dataclass
class DynamicsState:
    """Per-mode linear dynamics.

    Attributes:
        A: Dynamic matrices, shape ``(modes, latent, latent)``.
        Sigma: Process covariances, shape ``(modes, latent, latent)``.
        alpha: Optional ARD precisions, shape ``(modes, latent)``.
    """

    A: np.ndarray
    Sigma: np.ndarray
    alpha: np.ndarray | None = None

    @classmethod
    def from_sample(cls, sample: Sequence[np.ndarray]) -> DynamicsState:
        """Build a state from a prior draw.

        Args:
            sample: ``(A, Sigma)`` or ``(A, Sigma, alpha)``.

        Returns:
            A dynamics state. The ARD precision is present only for the
            three-array draw.
        """

        if len(sample) == 3:
            return cls(sample[0], sample[1], sample[2])
        return cls(sample[0], sample[1])


def load_latent_and_emission_scale(
    folder: str,
    latent_dim: int,
    observation_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the latent covariance and the emission scale.

    Historical packs store the emission scale in ``P0.npy`` and the latent
    covariance in ``R0.npy``. An honest layout stores the latent covariance
    in ``P0.npy`` and the emission scale in ``R0.npy``. Both files use those
    two names, so the array shapes against the model dimensions decide.

    Args:
        folder: Directory that contains the arrays.
        latent_dim: Latent state dimension.
        observation_dim: Observation dimension.

    Returns:
        The latent covariance and the emission-scale matrix, in that order.
    """

    named = {}
    for name in ("P0.npy", "R0.npy"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            named[name] = np.load(path)
    if "P0.npy" not in named or "R0.npy" not in named:
        missing = [name for name in ("P0.npy", "R0.npy") if name not in named]
        raise FileNotFoundError(f"Missing {missing} in {folder}.")
    p0 = named["P0.npy"]
    r0 = named["R0.npy"]
    if p0.shape[-1] == observation_dim and r0.shape[-1] == latent_dim:
        return r0, p0
    if p0.shape[-1] == latent_dim and r0.shape[-1] == observation_dim:
        return p0, r0
    raise ValueError(
        f"Cannot match P0.npy {p0.shape} and R0.npy {r0.shape} to "
        f"latent {latent_dim} and observation {observation_dim}."
    )
