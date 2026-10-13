"""Enable float64 before any Gibbs math runs."""

from __future__ import annotations

import jax


def enable_x64() -> None:
    """Turn on JAX's 64-bit mode.

    The regime HAR and the sticky HDP-SLDS both sample in float64. Calling
    this more than once is harmless.
    """

    jax.config.update("jax_enable_x64", True)


enable_x64()
