"""JAX inference for the sticky regime HAR and the sticky HDP-SLDS.

Importing this package enables float64. Every Gibbs block is a pure function
from ``(key, state, data)`` to a new state.
"""

from __future__ import annotations

import slds._dtype  # noqa: F401
from slds.regime_har import fit_regime_har, sample_regime_har, sample_regime_windows
from slds.slds import sample_slds, slds_sweep

__all__ = [
    "fit_regime_har",
    "sample_regime_har",
    "sample_regime_windows",
    "sample_slds",
    "slds_sweep",
]
