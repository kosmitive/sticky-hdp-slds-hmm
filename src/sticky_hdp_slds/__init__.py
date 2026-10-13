"""Sticky HDP-SLDS and sticky regime-HAR Gibbs samplers.

Importing this package enables float64. Gibbs blocks are pure functions from
``(key, state, data)`` to a new state.
"""

from __future__ import annotations

import sticky_hdp_slds._dtype  # noqa: F401
from sticky_hdp_slds.inference import (
    RegimeData,
    RegimeDraw,
    RegimePrior,
    RegimeState,
    SLDSData,
    SLDSPrior,
    SLDSState,
    backward_sample,
    filter_regime_draws,
    fit_regime_har,
    forward_filter,
    sample_regime_har,
    sample_regime_windows,
    sample_slds,
    slds_sweep,
)
from sticky_hdp_slds.volatility import (
    DATASET_SPECS,
    evaluate_returns,
    load_closes,
)

__all__ = [
    "DATASET_SPECS",
    "RegimeData",
    "RegimeDraw",
    "RegimePrior",
    "RegimeState",
    "SLDSData",
    "SLDSPrior",
    "SLDSState",
    "backward_sample",
    "evaluate_returns",
    "filter_regime_draws",
    "fit_regime_har",
    "forward_filter",
    "load_closes",
    "sample_regime_har",
    "sample_regime_windows",
    "sample_slds",
    "slds_sweep",
]
