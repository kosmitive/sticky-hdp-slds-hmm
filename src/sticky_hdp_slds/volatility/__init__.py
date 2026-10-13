"""Rolling-origin volatility scores and their baselines."""

from sticky_hdp_slds.volatility.baselines import (
    direct_har_forecasts,
    ewma_log_forecasts,
    fit_garch_student,
    garch_log_forecasts,
    garch_variance_leads,
    long_run_mean_forecasts,
)
from sticky_hdp_slds.volatility.data import DATASET_SPECS, datasets, load_closes
from sticky_hdp_slds.volatility.diebold import diebold_mariano
from sticky_hdp_slds.volatility.direct import (
    fit_direct_regime_coefficients,
    mixture_log_forecast,
    propagated_regime_weights,
    training_bias,
)
from sticky_hdp_slds.volatility.scoring import (
    CHAINS,
    FORECAST_DRAWS,
    HORIZONS,
    LAG,
    N_REGIMES,
    PRIOR_ALPHA,
    PRIOR_KAPPA,
    REFIT_EVERY,
    REFIT_YEARS,
    SEED,
    SWEEPS,
    evaluate_returns,
)
from sticky_hdp_slds.volatility.series import (
    feature_matrix,
    forward_square_mean,
    har_spectral_radius,
    log_squares,
    positive_floor,
)

__all__ = [
    "CHAINS",
    "DATASET_SPECS",
    "FORECAST_DRAWS",
    "HORIZONS",
    "LAG",
    "N_REGIMES",
    "PRIOR_ALPHA",
    "PRIOR_KAPPA",
    "REFIT_EVERY",
    "REFIT_YEARS",
    "SEED",
    "SWEEPS",
    "datasets",
    "diebold_mariano",
    "direct_har_forecasts",
    "evaluate_returns",
    "ewma_log_forecasts",
    "feature_matrix",
    "fit_direct_regime_coefficients",
    "fit_garch_student",
    "forward_square_mean",
    "garch_log_forecasts",
    "garch_variance_leads",
    "har_spectral_radius",
    "load_closes",
    "log_squares",
    "long_run_mean_forecasts",
    "mixture_log_forecast",
    "positive_floor",
    "propagated_regime_weights",
    "training_bias",
]
