# API

Import `sticky_hdp_slds`. That enables float64. State is a `NamedTuple` pytree, and each Gibbs block is a pure function of `(key, state, data)`.

`fit_regime_har` is the sticky regime HAR. `sample_slds` is the blocked sticky HDP-SLDS. `filter_regime_draws` is the forecast filter. `evaluate_returns` is the rolling-origin score, including the direct forecast and the HAR, mean, EWMA, and GARCH baselines. `diebold_mariano` is the loss comparison. `load_closes` reads a close snapshot.

The GARCH baseline needs the `volatility` extra. Plots in the examples need the `plots` extra.

## Regime HAR

::: sticky_hdp_slds.inference.regime_har.fit_regime_har

::: sticky_hdp_slds.inference.regime_har.sample_regime_har

::: sticky_hdp_slds.inference.regime_har.sample_regime_windows

## Sticky HDP-SLDS

::: sticky_hdp_slds.inference.slds.sample_slds

::: sticky_hdp_slds.inference.slds.slds_sweep

## Filter

::: sticky_hdp_slds.inference.hmm.filter_regime_draws

::: sticky_hdp_slds.inference.hmm.forward_filter

::: sticky_hdp_slds.inference.hmm.backward_sample

## Volatility

::: sticky_hdp_slds.volatility.scoring.evaluate_returns

::: sticky_hdp_slds.volatility.diebold.diebold_mariano

::: sticky_hdp_slds.volatility.direct.fit_direct_regime_coefficients

::: sticky_hdp_slds.volatility.data.load_closes

::: sticky_hdp_slds.volatility.baselines.direct_har_forecasts

::: sticky_hdp_slds.volatility.baselines.long_run_mean_forecasts

::: sticky_hdp_slds.volatility.baselines.ewma_log_forecasts

::: sticky_hdp_slds.volatility.baselines.garch_log_forecasts
