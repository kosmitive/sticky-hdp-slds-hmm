# API

Inference is JAX. Importing `slds` enables float64. State is a `NamedTuple` pytree, and each Gibbs block is a pure function of `(key, state, data)`.

`fit_regime_har` is the sticky regime HAR used by the rolling-origin volatility study. `sample_slds` is the blocked sticky HDP-SLDS. `filter_regime_draws` is the forecast filter: one forward recursion, mapped across draws.

## Regime HAR

::: slds.regime_har.fit_regime_har

::: slds.regime_har.sample_regime_har

::: slds.regime_har.sample_regime_windows

## Sticky HDP-SLDS

::: slds.slds.sample_slds

::: slds.slds.slds_sweep

## Filter

::: slds.hmm.filter_regime_draws

::: slds.hmm.forward_filter

::: slds.hmm.backward_sample
