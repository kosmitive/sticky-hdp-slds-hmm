# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829). The result below is an 80-step forecast of a noisy pendulum that the sampler never sees.

## Forecast

![Fitted window, 90% forecast band, and held-out observations for the largest release](img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](img/pendulum_forecast_horizon.png)

`examples/pendulum_forecast.py` fits the published series (seed 11, 300 sweeps) and scores 150 futures from sweeps 151–300. Pooled over four traces and both channels, the predictive mean has RMSE 0.148 on the first 10 steps and 1.456 over all 80. Repeating the last observation scores 1.446 and 3.032 on those same windows. The central 90% band covers 98.8% of the first 10 steps and 94.5% of the horizon.

The 2.35 rad release does not hold that rate. Its 90% band covers 78.8% of the horizon and 42.5% of the last 20-step block. The [results](results.md) page records the fit, the per-trace split, and how the forecast is drawn. The [volatility](volatility.md) page scores the same sampler on daily S&P 500 and bitcoin closes. The S&P target is log 20-day realized volatility. Bitcoin, which trades every calendar day, uses a 30-day window and a 365-day holdout. The return is the second observation and is close to unpredictable.

The sampler alternates between:

- block sampling of the latent state with a backward information-form Kalman filter
- block and sequential sampling of the mode sequence
- sticky HDP updates for the transition rows, the global weights, and the concentration parameters
- matrix-normal inverse-Wishart or automatic relevance determination updates for the mode dynamics
- an inverse-Wishart update for the shared emission covariance

Parameters live on two objects. `StickyHDPState` holds `alpha_plus_kappa`, `rho`, `gamma`, `beta`, and `pi`. `DynamicsState` holds the per-mode maps `A`, the process covariances `Sigma`, and the optional ARD precisions `alpha`. One call, `StickyHDPSLDSHMMModel.gibbs_step`, owns the sweep order.

The [install page](install.md) covers `uv`.
