# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829). The result below is an 80-step forecast of a noisy pendulum that the sampler never sees. The sampler is JAX.

## Forecast

![Fitted window, 90% forecast band, and held-out observations for the largest release](img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](img/pendulum_forecast_horizon.png)

`examples/pendulum_forecast.py` fits the training series with the blocked sampler (seed 11, 300 sweeps, sequential mode updates off) and scores 150 futures from sweeps 151–300. The chain starts at the prior means. Pooled over four traces and both channels, the predictive mean has RMSE 0.162 on the first 10 steps and 1.553 over all 80. The central 90% band covers 97.5% of the first 10 steps and 92.0% of the horizon. Repeating the last observation scores 1.446 and 3.032 on those same windows. The 2.35 rad release in the figure has horizon coverage 69.4%, and 40% on its last 20-step block.

The [results](results.md) page records the fit and the per-trace split. The [volatility](volatility.md) page scores daily Bitcoin and S&P 500 closes with a 3-regime sticky HAR.

The blocked sweep draws the latent states, the mode sequence, the sticky HDP, the dynamics, and the emission covariance. `sequential_modes=True` adds the single-site mode update. `ard=True` replaces the matrix-normal dynamics update. The pendulum fit uses neither.

The [install page](install.md) covers `uv`.
