# Volatility forecast

The score is the log of the mean squared return over the next 1, 5, 20, and 60 days.

## Rolling origin

`examples/regime_volatility.py` scores every origin in the holdout. The Bitcoin holdout is the last 365 calendar days. The S&P 500 holdout is the last 250 trading days. A horizon is kept only when the whole window sits inside the sample, so the 60-day score has fewer origins. The loss is squared error on that log mean. Diebold-Mariano statistics use a Newey-West variance with the Bartlett kernel and `h - 1` lags. A negative statistic means the regime model has the smaller loss. The p-value is two-sided normal.

Four baselines are fit only on data before the target window. HAR(1/5/22) is an ordinary least-squares regression of the log-mean target on an intercept, log daily variance, and the log of the 5-day and 22-day mean variance. The long-run mean is the average of those same completed targets. EWMA with decay 0.94 is reported as the log of its flat variance forecast. GARCH(1,1) with Student-t errors, fit with `arch` on the returns up to the origin, is reported as the log of the mean analytic variance over the horizon.

The regime model is a 3-state Markov-switching HAR on daily `log(r^2)`, not the sticky HDP-SLDS. The Gibbs step uses one shared HAR slope, with a separate intercept and innovation scale in each regime. Draws whose companion spectral radius is 0.999 or higher are rejected. The Dirichlet prior on each transition row has mean self-stay 0.98 (`alpha = 1`, `kappa = 97`). The sampler produces regime probabilities and a transition matrix.

The scored forecast does not simulate a path. For each horizon, each retained draw gets its own direct regression of `log(mean r^2)` over the next `h` days on an intercept and three lags of daily `log(r^2)`: the previous day, the mean of the previous 5, and the mean of the previous 22. Each regime has its own intercept and its own three slopes. An origin is weighted by the average of the filtered distribution after 1 through `h` steps of that draw's transition matrix. A regime whose weights sum to less than 30 uses the pooled regression. The reported forecast is the average, over draws, of that weighted prediction minus the draw's mean residual. The residual is computed only on origins whose `h`-day window has already ended, so the correction never sees the origin being scored. Nothing is exponentiated.

Parameters are refit every 21 origins on a trailing 4-year window: 1460 calendar days for bitcoin and 1008 trading days for the S&P 500. Each refit is 4 chains of 2000 sweeps from seed 23. Forecasts use 40 evenly spaced draws from the second half of each chain, 160 draws in total. The sweep count was raised until the worst refit on both holdouts had R-hat below 1.1. Nothing in the prior, the window, or the refit schedule was chosen on the holdout.

The "Simulated" column is the previous path forecast, kept for comparison and not re-estimated. That run drew 25 Gaussian paths per retained draw, exponentiated them, and averaged `log(mean r^2)`. It used 3 chains of 500 sweeps. Its HAR, EWMA, and mean columns match the tables on the following pages at three decimals. Its GARCH column had 10 Bitcoin failures; this run has 9, so the Bitcoin GARCH numbers are this run's.

Exact zero returns, one in the Bitcoin file and five in the S&P file, are replaced in `log(r^2)` by the smallest positive square in the estimation sample.

```bash
uv run python -m examples.regime_volatility btc
uv run python -m examples.regime_volatility sp500
```

Regime draws come from `sample_regime_windows`, and the forecast filter is `filter_regime_draws`. The baselines stay NumPy. An earlier NumPy Gibbs sampler on this same protocol took 651.1 s on bitcoin and 312.5 s on the S&P 500. See the [inference](inference.md) page.

The [Bitcoin](bitcoin.md) holdout is first. The [S&P 500](sp500.md) holdout follows.
