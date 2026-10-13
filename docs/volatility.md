# Volatility forecast

The score is the log of the mean squared return over the next 1, 5, 20, and 60 days.

## Rolling origin

`examples/regime_volatility.py` scores every origin in the holdout. The S&P 500 holdout is the last 250 trading days. Bitcoin's is the last 365 calendar days. A horizon is kept only when the whole window sits inside the sample, so the 60-day score has fewer origins. The loss is squared error on that log mean. Diebold-Mariano statistics use a Newey-West variance with the Bartlett kernel and `h - 1` lags. A negative statistic means the regime model has the smaller loss. The p-value is two-sided normal.

Four baselines are fit only on data before the target window. HAR(1/5/22) is an ordinary least-squares regression of the log-mean target on an intercept, log daily variance, and the log of the 5-day and 22-day mean variance. The long-run mean is the average of those same completed targets. EWMA with decay 0.94 is reported as the log of its flat variance forecast. GARCH(1,1) with Student-t errors, fit with `arch` on the returns up to the origin, is reported as the log of the mean analytic variance over the horizon.

The regime model is a 3-state Markov-switching HAR on daily `log(r^2)`, not the sticky HDP-SLDS. The Gibbs step uses one shared HAR slope, with a separate intercept and innovation scale in each regime. Draws whose companion spectral radius is 0.999 or higher are rejected. The Dirichlet prior on each transition row has mean self-stay 0.98 (`alpha = 1`, `kappa = 97`). The sampler produces regime probabilities and a transition matrix.

The scored forecast does not simulate a path. For each horizon, each retained draw gets its own direct regression of `log(mean r^2)` over the next `h` days on an intercept and three lags of daily `log(r^2)`: the previous day, the mean of the previous 5, and the mean of the previous 22. Each regime has its own intercept and its own three slopes. An origin is weighted by the average of the filtered distribution after 1 through `h` steps of that draw's transition matrix. A regime whose weights sum to less than 30 uses the pooled regression. The reported forecast is the average, over draws, of that weighted prediction minus the draw's mean residual. The residual is computed only on origins whose `h`-day window has already ended, so the correction never sees the origin being scored. Nothing is exponentiated.

Parameters are refit every 21 origins on a trailing 4-year window: 1008 trading days for the S&P 500 and 1460 calendar days for bitcoin. Each refit is 4 chains of 2000 sweeps from seed 23. Forecasts use 40 evenly spaced draws from the second half of each chain, 160 draws in total. The sweep count was raised until the worst refit on both holdouts had R-hat below 1.1. Nothing in the prior, the window, or the refit schedule was chosen on the holdout.

The "Simulated" column is the previous path forecast, kept for comparison and not re-estimated. That run drew 25 Gaussian paths per retained draw, exponentiated them, and averaged `log(mean r^2)`. It used 3 chains of 500 sweeps. Its HAR, EWMA, and mean columns match the tables below at three decimals. Its GARCH column had 10 Bitcoin failures; this run has 9, so the GARCH numbers below are this run's.

Exact zero returns, five in the S&P file and one in the Bitcoin file, are replaced in `log(r^2)` by the smallest positive square in the estimation sample.

```bash
uv run python -m examples.regime_volatility sp500
uv run python -m examples.regime_volatility btc
```

Regime draws come from `sample_regime_windows`, and the forecast filter is `filter_regime_draws`. The baselines stay NumPy. An earlier NumPy Gibbs sampler on this same protocol took 312.5 s on the S&P 500 and 651.1 s on bitcoin. See the [inference](inference.md) page.

### S&P 500

![20-day log mean variance, regime forecast and baselines](img/sp500_regime_forecast.png)

![RMSE by horizon](img/sp500_regime_horizon.png)

The run took 41.2 s. GARCH converged at every origin. The worst refit had R-hat 1.023. Every refit is between 1.006 and 1.023.

RMSE of log mean squared return. Direct is this run. Simulated is the previous path forecast. The lowest value on a row is the winner.

| Lead (days) | Origins | Direct | Simulated | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 250 | 2.708 | 2.750 | 2.706 | 3.097 | 3.152 | 2.696 |
| 5 | 246 | 0.855 | 0.891 | 0.820 | 0.856 | 0.901 | 0.807 |
| 20 | 231 | 0.567 | 0.850 | 0.549 | 0.568 | 0.618 | 0.480 |
| 60 | 191 | 0.393 | 0.992 | 0.419 | 0.450 | 0.516 | 0.307 |

Diebold-Mariano statistic of the direct regime model against each baseline, with the two-sided p-value in parentheses. The simulated column is not in this test.

| Lead (days) | vs HAR | vs mean | vs EWMA | vs GARCH |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.14 (0.887) | 0.61 (0.540) | -4.61 (4.0e-6) | -4.89 (1.0e-6) |
| 5 | 1.08 (0.280) | 1.21 (0.227) | -0.05 (0.959) | -1.18 (0.239) |
| 20 | 0.50 (0.618) | 2.39 (0.017) | -0.01 (0.989) | -1.26 (0.209) |
| 60 | -0.86 (0.388) | 1.89 (0.059) | -1.18 (0.238) | -1.57 (0.115) |

The long-run mean still has the lowest RMSE at every horizon. The direct forecast is not distinguishable from HAR at any lead. At 60 days its RMSE is below HAR, EWMA, and GARCH, and the gap against the mean is not significant at 5%. It beats one-day EWMA and GARCH. The mean error is +0.102, +0.107, +0.101, and +0.152 at leads 1, 5, 20, and 60. The simulated forecast's errors on those same leads were +0.103, +0.298, +0.650, and +0.929, and that gap grew with the horizon. EWMA's mean error still goes from +1.519 at one day to +0.001 at 60 days, because a one-day EWMA or GARCH number is the log of a variance and the target is the log of a squared return.

![Filtered regime probabilities and the one-day target](img/sp500_regime_probs.png)

On the first refit the posterior mean self-stay is 0.815, 0.952, and 0.896, so the implied runs last 5.4, 20.8, and 9.6 days. The prior mean was 0.98. Later refits stay in that neighborhood: the average self-stay across the three regimes drifts from 0.888 to 0.864. Ordered intercepts on the first refit are -5.791, -4.291, and -3.196, with shares 0.244, 0.284, and 0.472. The shared slopes are -0.101, 0.089, and 0.628, and the companion radius is 0.962. Every regression draw met the radius constraint. The second-half complete-data log likelihood on that refit has mean -2560 and standard deviation 38.

![Four chains on the first refit](img/sp500_regime_convergence.png)

### Bitcoin

![20-day log mean variance, regime forecast and baselines](img/btc_regime_forecast.png)

![RMSE by horizon](img/btc_regime_horizon.png)

The run took 76.1 s. GARCH did not converge at 9 origins. Those origins are omitted from the GARCH RMSE and from the DM test against GARCH, and they stay in the other columns. The worst refit had R-hat 1.068. The other seventeen refits are between 1.005 and 1.041.

| Lead (days) | Origins | Direct | Simulated | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 365 | 2.638 | 2.673 | 2.610 | 3.133 | 3.323 | 2.691 |
| 5 | 361 | 1.021 | 1.109 | 1.022 | 1.121 | 1.274 | 1.162 |
| 20 | 346 | 0.742 | 0.998 | 0.797 | 0.849 | 0.985 | 0.900 |
| 60 | 306 | 0.597 | 0.983 | 0.718 | 0.791 | 1.074 | 0.785 |

| Lead (days) | vs HAR | vs mean | vs EWMA | vs GARCH |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.23 (0.217) | -1.43 (0.153) | -6.19 (5.9e-10) | -7.22 (5.3e-13) |
| 5 | -0.03 (0.978) | -2.47 (0.013) | -2.11 (0.035) | -3.58 (3.5e-4) |
| 20 | -1.55 (0.121) | -1.97 (0.049) | -1.52 (0.130) | -3.52 (4.3e-4) |
| 60 | -1.78 (0.075) | -1.46 (0.145) | -1.43 (0.153) | -3.46 (5.3e-4) |

HAR has the lowest one-day RMSE. At five days the direct forecast and HAR differ by 0.001 (DM -0.03, p = 0.978). At 20 and 60 days the direct forecast is the lowest of the six columns. Those gaps against HAR have p = 0.121 and p = 0.075. The direct forecast beats the mean at 5 and 20 days, EWMA at 1 and 5 days, and GARCH at every horizon. Its mean error is +0.102, +0.174, +0.210, and +0.225. The simulated forecast's errors were +0.099, +0.327, +0.627, and +0.805.

![Filtered regime probabilities and the one-day target](img/btc_regime_probs.png)

The first refit's posterior self-stay is 0.852, 0.957, and 0.903, with implied runs of 6.7, 23.5, and 10.4 days. Ordered intercepts are -5.800, -4.253, and -3.336, with shares 0.247, 0.398, and 0.355. The shared slopes are 0.049, -0.210, and 0.689, and the companion radius is 0.957. The second-half log likelihood has mean -3769 and standard deviation 68. The average self-stay drifts from 0.904 on the first refit to 0.877 on the last.

![Four chains on the first refit](img/btc_regime_convergence.png)

### Where this still loses

The direct regression removes the horizon-growing bias of the simulated path. What remains is smaller. Daily `log(r^2)` is still a noisy series: the one-day RMSE is about 2.6 to 2.7, and the posterior self-stay settles near 0.86 to 0.90 rather than the prior mean of 0.98, so a typical run is days to a few weeks. On the S&P 500 the long-run mean wins every horizon, and the direct model is not separable from HAR. On bitcoin the direct model has the best point forecast at 20 and 60 days, the win over GARCH is clear, and the win over HAR is not significant at 5%. A positive mean error of about 0.10 on the S&P 500 and 0.10 to 0.22 on bitcoin is still there after the training-window correction.

A separate check, with one HAR fit on lags 5, 22, and 66 before the holdout and an EWMA bias correction fit on that same training span, matches the rolling-origin note supplied with the data: S&P 500 lead 20 is EWMA 0.562, HAR 0.522, mean 0.481, and lead 60 is 0.451, 0.375, 0.310. Bitcoin lead 20 is 0.845, 0.762, 0.910, and lead 60 is 0.789, 0.684, 0.792. The tables above are the stricter protocol: HAR(1/5/22) refit at each origin, and EWMA and GARCH with no additive correction.

An earlier single-origin sticky HDP-SLDS, scored on overlapping realized volatility rather than this log-mean target, is not part of this package. On that score EWMA won the longer S&P leads, and the SLDS won Bitcoin leads 20 and 60.
