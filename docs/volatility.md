# Volatility forecast

The rolling-origin score below is the log of the mean squared return over the next 1, 5, 20, and 60 days. The earlier single-origin comparison, which used an overlapping realized-volatility target, is kept further down.

## Rolling origin

`examples/regime_volatility.py` scores every origin in the holdout. The S&P 500 holdout is the last 250 trading days. Bitcoin's is the last 365 calendar days. A horizon is kept only when the whole window sits inside the sample, so the 60-day score has fewer origins. The loss is squared error on that log mean. Diebold-Mariano statistics use a Newey-West variance with the Bartlett kernel and `h - 1` lags. A negative statistic means the regime model has the smaller loss. The p-value is two-sided normal.

Four baselines are fit only on data before the target window. HAR(1/5/22) is an ordinary least-squares regression of the log-mean target on an intercept, log daily variance, and the log of the 5-day and 22-day mean variance. The long-run mean is the average of those same completed targets. EWMA with decay 0.94 is reported as the log of its flat variance forecast. GARCH(1,1) with Student-t errors, fit with `arch` on the returns up to the origin, is reported as the log of the mean analytic variance over the horizon.

The regime model is a 3-state Markov-switching HAR on daily `log(r^2)`, not the sticky HDP-SLDS. The three HAR slopes are shared. Each regime has its own intercept and innovation scale. Draws whose companion spectral radius is 0.999 or higher are rejected. The Dirichlet prior on each transition row has mean self-stay 0.98 (`alpha = 1`, `kappa = 97`). The forecast is the average, over retained Gibbs draws and 25 simulated paths per draw, of `log(mean r^2)` given the filtered regime probabilities at the origin. Parameters are refit every 21 origins on a trailing 4-year window: 1008 trading days for the S&P 500 and 1460 calendar days for bitcoin. Each refit is 3 chains of 500 sweeps from seed 23. Forecasts use 80 evenly spaced draws from the second half of each chain. The sweep count was set from the Gelman-Rubin statistic on the training window that ends at the first holdout origin. Nothing in the prior, the window, or the refit schedule was chosen on the holdout.

Exact zero returns, five in the S&P file and one in the Bitcoin file, are replaced in `log(r^2)` by the smallest positive square in the estimation sample.

```bash
uv run python examples/regime_volatility.py sp500
uv run python examples/regime_volatility.py btc
```

### S&P 500

![20-day log mean variance, regime forecast and baselines](img/sp500_regime_forecast.png)

![RMSE by horizon](img/sp500_regime_horizon.png)

The run took 425.9 s. GARCH converged at every origin. The worst refit had R-hat 1.148.

RMSE of log mean squared return. The lowest value on a row is the winner.

| Lead (days) | Origins | Regime | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 250 | 2.750 | 2.706 | 3.097 | 3.152 | 2.696 |
| 5 | 246 | 0.891 | 0.820 | 0.856 | 0.901 | 0.807 |
| 20 | 231 | 0.850 | 0.549 | 0.568 | 0.618 | 0.480 |
| 60 | 191 | 0.992 | 0.419 | 0.450 | 0.516 | 0.307 |

Diebold-Mariano statistic of the regime model against each baseline, with the two-sided p-value in parentheses.

| Lead (days) | vs HAR | vs mean | vs EWMA | vs GARCH |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.40 (0.162) | 1.83 (0.067) | -4.14 (3.5e-5) | -4.48 (7.5e-6) |
| 5 | 1.61 (0.108) | 1.95 (0.052) | 1.18 (0.240) | -0.33 (0.742) |
| 20 | 2.86 (0.004) | 3.65 (2.6e-4) | 2.71 (0.007) | 2.95 (0.003) |
| 60 | 4.89 (1.0e-6) | 5.93 (3.0e-9) | 3.71 (2.1e-4) | 5.87 (4.2e-9) |

The long-run mean has the lowest RMSE at every horizon. At one day the regime model beats EWMA and GARCH, and the gap against HAR and the mean is not distinguishable from noise. From 20 days on, every baseline beats the regime model. The regime forecast's mean error is +0.103, +0.298, +0.650, and +0.929 at leads 1, 5, 20, and 60: it sits above the realized log mean, and the gap grows with the horizon. EWMA's mean error goes the other way, from +1.519 at one day to +0.001 at 60 days. A one-day EWMA or GARCH number is the log of a variance, while the target is the log of a squared return, so those two baselines pay Jensen's gap at short horizons. The direct HAR and the mean are regressions on the target itself, and they do not.

![Filtered regime probabilities and the one-day target](img/sp500_regime_probs.png)

On the first refit the posterior mean self-stay is 0.815, 0.953, and 0.896, so the implied runs last 5.4, 21.1, and 9.6 days. The prior mean was 0.98. Later refits stay in that neighborhood: the average self-stay across the three regimes drifts from 0.888 to 0.861. Ordered intercepts on the first refit are -5.849, -4.394, and -3.304, with shares 0.261, 0.269, and 0.470. The shared slopes are -0.101, 0.074, and 0.632, and the companion radius is 0.962. Every regression draw met the radius constraint. The second-half complete-data log likelihood on that refit has mean -2563 and standard deviation 39.

![Three chains on the first refit](img/sp500_regime_convergence.png)

### Bitcoin

![20-day log mean variance, regime forecast and baselines](img/btc_regime_forecast.png)

![RMSE by horizon](img/btc_regime_horizon.png)

The run took 891.3 s. GARCH did not converge at 10 origins. Those origins are omitted from the GARCH RMSE and from the DM test against GARCH, and they stay in the other columns. The worst refit had R-hat 1.305. Three refits exceeded 1.2, so one scale or intercept on those windows has not fully settled. The other fifteen refits are between 1.039 and 1.175.

| Lead (days) | Origins | Regime | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 365 | 2.673 | 2.610 | 3.133 | 3.327 | 2.691 |
| 5 | 361 | 1.109 | 1.022 | 1.121 | 1.279 | 1.162 |
| 20 | 346 | 0.998 | 0.797 | 0.849 | 0.993 | 0.900 |
| 60 | 306 | 0.983 | 0.718 | 0.791 | 1.083 | 0.785 |

| Lead (days) | vs HAR | vs mean | vs EWMA | vs GARCH |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.35 (0.019) | -0.39 (0.696) | -5.56 (2.8e-8) | -6.58 (4.6e-11) |
| 5 | 2.44 (0.015) | -0.95 (0.344) | -0.27 (0.790) | -2.55 (0.011) |
| 20 | 4.14 (3.5e-5) | 2.38 (0.018) | 1.37 (0.171) | 0.26 (0.798) |
| 60 | 3.37 (7.4e-4) | 3.40 (6.6e-4) | 0.91 (0.363) | -1.38 (0.168) |

HAR has the lowest RMSE at every horizon. The regime model beats EWMA and GARCH at one day, and it beats GARCH at five days. Against the long-run mean at one and five days the DM tests do not separate them. HAR is ahead of the regime model at every horizon, including one day. At 20 and 60 days the regime model also loses to the mean. Its mean error is +0.099, +0.327, +0.627, and +0.805. The same high bias as on the S&P 500: the Gaussian draws of `log(r^2)`, once exponentiated and averaged, put too much weight in the right tail.

![Filtered regime probabilities and the one-day target](img/btc_regime_probs.png)

The first refit's posterior self-stay is 0.858, 0.961, and 0.886, with implied runs of 7.0, 25.9, and 8.7 days. Ordered intercepts are -5.772, -4.279, and -3.275, with shares 0.252, 0.414, and 0.334. Slopes are 0.045, -0.213, and 0.695, and the companion radius is 0.957. The second-half log likelihood has mean -3773 and standard deviation 69.

![Three chains on the first refit](img/btc_regime_convergence.png)

### Where this still loses

The overlapping 20-day and 30-day target is gone, the slopes are stationary, and each regime has an intercept. Two facts remain. Daily `log(r^2)` is a noisy series: the one-day RMSE is about 2.7, and the posterior self-stay settles near 0.86 to 0.90 rather than the prior mean of 0.98, so a typical run is days to a few weeks. The multi-day score then exposes the Gaussian emission. `exp` of those draws overstates the mean of `r^2`, and the bias grows from about 0.1 at one day to about 0.8 or 0.9 at 60 days. Direct HAR and the historical mean forecast the aggregated target in one step, so they never simulate that tail. On this sample the sticky regime HAR does not beat them at 20 or 60 days.

A separate check, with one HAR fit on lags 5, 22, and 66 before the holdout and an EWMA bias correction fit on that same training span, matches the rolling-origin note supplied with the data: S&P 500 lead 20 is EWMA 0.562, HAR 0.522, mean 0.481, and lead 60 is 0.451, 0.375, 0.310. Bitcoin lead 20 is 0.845, 0.762, 0.910, and lead 60 is 0.789, 0.684, 0.792. The tables above are the stricter protocol: HAR(1/5/22) refit at each origin, and EWMA and GARCH with no additive correction.

## Earlier single-origin score

Both series are daily closes. The target is the log of a trailing realized volatility, the root mean square of recent daily log returns. The S&P 500 uses a 20-trading-day window. Bitcoin trades every calendar day, including weekends, so its log returns are on a 7-day calendar and the window is 30 days. The figures show that forecast twice: as the log, and as an annualized volatility in percent, `100 * exp(log rv) * sqrt(days)`, with `days` 252 for the S&P 500 and 365 for bitcoin. The band is the central 90% of the posterior predictive draws. The held-out path is the realized value.

Three baselines never see the holdout. Trailing volatility holds the last training window flat. EWMA with decay 0.94 (the RiskMetrics benchmark) is updated on training returns and then held at its last variance forecast. GARCH(1,1) with Student-t errors is fit with `arch` on the training returns only. For each holdout day the root mean square uses realized squared returns inside the training sample and the variance forecast after that. The window in that average is the same window as the target: 20 days for the S&P 500 and 30 days for bitcoin. On the S&P 500 the first 19 holdout windows still contain known returns, so leads 1 and 5 are not fully out of sample and leads 20 and 60 are. On bitcoin the first 29 windows still contain known returns, so leads 1, 5, and 20 are not fully out of sample and lead 60 is.

The score is the RMSE of log realized volatility through the first 1, 5, 20, and 60 holdout days, and through the whole holdout (250 trading days, or 365 calendar days). One series means the 1-day number is a single absolute error. Nothing here was chosen on the holdout.

The return stays in the model as the second observation. On both series that forecast is close to unpredictable: it does not beat predicting zero.

```bash
uv run python examples/volatility_forecast.py sp500
uv run python examples/volatility_forecast.py btc
```

Each script accepts a sweep count as its only argument. The default is 300.

## S&P 500

![Annualized and log volatility, with the 90% band, against the holdout](img/sp500_forecast.png)

![RMSE of log volatility through each lead](img/sp500_forecast_horizon.png)

The snapshot `examples/data/sp500_daily.csv` is the Yahoo Finance chart API `^GSPC` daily close, interval 1d, downloaded 2026-10-11. The source and that date are in `examples/data/README.md`. The file runs from 1990-01-02 through 2026-10-09 (9261 rows). The only gap longer than five days is the market closure from 2001-09-10 to 2001-09-17, and that closure is kept. After the 20-day window the fit uses every remaining observation before the holdout, 1990-01-30 through 2025-10-10 (8991 days), and holds out 2025-10-13 through 2026-10-09 (250 trading days). That training window was long enough to finish, so it was not cut shorter. The run took 681.4 s.

RMSE of log 20-day realized volatility:

| Lead (days) | SLDS | Trailing 20-day | EWMA 0.94 | GARCH-t | Lowest RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.068 | 0.094 | 0.074 | 0.059 | GARCH |
| 5 | 0.072 | 0.101 | 0.041 | 0.044 | EWMA |
| 20 | 0.145 | 0.178 | 0.041 | 0.150 | EWMA |
| 60 | 0.180 | 0.181 | 0.163 | 0.350 | EWMA |
| 250 | 0.229 | 0.228 | 0.217 | 0.487 | EWMA |

GARCH wins day 1 (0.059). That day still has 19 known returns in the window, so the score is mostly a one-step variance forecast. From lead 5 on, EWMA wins, and at lead 20, the first window with no training returns, EWMA (0.041) is well ahead of GARCH (0.150) and the SLDS (0.145). The SLDS does not win a lead. Over 250 days EWMA is 0.217, the trailing value is 0.228, the SLDS is 0.229, and GARCH is 0.487.

The SLDS 90% band covers every holdout day. The GARCH 90% band, the 5th to 95th percentile of 2000 simulated paths, also covers every holdout day. Both bands are wide. The GARCH fit has alpha 0.098, beta 0.897, persistence 0.994, and Student-t degrees of freedom 6.38.

The return RMSE is 0.711 against 0.704 for predicting zero.

![Last sweep, colored from calm to turbulent](img/sp500_modes.png)

Labels in that figure are the last sweep, sorted so mode 0 has the lowest mean log volatility. Mean annualized volatility in those five slices is about 11%, 11%, 12%, 23%, and 26%. The two slices near 11% cover 24% and 38% of the training days. The 12% slice covers 3%, so the calm cluster is about 65% of the sample. The 23% slice covers 25% and the 26% slice covers 11%. The annualized averages rise with the log-volatility sort on this draw. What separates is a calm band near 11% and an elevated band near 23–26%, including the early 1990s and the 2008 and 2020 declines. That is a coarse calm-versus-turbulent split, and five occupied modes means the truncation is full. Every sweep in the second half occupies all 5 modes. The log joint over sweeps 151–300 has mean 17545, standard deviation 267, and slope -3.23 per sweep. That slope is a small fraction of the sweep-to-sweep scatter.

![Log joint and occupied modes](img/sp500_convergence.png)

## Bitcoin

![Annualized and log volatility, with the 90% band, against the holdout](img/btc_forecast.png)

![RMSE of log volatility through each lead](img/btc_forecast_horizon.png)

The snapshot `examples/data/btc_daily.csv` is the Yahoo Finance chart API `BTC-USD` daily close, interval 1d, downloaded 2026-10-11. The download ran through 2026-10-11. That last row is the download date, so it is dropped as an incomplete session. The file kept here has a close on every calendar day from 2014-09-17 through 2026-10-10 (4407 rows). After the 30-day window the fit uses every remaining observation before the holdout, 2014-10-17 through 2025-10-10 (4012 days), and holds out 2025-10-11 through 2026-10-10 (365 days). The same prior and seed as the S&P 500 are used. Nothing was retuned, and the training window was not cut shorter. The run took 296.0 s.

RMSE of log 30-day realized volatility. Leads 1, 5, and 20 still contain known training returns. Lead 60 does not.

| Lead (days) | SLDS | Trailing 30-day | EWMA 0.94 | GARCH-t | Lowest RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.012 | 0.011 | 0.004 | 0.021 | EWMA |
| 5 | 0.055 | 0.069 | 0.014 | 0.047 | EWMA |
| 20 | 0.069 | 0.106 | 0.075 | 0.219 | SLDS |
| 60 | 0.074 | 0.150 | 0.094 | 0.421 | SLDS |
| 365 | 0.409 | 0.300 | 0.316 | 1.116 | trailing |

EWMA wins day 1 (0.004) and lead 5 (0.014). The SLDS wins lead 20 (0.069) and lead 60 (0.074). Lead 60 is the first of these horizons whose 30-day window contains no training returns, and there the SLDS beats EWMA (0.094), the trailing value (0.150), and GARCH (0.421). Over all 365 days the trailing 30-day value wins (0.300), ahead of EWMA (0.316) and the SLDS (0.409). GARCH is the worst of the four on that full window (1.116). Its persistence rounds to 1, so the variance forecast barely mean-reverts. The GARCH fit has alpha 0.107, beta 0.893, and Student-t degrees of freedom 3.18.

Both 90% bands cover every one of the first 60 holdout days. Over all 365 days the SLDS band covers 85.5% and the GARCH band covers 98.4%. The GARCH coverage is high because the band is wide: read it next to the RMSE of 1.116.

The return RMSE is 0.670 against 0.657 for predicting zero.

![Last sweep, colored from calm to turbulent](img/btc_modes.png)

The same last-sweep ordering, now in `sqrt(365)` units, gives mean annualized volatilities of about 34%, 44%, 46%, 67%, and 80%. The 44% slice covers 59% of training days. The 46% slice covers 12%, and the 80% slice covers 20%. The 34% and 67% slices are thin, at 6% and 3%. Relative to bitcoin's own history that is a calm-versus-turbulent split. The 2020–2022 stretch and the late-2024 rise sit in the high slices. Every sweep in the second half occupies all 5 modes. Earlier sweeps sometimes used 4. Sweeps 151–300 have log-joint mean 9360, standard deviation 142, and slope -0.57 per sweep. That slope is small next to the scatter.

![Log joint and occupied modes](img/btc_convergence.png)

## How the series is built

Daily closes are not the observations. A price level trends, so a linear dynamical mode spends its capacity on that drift. The return is `r_t = log(P_t / P_{t-1})`. On the S&P 500 those returns fall on trading days, and the 2001-09-17 return is the move from the 2001-09-10 close across the closure. On bitcoin the file has a close every calendar day, so the return is from one calendar day to the next. The volatility channel is the log of the root mean square over the dataset's window. Each channel is standardized with the mean and standard deviation of the training window only.

The model is a sticky HDP-SLDS with truncation 5, identity emission, and a weakly informative matrix-normal prior whose mean says a return is unpredictable and log volatility is persistent. Column precision is the identity, so the sample dominates. The sticky prior has mean `alpha + kappa = 8`, mean `rho = 0.8`, and mean `gamma = 2`. `gibbs_step` runs 300 sweeps from seed 23. Forecasts are draws from the second half of the chain, after the Gibbs loop.

## How the GARCH forecast is converted

`arch` fits a zero-mean GARCH(1,1) with Student-t errors (`dist="t"`, `rescale=True`). Rescaling multiplies the returns by 100. Reported `omega` and the variance forecasts are divided by `100**2` so they are back in return units. `alpha`, `beta`, and `nu` are unchanged by that scale.

The analytic variance of the return `h` days after the last training return follows

`v_1 = omega + alpha * r_T^2 + beta * sigma_T^2`

`v_h = omega + (alpha + beta) * v_{h-1}`

When `alpha + beta` is strictly below 1, that is the same path as the closed form that mean-reverts to `omega / (1 - alpha - beta)`. `arch`'s analytic forecast matches this recursion in original units. On bitcoin the fitted persistence rounds to 1, so `v_h` barely moves, and the long-horizon score shows that.

The point forecast of the SLDS target keeps realized squares for any day in the window that is still inside the training sample, puts `v_h` in for each later day, and takes the log of the root mean square. The window is 20 trading days for the S&P 500 and 30 calendar days for bitcoin, the same window as the fitted target. It is not the average of the log.

The 90% band draws 2000 paths from the same fit, seed 23. Each shock is a Student-t draw divided by `sqrt(nu / (nu - 2))`, so it has variance 1, which is the shock the GARCH recursion expects. Each path is converted with the same mix of known squares and simulated squares. The band is the 5th and 95th percentiles. Coverage is the share of realized holdout log-volatility values inside that band.

## Limits

One series is not a panel, so a coverage rate is a single path and moves around. Mode labels are the last sweep, ordered by mean log volatility for that draw only. They are not aligned across sweeps. The 90% band can cover the holdout by being wide. Read it next to the RMSE. Trailing volatility and EWMA are point forecasts, not bands. The GARCH band is a simulated percentile, and it can be wide too. A linear mode fitted on past returns is not a trading rule, and these runs were not asked to beat a market.
