# Volatility forecast

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

The snapshot `examples/data/sp500_daily.csv` is the Yahoo Finance `^GSPC` daily close, downloaded 2026-10-11. The fit uses 2005-02-01 through 2025-10-10 (5207 days) and holds out 2025-10-13 through 2026-10-09 (250 trading days). The run took 387.5 s.

RMSE of log 20-day realized volatility:

| Lead (days) | SLDS | Trailing 20-day | EWMA 0.94 | GARCH-t | Lowest RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.069 | 0.094 | 0.074 | 0.048 | GARCH |
| 5 | 0.082 | 0.101 | 0.041 | 0.064 | EWMA |
| 20 | 0.154 | 0.178 | 0.041 | 0.215 | EWMA |
| 60 | 0.186 | 0.181 | 0.163 | 0.451 | EWMA |
| 250 | 0.228 | 0.228 | 0.217 | 0.608 | EWMA |

GARCH wins day 1 (0.048). That day still has 19 known returns in the window, so the score is mostly a one-step variance forecast. From lead 5 on, EWMA wins, and at lead 20, the first window with no training returns, EWMA (0.041) is well ahead of GARCH (0.215) and the SLDS (0.154). The SLDS does not win a lead. Over 250 days EWMA is 0.217, the SLDS and the trailing value tie at 0.228, and GARCH is 0.608.

The SLDS 90% band covers every holdout day. The GARCH 90% band, the 5th to 95th percentile of 2000 simulated paths, also covers every holdout day. Both bands are wide. The GARCH fit has alpha 0.130, beta 0.864, persistence 0.993, and Student-t degrees of freedom 5.83.

The return RMSE is 0.667 against 0.661 for predicting zero.

![Last sweep, colored from calm to turbulent](img/sp500_modes.png)

Labels in that figure are the last sweep, sorted so mode 0 has the lowest mean log volatility. Mean annualized volatility in those five slices is about 10%, 10%, 22%, 27%, and 23%. The two calm slices cover 18% and 37% of the training days. The 27% slice covers 3%. The annualized averages are not strictly increasing, because the sort is on mean log-volatility and a short slice can have a higher average after the exponential. What separates is a calm band near 10% and an elevated band near 23%, including the 2008 and 2020 declines. That is a coarse calm-versus-turbulent split, not five regimes. Every sweep in the second half occupies all 5 modes, so the truncation is full. The log joint over sweeps 151–300 has mean 10609, standard deviation 192, and slope -2.26 per sweep.

![Log joint and occupied modes](img/sp500_convergence.png)

## Bitcoin

![Annualized and log volatility, with the 90% band, against the holdout](img/btc_forecast.png)

![RMSE of log volatility through each lead](img/btc_forecast_horizon.png)

The snapshot `examples/data/btc_daily.csv` is the Yahoo Finance `BTC-USD` daily close, downloaded 2026-10-11. A later fetch did not replace it: Yahoo answered 429 and Stooq answered with a browser check. The file has a close on every calendar day from 2014-09-16 through 2026-10-10, with no gaps. After the 30-day window the fit uses 2014-10-16 through 2025-10-10 (4013 days) and holds out 2025-10-11 through 2026-10-10 (365 days). The same prior and seed as the S&P 500 are used. Nothing was retuned. The run took 297.1 s.

RMSE of log 30-day realized volatility. Leads 1, 5, and 20 still contain known training returns. Lead 60 does not.

| Lead (days) | SLDS | Trailing 30-day | EWMA 0.94 | GARCH-t | Lowest RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.047 | 0.055 | 0.034 | 0.019 | GARCH |
| 5 | 0.047 | 0.073 | 0.016 | 0.042 | EWMA |
| 20 | 0.042 | 0.096 | 0.079 | 0.217 | SLDS |
| 60 | 0.070 | 0.143 | 0.091 | 0.407 | SLDS |
| 365 | 0.404 | 0.298 | 0.314 | 1.112 | trailing |

GARCH wins day 1 (0.019). EWMA wins lead 5 (0.016). The SLDS wins lead 20 (0.042) and lead 60 (0.070). Lead 60 is the first of these horizons whose 30-day window contains no training returns, and there the SLDS beats EWMA (0.091), the trailing value (0.143), and GARCH (0.407). Over all 365 days the trailing 30-day value wins (0.298), ahead of EWMA (0.314) and the SLDS (0.404). GARCH is the worst of the four on that full window. Its persistence rounds to 1, so the variance forecast barely mean-reverts. The GARCH fit has alpha 0.107, beta 0.893, and Student-t degrees of freedom 3.19.

Both 90% bands cover every one of the first 60 holdout days. Over all 365 days the SLDS band covers 78.4% and the GARCH band covers 98.4%. The GARCH coverage is high because the band is wide: read it next to the RMSE of 1.112.

The return RMSE is 0.661 against 0.656 for predicting zero.

![Last sweep, colored from calm to turbulent](img/btc_modes.png)

The same last-sweep ordering, now in `sqrt(365)` units, gives mean annualized volatilities of about 34%, 43%, 47%, 62%, and 79%. The 43% slice covers 55% of training days. The 79% slice covers 21%. The 34% and 62% slices are thin. Relative to bitcoin's own history that is a calm-versus-turbulent split, not five separate regimes. Every sweep in the second half occupies all 5 modes. Sweeps 151–300 have log-joint mean 9695, standard deviation 237, and slope -4.39 per sweep, so that half is still moving.

![Log joint and occupied modes](img/btc_convergence.png)

## How the series is built

Daily closes are not the observations. A price level trends, so a linear dynamical mode spends its capacity on that drift. The return is `r_t = log(P_t / P_{t-1})`. On the S&P 500 those returns fall on trading days. On bitcoin the file has a close every calendar day, so the return is from one calendar day to the next. The volatility channel is the log of the root mean square over the dataset's window. Each channel is standardized with the mean and standard deviation of the training window only.

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
