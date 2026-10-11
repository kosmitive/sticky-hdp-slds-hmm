# Equity and bitcoin volatility

Daily closes are not the observations. A price level trends, so a linear dynamical mode spends its capacity on that drift. Both examples use the same transform.

The return is the daily log change, `r_t = log(P_t / P_{t-1})`. The second channel is the log of a 20-day realized volatility, the root mean square of the last 20 log returns. The log keeps that positive scale roughly symmetric. Each channel is then standardized with the mean and standard deviation of the training window only. The holdout does not set the units, and it is not used to choose the prior.

The model is a sticky HDP-SLDS with truncation 5, identity emission, and a weakly informative matrix-normal prior whose mean says a return is unpredictable and log volatility is persistent. Column precision is the identity, so the sample dominates. The sticky prior has mean `alpha + kappa = 8`, mean `rho = 0.8`, and mean `gamma = 2`. `gibbs_step` runs 300 sweeps from seed 23. Forecasts are draws from the second half of the chain, after the Gibbs loop, so they do not consume the sampler's random stream. The predictive mean and the central 90% band are the summary of those draws.

Volatility baselines, also denied the holdout:

- Trailing volatility holds the last training 20-day value flat.
- EWMA with decay 0.94 (RiskMetrics) is updated on training returns and then held at its last variance forecast.
- GARCH(1,1) with Student-t errors is fit by maximum likelihood on training returns. The multi-step variance forecast mean-reverts at the estimated persistence.

For each holdout day the baseline's 20-day root mean square uses realized squared returns when they fall inside the training window and the model's variance forecast after that. The first 19 holdout windows still contain known returns, so the comparison that starts at lead 20 is the one that is fully out of sample. Scores are in the standardized log-volatility units.

Returns are scored against zero, the training mean, and against repeating the last training return. A return forecast that does not beat zero is not a forecast.

```bash
uv run python examples/sp500_forecast.py
uv run python examples/btc_forecast.py
```

Each script accepts a sweep count as its only argument. The default is 300.

## S&P 500

The snapshot `examples/data/sp500_daily.csv` is the Yahoo Finance `^GSPC` daily close, downloaded 2026-10-11. Stooq's `^SPX` file answered with a browser check and was not used. The file runs from 2005-01-03 to 2026-10-09. After the 20-day window, the fit uses 2005-02-01 through 2025-10-10 (5207 days) and holds out 2025-10-13 through 2026-10-09 (250 trading days). The run took 380.2 s.

The log joint leaves the initialization immediately, from -5419 after sweep 1 to a band near 10600. Over sweeps 151–300 the mean is 10609, the standard deviation is 192, and the slope is -2.26 per sweep. That slope is a small fraction of the scatter. Every one of those sweeps occupies all 5 modes, so the truncation is full. The last sample still switches as the index moves, with calmer stretches and the 2008 and 2020 selloffs sitting in different labels, but this is not a two-regime fit.

On the holdout, standardized log-return RMSE is 0.667 for the predictive mean, 0.661 for predicting zero, and 2.414 for repeating the last return. The mean does not beat zero. The 90% band covers 96% of the return path. Standardized log-volatility RMSE is 0.441 for the model, 0.440 for trailing volatility, 0.419 for EWMA 0.94, and 0.937 for the GARCH(1,1) Student-t forecast. From lead 20 on, dropping windows that still contain training returns, those four numbers are 0.451, 0.447, 0.435, and 0.969. EWMA is the best of the four. The model's volatility band covers every holdout day, which means the band is wide. The GARCH fit on training returns has alpha 0.125, beta 0.863, persistence 0.988, and Student-t degrees of freedom 5.98. Mapped into the same log-volatility units, that path averages +0.66 while the holdout averages -0.17, so it loses to holding the last 20-day value flat.

![Log joint and occupied modes](img/sp500_convergence.png)

![Last mode sample against the index](img/sp500_modes.png)

![Holdout forecast for both channels](img/sp500_forecast.png)

![RMSE through the holdout against the baselines](img/sp500_forecast_horizon.png)

## Bitcoin

The snapshot `examples/data/btc_daily.csv` is the Yahoo Finance `BTC-USD` daily close, downloaded the same day. It runs from 2014-09-16 to 2026-10-10 and includes weekends, because the spot market does not close. After the 20-day window the fit uses 2014-10-06 through 2026-02-02 (4138 days) and holds out 2026-02-03 through 2026-10-10 (250 days). The same prior and seed are used. Nothing was retuned for bitcoin. The run took 313.8 s.

The log joint starts at -4802 and is still climbing at the end. Sweeps 151–300 have mean 8644, standard deviation 230, and slope +4.26 per sweep, and sweep 300 is the highest point of the run. The chain has not settled. The second half again occupies all 5 modes. The last sample changes through the 2018 and 2022 drawdowns, but five occupied modes is the truncation, not a discovered count.

Standardized log-return RMSE is 0.685 for the predictive mean, 0.680 for zero, and 1.321 for the last return. Again the mean does not beat zero. Return coverage is 95.6%. Standardized log-volatility RMSE is 0.950 for the model, 0.862 for trailing volatility, 0.841 for EWMA, and 1.890 for GARCH. From lead 20 on those are 0.927, 0.825, 0.821, and 1.952. The sticky model is worse than both the trailing value and EWMA. Its 90% volatility band covers 79.2% of the holdout, under the nominal 90%. The GARCH fit has alpha 0.107, beta 0.888, persistence 0.995, and degrees of freedom 3.23. Its mapped path averages +1.07 while the holdout averages -0.55, and it is the worst of the four.

![Log joint and occupied modes](img/btc_convergence.png)

![Last mode sample against the bitcoin price](img/btc_modes.png)

![Holdout forecast for both channels](img/btc_forecast.png)

![RMSE through the holdout against the baselines](img/btc_forecast_horizon.png)

## Limits

One series is not a panel, so a coverage rate is a single path and moves around. Mode labels in the figure are the last sweep, ordered by mean volatility for that draw only; labels are not aligned across sweeps. The 90% band can cover the holdout by being wide. Read it next to the RMSE. The GARCH and EWMA lines are point forecasts of volatility, not bands. A linear mode fitted on past returns is not a trading rule, and these runs were not asked to beat a market.
