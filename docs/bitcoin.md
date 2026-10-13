# Bitcoin

The protocol is on the [volatility](volatility.md) page.

![20-day log mean variance, regime forecast and baselines](img/btc_regime_forecast.png)

![RMSE by horizon](img/btc_regime_horizon.png)

The run took 76.1 s. GARCH did not converge at 9 origins. Those origins are omitted from the GARCH RMSE and from the DM test against GARCH, and they stay in the other columns. The worst refit had R-hat 1.068. The other seventeen refits are between 1.005 and 1.041.

RMSE of log mean squared return. Direct is this run. Simulated is the previous path forecast. The lowest value on a row is the winner.

| Lead (days) | Origins | Direct | Simulated | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 365 | 2.638 | 2.673 | 2.610 | 3.133 | 3.323 | 2.691 |
| 5 | 361 | 1.021 | 1.109 | 1.022 | 1.121 | 1.274 | 1.162 |
| 20 | 346 | 0.742 | 0.998 | 0.797 | 0.849 | 0.985 | 0.900 |
| 60 | 306 | 0.597 | 0.983 | 0.718 | 0.791 | 1.074 | 0.785 |

Diebold-Mariano statistic of the direct regime model against each baseline, with the two-sided p-value in parentheses. The simulated column is not in this test.

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

The [S&P 500](sp500.md) holdout follows.
