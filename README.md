# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829). The result below is an 80-step forecast of a noisy pendulum that the sampler never sees.

## Forecast

![Fitted window, 90% forecast band, and held-out observations for the largest release](docs/img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](docs/img/pendulum_forecast_horizon.png)

`examples/pendulum_forecast.py` fits the published series (seed 11, 300 sweeps) and scores 150 futures from sweeps 151–300. Pooled over four traces and both channels, the predictive mean has RMSE 0.148 on the first 10 steps and 1.456 over all 80. Repeating the last observation scores 1.446 and 3.032 on those same windows. The central 90% band covers 98.8% of the first 10 steps and 94.5% of the horizon.

The 2.35 rad release does not hold that rate. Its 90% band covers 78.8% of the horizon and 42.5% of the last 20-step block.

## Volatility

![S&P 500 annualized and log 20-day realized volatility, 90% band, and the holdout](docs/img/sp500_forecast.png)

`examples/volatility_forecast.py` forecasts log realized volatility for one daily dataset. The return is the second observation. Nothing is fit to the holdout. The baselines are the last realized-volatility window held flat, EWMA with decay 0.94, and a GARCH(1,1) Student-t fit with `arch` on the training returns. GARCH is converted with that same window: 20 trading days for the S&P 500, 30 calendar days for bitcoin. The table is RMSE of log volatility. The lowest value on each row is the winner.

The S&P 500 holds out the last 250 trading days. Bitcoin trades every day, so the series is a 7-day calendar, the window is 30 days, annualization is `sqrt(365)`, and the holdout is the last 365 days. The close files are in `examples/data/`: Yahoo Finance chart API daily closes, interval 1d, downloaded 2026-10-11. The S&P file keeps the 2001-09-10 to 2001-09-17 market closure. The Bitcoin file drops 2026-10-11, the download date. The fit uses the whole history before each holdout. See `examples/data/README.md`.

S&P 500:

| Lead (days) | SLDS | Trailing | EWMA 0.94 | GARCH-t |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.068 | 0.094 | 0.074 | 0.059 |
| 5 | 0.072 | 0.101 | 0.041 | 0.044 |
| 20 | 0.145 | 0.178 | 0.041 | 0.150 |
| 60 | 0.180 | 0.181 | 0.163 | 0.350 |
| 250 | 0.229 | 0.228 | 0.217 | 0.487 |

GARCH wins day 1. EWMA wins every longer lead. Both 90% bands cover the whole holdout, so those bands are wide.

Bitcoin, log 30-day realized volatility. Leads 1, 5, and 20 still contain known returns. Lead 60 does not.

| Lead (days) | SLDS | Trailing | EWMA 0.94 | GARCH-t |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.012 | 0.011 | 0.004 | 0.021 |
| 5 | 0.055 | 0.069 | 0.014 | 0.047 |
| 20 | 0.069 | 0.106 | 0.075 | 0.219 |
| 60 | 0.074 | 0.150 | 0.094 | 0.421 |
| 365 | 0.409 | 0.300 | 0.316 | 1.116 |

EWMA wins day 1 and lead 5. The SLDS wins leads 20 and 60. The trailing 30-day value wins the full 365 days, and GARCH is the worst there. The SLDS 90% band covers 85.5% of that holdout. The GARCH band covers 98.4%, which is a wide band next to its RMSE of 1.116.

The return forecast is close to unpredictable: 0.711 against 0.704 for predicting zero on the S&P 500, and 0.670 against 0.657 on bitcoin. The conversion, the bands, and the mode split are on the [volatility page](https://kosmitive.github.io/sticky-hdp-slds-hmm/volatility/).

Documentation: [https://kosmitive.github.io/sticky-hdp-slds-hmm/](https://kosmitive.github.io/sticky-hdp-slds-hmm/).

The sampler alternates between:

- block sampling of the latent state with a backward information-form Kalman filter
- block and sequential sampling of the mode sequence
- sticky HDP updates for the transition rows, the global weights, and the concentration parameters
- matrix-normal inverse-Wishart or automatic relevance determination updates for the mode dynamics
- an inverse-Wishart update for the shared emission covariance

`StickyHDPState` holds `alpha_plus_kappa`, `rho`, `gamma`, `beta`, and `pi`. `DynamicsState` holds `A`, `Sigma`, and the optional ARD precision `alpha`. `StickyHDPSLDSHMMModel.gibbs_step` runs one sweep in that order. Flags turn individual blocks off.

## Quick start

[uv](https://docs.astral.sh/uv/) installs the locked environment. Python 3.9 or newer is enough.

```bash
uv sync
uv run pytest
uv run python examples/smoke_slds.py
uv run pre-commit install
```

`uv sync` installs the package and the `dev` group (pytest, ruff, pre-commit, and the `examples` group). The volatility scripts use `arch` from that examples group. Add the docs group when you want the site:

```bash
uv sync --group docs
uv run mkdocs serve
```

The smoke script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, a self-transition weight `rho` in `(0, 1)`, and a large diagonal entry for a mode that never switches.

## Layout

| Path | Role |
| --- | --- |
| `model/` | Priors, typed parameter state, and the model container |
| `slds/` | One mixin per block of Gibbs updates |
| `misc/` | Categorical sampling, stick breaking, I/O |
| `data/` | Synthetic data generation |
| `data_packs/` | MNIW and ARD datasets regenerated with seed `0` |
| `examples/smoke_slds.py` | Short end-to-end run |
| `examples/pendulum_demo.py` | Noisy pendulum, with the figures below |
| `examples/pendulum_forecast.py` | 80-step forecast from that fit |
| `examples/volatility_forecast.py` | S&P 500 or Bitcoin volatility forecast |
| `examples/sp500_forecast.py` | S&P 500 entry point |
| `examples/btc_forecast.py` | Bitcoin entry point, 30-day window |
| `tests/` | Regression tests for the sampler |
| `docs/` | This site's source |

## Results

### Noisy pendulum

```bash
uv run python examples/pendulum_demo.py
```

The script integrates a nonlinear pendulum with the symplectic Euler method (`g/l = 9.81`) and fits the sticky HDP-SLDS. Four traces are released from rest at 0.45, 1.05, 1.70, and 2.35 rad. Each trace has 160 steps of `dt = 0.05` s. The observations are the angle and the angular velocity, with standard deviations 0.04 and 0.12, and the velocity also receives a process disturbance of standard deviation 0.01. The emission matrix is the identity, so piecewise-linear modes are asked to track the angle dependence of `sin(theta)`.

The truncation is 5 modes. Prior means are `alpha + kappa = 12`, `rho = 0.92`, and `gamma = 1.5`. The matrix-normal mean is the small-angle map of the same integrator, with column precision `40 I`. Those scales belong to this series. A different length, noise level, or sampling rate needs its own prior.

The figures below are seed 11, 300 Gibbs sweeps, 104.7 s on the machine that produced them. The complete-data log joint leaves the initialization immediately: 3076 after sweep 1, then a band around 3500. The four quarter means are 3494, 3552, 3512, and 3548. Over sweeps 151–300 the mean is 3530, the standard deviation is 44, and the least-squares slope is +0.46 per sweep. That slope is a small fraction of the sweep-to-sweep scatter. The occupied-mode count stays at 2, with a second-half average of 2.03. A third mode shows up on isolated sweeps and is dropped again.

The last sample splits the swing into two sticky regimes. The mode whose mean absolute angle is 0.54 has restoring entry `A_{2,1} = -0.442` and a mean dwell of 30 steps. The small-angle integrator has `A_{2,1} = -dt * g/l = -0.491`. The mode whose mean absolute angle is 1.60 has `A_{2,1} = -0.258` and a mean dwell of 17 steps, the softer restoring force of a large swing. The posterior mean of the latent state over the second half of the chain has RMSE 0.057 against the noise-free trajectory. The raw observations have RMSE 0.089.

![Log joint and occupied modes across 300 sweeps](docs/img/pendulum_convergence.png)

![Observed, noise-free, and posterior-mean angle and angular velocity](docs/img/pendulum_reconstruction.png)

![Final mode sample against the noise-free angle](docs/img/pendulum_modes.png)

### Forecast

```bash
uv run python examples/pendulum_forecast.py
```

The forecast uses the same training series, seed, and hyperparameters as the fit above. Eighty further steps are integrated with continuation seed 12 and are never shown to the sampler. Sweeps 151–300 each contribute one future: the sweep’s sampled terminal state and mode, then modes from the sticky transition row, then latent states and observations. That is 150 predictive trajectories. The run took 107.6 s and reproduced the training chain (log joint 3076.00 at sweep 1 and 3495.25 at sweep 300). The headline scores and both figures are at the top of this page.

Angle and angular velocity coverage over the full horizon are 95.3% and 93.8%. The three smaller releases have holdout RMSE 0.539, 0.617, and 0.813, with coverage 99.4%, 100%, and 100%, so those bands are wider than the noise. The largest release, 2.35 rad, has RMSE 0.203 on the first 10 steps (coverage 100%) and RMSE 2.675 over the horizon (coverage 78.8%). In 20-step blocks its RMSE is 0.223, 1.516, 2.930, and 4.205, and the band coverage is 100%, 97.5%, 75%, and 42.5%. The last second of that swing leaves the band. The prior was not adjusted to the holdout.

The chain mixes inside a band; it does not freeze. Two modes are what this prior and this truncation keep. The forecast on the 2.35 rad release is the case where a linear mode fitted on the observed swing does not carry the trajectory for the full extra horizon.

## Tests

`uv run pytest` runs the checks in `tests/`. They cover stick-breaking initialization, the sticky table counts, emission degrees of freedom, the Kalman initialization, the two mode samplers, both dynamics posteriors, save/load of the ARD precision, the typed parameter state, `gibbs_step`, and the smoke run.

`uv run python -m test_runner.exec_tests` is the original experiment driver. It writes plots under `test_runs/` and prints sampler traces.

## License

MIT. See [LICENSE](LICENSE).
