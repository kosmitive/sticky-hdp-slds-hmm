# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829). The result below is an 80-step forecast of a noisy pendulum that the sampler never sees.

## Forecast

![Fitted window, 90% forecast band, and held-out observations for the largest release](docs/img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](docs/img/pendulum_forecast_horizon.png)

`examples/pendulum_forecast.py` fits the published series (seed 11, 300 sweeps) and scores 150 futures from sweeps 151–300. Pooled over four traces and both channels, the predictive mean has RMSE 0.148 on the first 10 steps and 1.456 over all 80. Repeating the last observation scores 1.446 and 3.032 on those same windows. The central 90% band covers 98.8% of the first 10 steps and 94.5% of the horizon.

The 2.35 rad release does not hold that rate. Its 90% band covers 78.8% of the horizon and 42.5% of the last 20-step block.

## Volatility

![S&P 500 rolling-origin RMSE of log mean squared return](docs/img/sp500_regime_horizon.png)

`examples/regime_volatility.py` scores the log of the mean squared return over the next 1, 5, 20, and 60 days, at every holdout origin. The S&P 500 holdout is the last 250 trading days. Bitcoin's is the last 365 calendar days. The regime model is a 3-state sticky HAR on daily `log(r^2)`, refit every 21 origins on a trailing 4-year window. The scored forecast is a direct regression at each horizon, with a separate intercept and HAR slopes in each regime, weighted by the filtered probabilities after they are moved forward with the transition matrix. A bias correction uses only windows that have already ended. Baselines are HAR(1/5/22), the expanding mean of the target, EWMA 0.94, and GARCH(1,1) Student-t. Each baseline uses only data before the target window. Diebold-Mariano tests are on the [volatility page](https://kosmitive.github.io/sticky-hdp-slds-hmm/volatility/). A negative statistic there means the regime model has the smaller squared error.

The close files are in `examples/data/`: Yahoo Finance chart API daily closes, interval 1d, downloaded 2026-10-11. The S&P file keeps the 2001-09-10 to 2001-09-17 market closure. The Bitcoin file drops 2026-10-11, the download date. See `examples/data/README.md`.

S&P 500. The long-run mean wins every horizon. Direct is this run. Simulated is the previous path forecast and was not re-estimated.

| Lead (days) | Direct | Simulated | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.708 | 2.750 | 2.706 | 3.097 | 3.152 | 2.696 |
| 5 | 0.855 | 0.891 | 0.820 | 0.856 | 0.901 | 0.807 |
| 20 | 0.567 | 0.850 | 0.549 | 0.568 | 0.618 | 0.480 |
| 60 | 0.393 | 0.992 | 0.419 | 0.450 | 0.516 | 0.307 |

Bitcoin. The direct forecast has the lowest RMSE at 20 and 60 days. GARCH missed 9 origins.

| Lead (days) | Direct | Simulated | HAR | EWMA 0.94 | GARCH-t | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.638 | 2.673 | 2.610 | 3.133 | 3.323 | 2.691 |
| 5 | 1.022 | 1.109 | 1.022 | 1.121 | 1.274 | 1.162 |
| 20 | 0.742 | 0.998 | 0.797 | 0.849 | 0.985 | 0.900 |
| 60 | 0.598 | 0.983 | 0.718 | 0.791 | 1.074 | 0.785 |

On the S&P 500 the direct model is not distinguishable from HAR, and it beats one-day EWMA and GARCH. On bitcoin it beats the mean at 5 and 20 days and GARCH at every horizon. The gaps against HAR are not significant at 5%. The simulated path was biased high by about 0.65 and 0.93 on the S&P 500 at 20 and 60 days, and by about 0.63 and 0.81 on bitcoin. The direct forecast's mean errors on those leads are 0.10 and 0.15, and 0.21 and 0.22.

An earlier single-origin sticky HDP-SLDS comparison, on an overlapping realized-volatility target, is on the same page. On that score EWMA wins the longer S&P leads, and the SLDS wins Bitcoin leads 20 and 60.

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
| `examples/volatility_forecast.py` | Single-origin S&P 500 or Bitcoin forecast |
| `examples/regime_volatility.py` | Rolling-origin sticky regime HAR |
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
