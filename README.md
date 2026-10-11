# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829). The result below is an 80-step forecast of a noisy pendulum that the sampler never sees.

## Forecast

![Fitted window, 90% forecast band, and held-out observations for the largest release](docs/img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](docs/img/pendulum_forecast_horizon.png)

`examples/pendulum_forecast.py` fits the published series (seed 11, 300 sweeps) and scores 150 futures from sweeps 151–300. Pooled over four traces and both channels, the predictive mean has RMSE 0.148 on the first 10 steps and 1.456 over all 80. Repeating the last observation scores 1.446 and 3.032 on those same windows. The central 90% band covers 98.8% of the first 10 steps and 94.5% of the horizon.

The 2.35 rad release does not hold that rate. Its 90% band covers 78.8% of the horizon and 42.5% of the last 20-step block.

## S&P 500 and bitcoin

`examples/sp500_forecast.py` and `examples/btc_forecast.py` fit the same sampler to daily log returns and log 20-day realized volatility. Prices themselves are not the observations. The last 250 days are held out. On the S&P 500 holdout the return RMSE is 0.667 against 0.661 for predicting zero, and the volatility RMSE is 0.441 against 0.440 for a flat trailing value and 0.419 for EWMA 0.94. On bitcoin the return RMSE is 0.685 against 0.680 for zero, and the volatility RMSE is 0.950 against 0.862 trailing and 0.841 for EWMA. A GARCH(1,1) Student-t forecast, fit on the training returns only, is worse than those on both series. The return forecasts do not beat zero. Figures and the rest of the scores are on the [volatility page](https://kosmitive.github.io/sticky-hdp-slds-hmm/volatility/).

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

`uv sync` installs the package and the `dev` group (pytest, ruff, pre-commit). Add the docs group when you want the site:

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
| `examples/sp500_forecast.py` | S&P 500 log-return and volatility forecast |
| `examples/btc_forecast.py` | Bitcoin log-return and volatility forecast |
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
