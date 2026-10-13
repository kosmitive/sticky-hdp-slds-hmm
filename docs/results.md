# Pendulum fit and forecast

## Forecast

```bash
uv run python examples/pendulum_forecast.py
```

![Fitted window, 90% forecast band, and held-out observations for the largest release](img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](img/pendulum_forecast_horizon.png)

The forecast uses the same training series, seed, and hyperparameters as the fit below. Eighty further steps are integrated with continuation seed 12 and are never shown to the sampler. The chain is 300 blocked sweeps, sequential mode updates off, and the sampler took 6.8 s. It starts at the prior means: the small-angle map, a process covariance of `0.15 I`, and the observations as the latent path. Sweeps 151–300 each contribute one future: the sweep's sampled terminal state and mode, then modes from the sticky transition row, then latent states and observations. That is 150 predictive trajectories.

Against the noisy holdout, pooled over all four traces and both channels, the predictive mean has RMSE 0.162 on the first 10 steps and 1.553 over the full 80. Repeating the last observation scores 1.446 and 3.032 on those same windows. The central 90% band covers 97.5% of the first 10 steps and 92.0% of the full horizon. An earlier sequential sampler of this holdout, which updated one mode site at a time, scored RMSE 1.456 and 94.5% coverage over the 80 steps (0.148 and 98.8% on the first 10) and took 107.6 s. An earlier blocked chain that started from a random initialization scored 0.149 and 1.53, with coverage 98.75% and 91.6%.

Angle and angular velocity coverage over the full horizon are 91.9% and 92.2%. The three smaller releases have holdout RMSE 0.419, 0.578, and 1.064, with coverage 99.4%, 99.4%, and 100%. The largest release, 2.35 rad, has RMSE 0.223 on the first 10 steps (coverage 100%) and RMSE 2.830 over the horizon (coverage 69.4%). In 20-step blocks its RMSE is 0.241, 1.574, 3.084, and 4.471, and the band coverage is 100%, 75%, 62.5%, and 40%. The prior was not adjusted to the holdout.

## Fit

```bash
uv run python examples/pendulum_demo.py
```

The script integrates a nonlinear pendulum with the symplectic Euler method (`g/l = 9.81`) and fits the sticky HDP-SLDS. Four traces are released from rest at 0.45, 1.05, 1.70, and 2.35 rad. Each trace has 160 steps of `dt = 0.05` s. The observations are the angle and the angular velocity, with standard deviations 0.04 and 0.12, and the velocity also receives a process disturbance of standard deviation 0.01. The emission matrix is the identity, so piecewise-linear modes are asked to track the angle dependence of `sin(theta)`.

The truncation is 5 modes. Prior means are `alpha + kappa = 12`, `rho = 0.92`, and `gamma = 1.5`. The matrix-normal mean is the small-angle map of the same integrator, with column precision `40 I`. Those scales belong to this series. A different length, noise level, or sampling rate needs its own prior.

The figures are the blocked sampler: seed 11, 300 sweeps, sequential mode updates off, 6.6 s. The complete-data log joint starts at 2109, the value of the prior-mean state, and ends at 3489. The four quarter means are 3462, 3516, 3540, and 3529. Over sweeps 151–300 the mean is 3534, the standard deviation is 43, and the least-squares slope is -0.07 per sweep. The occupied-mode count on the last sweep is 2. Across the chain it ranges from 2 to 5, and the second-half average is 2.05.

The last sample splits the swing into two sticky regimes. The mode whose mean absolute angle is 0.614 has restoring entry `A_{2,1} = -0.424` and a mean dwell of 30.6 steps. The small-angle integrator has `A_{2,1} = -dt * g/l = -0.491`. The mode whose mean absolute angle is 1.877 has `A_{2,1} = -0.217` and a mean dwell of 9.4 steps, the softer restoring force of a large swing. The posterior mean of the latent state over the second half of the chain has RMSE 0.056 against the noise-free trajectory. The raw observations have RMSE 0.089.

![Log joint and occupied modes across 300 sweeps](img/pendulum_convergence.png)

![Observed, noise-free, and posterior-mean angle and angular velocity](img/pendulum_reconstruction.png)

![Final mode sample against the noise-free angle](img/pendulum_modes.png)

## Limits

The chain mixes inside a band; it does not freeze. Two modes are what the second half of this chain keeps, not a claim that a pendulum has two linear regimes. The forecast degrades on the largest release because a linear mode fitted on the observed swing does not carry that trajectory for a full extra 4 seconds. Coverage above 90% on the smaller releases means those bands are conservative.
