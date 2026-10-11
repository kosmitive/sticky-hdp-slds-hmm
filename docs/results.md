# Pendulum fit and forecast

## Forecast

```bash
uv run python examples/pendulum_forecast.py
```

![Fitted window, 90% forecast band, and held-out observations for the largest release](img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](img/pendulum_forecast_horizon.png)

The forecast uses the same training series, seed, and hyperparameters as the fit below. Eighty further steps are integrated with continuation seed 12 and are never shown to the sampler. Sweeps 151–300 each contribute one future: the sweep's sampled terminal state and mode, then modes from the sticky transition row, then latent states and observations. That is 150 predictive trajectories. The run took 107.6 s and reproduced the training chain (log joint 3076.00 at sweep 1 and 3495.25 at sweep 300).

Against the noisy holdout, pooled over all four traces and both channels, the predictive mean has RMSE 0.148 on the first 10 steps and 1.456 over the full 80. Repeating the last observation scores 1.446 and 3.032 on those same windows. The central 90% band covers 98.8% of the first 10 steps and 94.5% of the full horizon. Angle and angular velocity are close to that pooled rate (95.3% and 93.8%).

The pool hides a split. The three smaller releases have holdout RMSE 0.539, 0.617, and 0.813, with coverage 99.4%, 100%, and 100%, so those bands are wider than the noise. The largest release, 2.35 rad, has RMSE 0.203 on the first 10 steps (coverage 100%) and RMSE 2.675 over the horizon (coverage 78.8%). In 20-step blocks its RMSE is 0.223, 1.516, 2.930, and 4.205, and the band coverage is 100%, 97.5%, 75%, and 42.5%. The last second of that swing leaves the band. The prior was not adjusted to the holdout.

## Fit

```bash
uv run python examples/pendulum_demo.py
```

The script integrates a nonlinear pendulum with the symplectic Euler method (`g/l = 9.81`) and fits the sticky HDP-SLDS. Four traces are released from rest at 0.45, 1.05, 1.70, and 2.35 rad. Each trace has 160 steps of `dt = 0.05` s. The observations are the angle and the angular velocity, with standard deviations 0.04 and 0.12, and the velocity also receives a process disturbance of standard deviation 0.01. The emission matrix is the identity, so piecewise-linear modes are asked to track the angle dependence of `sin(theta)`.

The truncation is 5 modes. Prior means are `alpha + kappa = 12`, `rho = 0.92`, and `gamma = 1.5`. The matrix-normal mean is the small-angle map of the same integrator, with column precision `40 I`. Those scales belong to this series. A different length, noise level, or sampling rate needs its own prior.

The figures below are seed 11, 300 Gibbs sweeps, 104.7 s on the machine that produced them. The complete-data log joint leaves the initialization immediately: 3076 after sweep 1, then a band around 3500. The four quarter means are 3494, 3552, 3512, and 3548. Over sweeps 151–300 the mean is 3530, the standard deviation is 44, and the least-squares slope is +0.46 per sweep. That slope is a small fraction of the sweep-to-sweep scatter. The occupied-mode count stays at 2, with a second-half average of 2.03. A third mode shows up on isolated sweeps and is dropped again.

The last sample splits the swing into two sticky regimes. The mode whose mean absolute angle is 0.54 has restoring entry `A_{2,1} = -0.442` and a mean dwell of 30 steps. The small-angle integrator has `A_{2,1} = -dt * g/l = -0.491`. The mode whose mean absolute angle is 1.60 has `A_{2,1} = -0.258` and a mean dwell of 17 steps, the softer restoring force of a large swing. The posterior mean of the latent state over the second half of the chain has RMSE 0.057 against the noise-free trajectory. The raw observations have RMSE 0.089.

![Log joint and occupied modes across 300 sweeps](img/pendulum_convergence.png)

![Observed, noise-free, and posterior-mean angle and angular velocity](img/pendulum_reconstruction.png)

![Final mode sample against the noise-free angle](img/pendulum_modes.png)

## Limits

The chain mixes inside a band; it does not freeze. Two modes are what this prior and this truncation keep, not a claim that a pendulum has two linear regimes. The forecast degrades on the largest release because a linear mode fitted on the observed swing does not carry that trajectory for a full extra 4 seconds. Coverage above 90% on the smaller releases means those bands are conservative.
