# Inference

`sticky_hdp_slds` is the Gibbs sampler. State is an immutable pytree (`NamedTuple`). Each block is a pure function `(key, state, data) -> state`. Sweeps run under `lax.scan`. Chains, and the equal-length rolling refit windows of the volatility study, run under `vmap`. Arithmetic is float64 (`jax_enable_x64`).

Public functions and the Gibbs blocks are annotated with `jaxtyping` array shapes. The test suite checks one of those signatures at runtime with `beartype`.

## Regime HAR

The Gibbs likelihood is the daily sticky HAR: shared slopes on lags 1, 5, and 22, a separate intercept and scale in each regime, and a Dirichlet transition prior with mean self-stay 0.98. The scored forecast is a separate direct regression at each horizon. It is not inside the Gibbs likelihood.

The companion-root check is a Schur-Cohn reflection test. Scaling the distributed-lag polynomial by `radius**(k+1)` makes every companion eigenvalue lie inside radius 0.999 exactly when every reflection coefficient has absolute value below one. The stationary distribution is a linear solve.

```bash
uv run python -m examples.regime_volatility sp500
uv run python -m examples.regime_volatility btc
```

Every refit calls `sample_regime_windows`. The forecast filter is `filter_regime_draws`: the local forward recursion, jitted and mapped across draws. Origins in a refit block are positions in that one filtered series.

## Sticky HDP-SLDS

The default sweep is the blocked weak-limit sampler of Fox, Sudderth, Jordan, and Willsky, [arXiv:1003.3829](https://arxiv.org/abs/1003.3829).

- Section II-B is the sticky HDP-HMM. The expected transition row is a mixture of the global weights `beta` and a point mass on the current mode.
- Section III-B3 and Appendix C-A draw the mode path given the latent path, on a truncation of size `L`.
- Section III-B4 and Appendix C-B draw the latent path given the modes. The backward pass is the information filter in Algorithm 5. The forward pass used by the optional single-site update is Algorithm 6. The prior precision is the supplement's Equation 181.
- Algorithm 1 steps 5–7 then update the sticky HDP (`beta`, the transition rows, `alpha + kappa`, `rho`, `gamma`, with the Chinese-restaurant table counts), the matrix-normal inverse-Wishart dynamics (Section III-A1), and the shared emission covariance (Equation 31). Several traces contribute `traces * steps` emission observations.
- The ARD conditional (Section III-A2) is available with `ard=True`. The pendulum prior is matrix-normal, so the default sweep does not use it.
- Section III-B5, the single-site mode update that marginalizes the latent state, runs only when `sequential_modes=True`. The pendulum example leaves it off.

## Speed

On this machine (4 cores, Intel Xeon at 2400 MHz, float64) the removed NumPy Gibbs sampler was slower by about 6.33× on four chains of the 1460-day Bitcoin window: 1602.8 against 253.3 sweeps per second, 4.99 s against 31.58 s, with compilation excluded. The rolling-origin runs are 41.2 s against 312.5 s on the S&P 500 and 76.1 s against 651.1 s on Bitcoin.
