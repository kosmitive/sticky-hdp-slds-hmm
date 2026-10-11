# Refactor analysis: sticky HDP-SLDS

This note is a map of the Python implementation of Fox, Sudderth, Jordan, and Willsky, [arXiv:1003.3829](https://arxiv.org/abs/1003.3829), as it stands after the correctness pass in this change. It is written so a reader can see the architecture, the defects that were real, and the work that is still worth doing.

## What the code is

The model is a switching linear dynamical system whose mode chain is a sticky HDP-HMM, truncated at a finite number of modes. Each mode has a linear map `A` and a process-noise covariance `Sigma`. Observations are `y = C x + noise` with one shared emission covariance `R`. The dynamics prior is either matrix-normal inverse-Wishart (MNIW) or an automatic relevance determination (ARD) prior on the columns of `A`.

Inference is Gibbs. The conditionals follow Section III and Appendix C of the paper, together with the sticky HDP auxiliary-variable sampler in Fox et al., *A Sticky HDP-HMM with Application to Speaker Diarization*.

## Current structure

`StickyHDPSLDSHMMModel` in `model/sticky_hdp_slds_hmm.py` is a container plus nine mixins from `slds/`:

| Mixin | Responsibility |
| --- | --- |
| `slds_init` | Draw an initial mode path from `beta` and `pi` |
| `slds_kalman_filter_y` | Forward and backward information filters |
| `slds_block_sample_x` | Sample the latent path given the modes |
| `slds_block_sample_z` | Sample the whole mode path given the latents |
| `slds_seq_sample_z` | Sample one mode at a time, marginalizing the latent |
| `slds_sample_sticky_hdp_params` | Table counts, overrides, `beta`, `pi`, and the concentrations |
| `slds_mniw_prior_sample` | Conjugate dynamics update |
| `slds_ard_prior_sample` | ARD dynamics update |
| `slds_sample_r` | Shared emission covariance |

Priors live in `model/`. `model/gt_sticky_hdp_slds_hmm.py` is an experiment harness: it loads a saved ground-truth pack, copies selected variables onto an iterate model, and records errors. `misc/` holds stick breaking, categorical sampling, and NumPy/JSON I/O. `data/generate_hdp_slds_data.py` builds the packs under `data_packs/`. `test_runner/` is the original plotting driver.

Parameters are positional lists. `hdp_params` is `[alpha + kappa, rho, gamma, beta, pi]`. `slds_params` is `[A, Sigma]` or `[A, Sigma, alpha]`.

## Defects that were in the sampler

These were confirmed by reading the updates against the paper and by running a two-dimensional synthetic system. They are fixed, and `tests/test_sampler.py` locks the corrected behavior.

1. **Stick-breaking initialization crashed.** `stick_breaking_construction_dp` compared an arbitrary base measure to the string `'uniform'` with `==`. After the global weights were drawn, each transition row passed a NumPy vector into that check and `init_params` raised. Prior and posterior HDP draws could not be constructed.

2. **Sticky table counts were off by one.** The Chinese restaurant step in Algorithm 1 of the sticky HDP-HMM supplement starts the seat index at 0, so the first customer opens a table with probability 1. The code started at 1. With a single self-transition, the table count was Bernoulli and averaged well below 1. That count feeds `beta`, `gamma`, `rho`, and `alpha + kappa`.

3. **Empty truncation states were given auxiliary variables.** The posterior for `alpha + kappa` was multiplied by a Beta/Bernoulli factor for every row, including rows with no departures, using a `1e-6` stand-in for a zero count. Empty restaurants are omitted.

4. **Emission degrees of freedom added traces and time.** Equation (34) uses `r0 + T` for `T` observations. The code used `num_traces + num_steps`. The scatter matrix summed every residual, and the same array was the prior scale, so each update permanently inflated the prior.

5. **The forward information filter was initialized with the prior covariance.** Equation (181) sets the initial information matrix to `P0^{-1}`. The code stored `P0`. On the checked-in packs the latent covariance, stored in `R0.npy`, is `1e-6 I`, so the filter treated a tight parent state as an extremely vague one. With the precision initialization, one step of the stable recursion matches the direct information update.

6. **The latent parent was drawn from `N(0, I)`.** Block sampling of `x` draws an auxiliary parent and then conditions each stored state on it. The parent is now drawn from `N(0, P0)` updated by the backward message at index 0, which is the message the backward filter already computes and previously ignored.

7. **Backward mode messages used the current likelihood.** Equation (35) multiplies the local transition density once, at the sampling step. The message recursion included that density as well, and the initial mode was uniform rather than `beta`. On a two-step chain the joint frequencies now match the exact categorical posterior.

8. **Sequential mode likelihood had a stray logarithm, and it used the wrong `A`.** Equation (39) scores candidate `k` with both `A^{(k)}` and `Sigma^{(k)}`, and with `log` of the square root of a determinant ratio. The code put `log(det)` in the denominator and held `A` fixed at the mode currently stored in `z_t`. The transition into the first stored time was also dropped (`t > 1` instead of `t > 0`); that site now uses `beta`.

9. **MNIW draws used the precision matrices as if they were covariances.** The sufficient statistics correctly treat `K` as the column precision in Equation (21). `matrix_normal.rvs` was called with `Sigma^{-1}` and `S_xx` instead of `Sigma` and `S_xx^{-1}`, and prior draws passed `K` as the column covariance. Posterior samples of `A` now concentrate on a known linear map. Prior draws use `K^{-1}` as the column covariance so they match that update.

10. **Regression degrees of freedom counted the first state.** `z_0` has no stored parent, and the scatter only contains pairs `t >= 1`. The inverse-Wishart degrees of freedom included `z_0` anyway. They now equal the number of pairs.

11. **ARD posterior did not match Equations (30) and (31).** The Gaussian update multiplied by `Sigma` instead of `Sigma^{-1}`, put `1/alpha` on the prior precision diagonal, and reshaped the column-major vector in C order. The gamma rate omitted the factor `1/2`. Separately, `scipy.stats.gamma.rvs(shape, 1/rate)` passes `1/rate` as a location. A column with a huge precision is now the one that collapses, the posterior mean recovers a non-symmetric `A`, and prior precisions have mean `shape/rate`.

12. **ARD precisions were dropped on load.** `data['prior'] is 'ard'` is false for a string produced by `json.loads`. Reloading an ARD model kept the prior object and discarded `alpha`. The comparison is `==`. The checked-in ARD pack now loads three dynamics arrays.

13. **`rounds_hdp` was written into `rounds_ard`.** A test configuration that asked for extra HDP sweeps changed the ARD repeat count and left the HDP repeat at 1.

14. **The latent log joint kept only the last mode.** The accumulator was reset inside the mode loop, and the index arithmetic did not address the stored pairs. The joint now sums every transition `t >= 1`, every emission, every Markov step, and the `beta` probability of the first mode.

15. **Emission noise and the sticky override shared `self.w`.** Auxiliary sampling replaced the observation noise with a length-`K` override count. Overrides are stored on `override_counts`. `simulate` also saved a flat noise array that still contained the discarded initial time. New simulations store `w` with the same shape as `y`. The checked-in packs were written by the old simulator, so their `w` arrays stay length `num_traces * (num_steps + 1)`.

`rho` and `gamma` were checked against the supplement and left in place. The `rho` update is the Beta posterior in (D.11), whose failure count is the total number of tables minus the overrides. The `gamma` update is the auxiliary-variable mixture in (D.8).

## Structural problems

**Naming.** `hdp_params[0]` through `hdp_params[4]` and `slds_params[0]` through `slds_params[2]` are the real API. `fn_R0` is the filename `P0.npy` and `fn_P0` is `R0.npy`. Load and store both swap them, and the checked-in packs follow that swap (`P0.npy` is the 2 by 2 emission scale, `R0.npy` is the 3 by 3 latent covariance), so the names were not changed. `StickyHDPSLDSHMMConfiguration` is an experiment runner rather than a second model. Several files spell “ground truth” as `groundtrouth`.

**Duplication.** `__split_into_regression_problems` is copied on the MNIW and ARD mixins. Method resolution uses the MNIW copy. The original experiment file repeats the same flag dictionary for every case. The MIT header is pasted into every module.

**Module shape.** The mixin list is the sampler. There is no object that represents “one Gibbs sweep” with an explicit order, so call sites (the experiment harness and `examples/smoke_slds.py`) each restate the order. Filters, mode sampling, and parameter sampling cannot be tested without building a full model. `misc/`, `data/`, and `test_runner/` are importable packages now, but they are still a flat script layout underneath.

**Performance.** Message passing, likelihoods, and the ARD regression loop in Python over traces, time, and modes. Stick breaking draws 10,000 beta variables per row on every prior sample. The forward filter inverts `A` on every step, which Algorithm 6 also requires, and which fails when a sampled map is singular. The block mode sampler still evaluates Gaussian densities in probability space. Per-step normalization avoids a hard zero, and a long chain in higher dimension will still underflow.

**Numerical stability.** Small inverse-Wishart scales and information matrices near `1e6 I` (the checked-in `P0`) make the filters sensitive to symmetry error. The MNIW residual and the latent conditional covariance are symmetrized. There is no jitter, no Joseph form, and no fallback when `A` is singular. `sample_cat` adds `1e-4` and divides by `1.0001` before the Gumbel-max trick, which is not a normalized measure. The top-level concentration update still uses a `1e-6` second Beta parameter when every informative table count is zero.

**Indexing of the first state.** Stored time 0 has no parent in `x`. Block mode sampling therefore gives that site no dynamics likelihood. Block state sampling invents a parent and lets `z_0` drive the transition into `x_0`. The regression ignores that pair because the parent is not stored. `simulate` draws one extra parent and then discards it, so the first stored mode is one transition away from `beta`, while `slds_init_z` draws the first stored mode from `beta`. The filters and the regression agree with the stored arrays. They do not agree with each other about whether `z_0` generated `x_0`.

**Tests and packaging.** Before this change there was no install metadata, no README, and no assertion suite. Importing `misc/disc_sample.py`, `data/generate_hdp_slds_data.py`, or `test_runner/exec_tests.py` ran work immediately: a KL print, a full regeneration of both data packs, or the experiment driver. Those calls now sit under `__name__ == '__main__'`.

## What this change did

- Corrected the fifteen sampler defects above without replacing the Gibbs decomposition.
- Added `tests/test_sampler.py` and `examples/smoke_slds.py`. The smoke run is the install check: a 2-mode, 2-dimensional system, two sequences of length 12, three sweeps.
- Added `pyproject.toml`, package `__init__` modules, and this README, so `pip install -e ".[dev]"` and `pytest` work from a checkout.
- Documented the public container (`hdp_params`, `slds_params`, `C`, the two priors) and the corrected conditionals that callers have to get right.
- Stopped the three import-time side effects.

The algorithm’s shape is unchanged: the same mixins, the same truncation, the same information filters, the same split between block and sequential mode sampling.

## Status of the follow-up

Items 1, 2, 4, 6, 7, and 8 from the list below are done. Items 3 and 5 are deliberately left. They change the sampled distribution, and this pass does not.

1. **Done.** `StickyHDPState` and `DynamicsState` in `model/state.py` replace the positional lists. Load still reads the historical `P0.npy` / `R0.npy` swap, and it reads honest names when the array sizes distinguish them. Store keeps writing the historical names.

2. **Done.** `StickyHDPSLDSHMMModel.gibbs_step` owns the order: block modes, information filters, sequential modes, latents, sticky HDP, dynamics, emission. Flags omit a block. The experiment harness, the smoke script, and both pendulum scripts call it.

3. **Left on purpose.** The first stored state still has no parent. Resolving that convention would change numerical results and needs its own tests against a hand-computed two-step chain.

4. **Done.** Backward mode messages and the block-mode categorical are in log space. Emission and latent log joints, and the ARD normal equations, are dense linear algebra. `C^T R^{-1} C` is cached until `C` or `R` is replaced. The forward filter still inverts `A`.

5. **Left on purpose.** The forward filter is still Algorithm 6, including the inverse of `A`. A singular map still fails there.

6. **Done.** `slds/regression.py` is the only regression splitter. `test_runner/all/slds_basic_test.py` builds cases from a flag table.

7. **Done.** `data_packs/hdp_slds_mniw` and `data_packs/hdp_slds_ard` were regenerated with seed `0`, recorded in each pack as `SEED`. `w` has the same shape as `y`.

8. **Done.** GitHub Actions runs ruff and pytest on Python 3.9 and 3.13, runs the pre-commit hooks, and builds this site. Pushes to `master` deploy it with GitHub Pages.

The project is installed with uv (`uv sync`, `uv run`). Ruff lints and formats. MkDocs publishes the notes on this site.

## Pendulum check

`examples/pendulum_demo.py` runs this sampler on a noisy nonlinear pendulum: four releases, 160 steps of 0.05 s, identity emissions, truncation 5. The figures in `docs/img/` are seed 11 and 300 Gibbs sweeps. The complete-data log joint leaves initialization in the first few sweeps and then fluctuates around 3530. The chain occupies two sticky modes, and the last sample gives a stiffer restoring gain at small absolute angle than at large absolute angle. The README records the prior scales, the quarter means, and the RMSE. `examples/pendulum_forecast.py` continues those same traces for 80 steps that the sampler never sees. The predictive mean beats a last-observation baseline on the pooled horizon, and the 90% band on the largest release falls to 42.5% coverage in the last second.
