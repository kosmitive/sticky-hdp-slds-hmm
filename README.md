# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829).

The sampler alternates between:

- block sampling of the latent state with a backward information-form Kalman filter
- block and sequential sampling of the mode sequence
- sticky HDP updates for the transition rows, the global weights, and the concentration parameters
- matrix-normal inverse-Wishart or automatic relevance determination updates for the mode dynamics
- an inverse-Wishart update for the shared emission covariance

## Install

From this repository:

```bash
pip install -e ".[dev]"
```

Runtime dependencies are NumPy, SciPy, and Matplotlib. The `dev` extra adds pytest.

## Run a small example

```bash
python examples/smoke_slds.py
```

The script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, a self-transition weight `rho` in `(0, 1)`, and a large diagonal entry for a mode that never switches.

The same entry point is covered by the smoke test:

```bash
pytest
```

## Layout

| Path | Role |
| --- | --- |
| `model/` | Priors and the `StickyHDPSLDSHMMModel` container |
| `slds/` | One mixin per block of Gibbs updates |
| `misc/` | Categorical sampling, stick breaking, I/O |
| `data/` | Synthetic data generation |
| `data_packs/` | Checked-in MNIW and ARD datasets |
| `examples/smoke_slds.py` | Short end-to-end run |
| `tests/` | Regression tests for the sampler |
| `docs/REFACTOR_ANALYSIS.md` | Structure, defects, and the remaining refactor |

`hdp_params` is `[alpha + kappa, rho, gamma, beta, pi]`. `slds_params` is `[A, Sigma]` for the matrix-normal inverse-Wishart prior and `[A, Sigma, alpha]` for the ARD prior.

## Tests

`pytest` runs the checks in `tests/`. They cover stick-breaking initialization, the sticky table counts, emission degrees of freedom, the Kalman initialization, the two mode samplers, both dynamics posteriors, save/load of the ARD precision, and the smoke run.

`python -m test_runner.exec_tests` is the original experiment driver. It writes plots under `test_runs/` and prints sampler traces.

## License

MIT. Copyright (c) 2018 Markus Semmler.
