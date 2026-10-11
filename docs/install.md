# Install and usage

This project targets Python 3.9 and later. [uv](https://docs.astral.sh/uv/) installs the locked dependencies from `uv.lock`.

```bash
uv sync
uv run pytest
uv run python examples/smoke_slds.py
```

`uv sync` installs the package and the `dev` group (pytest, ruff, pre-commit, and the `examples` group). The volatility scripts use `arch` from that examples group. The documentation group is separate:

```bash
uv sync --group docs
uv run mkdocs serve
```

The smoke script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, a self-transition weight `rho` in `(0, 1)`, and a large diagonal entry for a mode that never switches.

## Examples

```bash
uv run python examples/pendulum_demo.py
uv run python examples/pendulum_forecast.py
uv run python examples/volatility_forecast.py sp500
uv run python examples/volatility_forecast.py btc
```

`pendulum_demo.py` accepts a sweep count as its only argument. The default is 300. The forecast script uses that same default and does not read the holdout while fitting.

## Layout

| Path | Role |
| --- | --- |
| `model/` | Priors, typed parameter state, and `StickyHDPSLDSHMMModel` |
| `slds/` | One mixin per block of Gibbs updates |
| `misc/` | Categorical sampling, stick breaking, I/O |
| `data/` | Synthetic data generation |
| `data_packs/` | Checked-in MNIW and ARD datasets, seed `0` |
| `examples/smoke_slds.py` | Short end-to-end run |
| `examples/pendulum_demo.py` | Noisy pendulum fit |
| `examples/pendulum_forecast.py` | 80-step forecast from that fit |
| `examples/volatility_forecast.py` | S&P 500 or Bitcoin volatility forecast |
| `examples/sp500_forecast.py` | S&P 500 entry point for that script |
| `examples/btc_forecast.py` | Bitcoin entry point, 30-day window |
| `examples/data/` | Close snapshots used by those two scripts |
| `tests/` | Regression tests for the sampler |

Checked-in packs still use the historical filenames: `P0.npy` is the emission scale and `R0.npy` is the latent covariance. Load also accepts the honest assignment when the two arrays have different sizes.

## Experiments

`uv run python -m test_runner.exec_tests` is the original experiment driver. It writes plots under `test_runs/` and prints sampler traces. Sampling flags for those cases are a table in `test_runner/all/slds_basic_test.py`.
