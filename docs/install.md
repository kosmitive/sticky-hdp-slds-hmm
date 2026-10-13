# Install and usage

This project targets Python 3.11 and later. The import name is `sticky_hdp_slds`. [uv](https://docs.astral.sh/uv/) installs the locked dependencies from `uv.lock` and builds the package with the uv build backend. The Gibbs samplers are JAX. NumPy is used for the price files, the baseline regressions, and the pendulum integrator.

```bash
uv sync
uv run pytest
uv run python -m examples.smoke_slds
```

`uv sync` installs the package and the `dev` group (pytest, ruff, pre-commit, and the `examples` group). The `examples` group provides `arch` and `matplotlib`, the same libraries as the `volatility` and `plots` extras. The documentation group is separate:

```bash
uv sync --group docs
uv run mkdocs serve
uv build
```

A wheel install from another project uses the same extras:

```bash
uv add sticky-hdp-slds
uv add "sticky-hdp-slds[volatility,plots]"
```

The smoke script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three blocked Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, and a self-transition weight `rho` in `(0, 1)`.

## Examples

```bash
uv run python -m examples.pendulum_demo
uv run python -m examples.pendulum_forecast
uv run python -m examples.regime_volatility sp500
uv run python -m examples.regime_volatility btc
```

`pendulum_demo.py` and `pendulum_forecast.py` accept a sweep count as their only argument. The default is 300. The forecast fits the training window only. `regime_volatility.py` accepts one or more dataset names and an optional sweep count. The default is 4 chains of 2000 sweeps.

## Layout

| Path | Role |
| --- | --- |
| `src/sticky_hdp_slds/` | Installable package: Gibbs samplers and the volatility study |
| `examples/smoke_slds.py` | Short end-to-end run |
| `examples/pendulum_demo.py` | Noisy pendulum fit |
| `examples/pendulum_forecast.py` | 80-step forecast from the blocked sampler |
| `examples/regime_volatility.py` | Plots and the command line for the rolling-origin study |
| `examples/data/` | S&P 500 and Bitcoin close snapshots |
| `tests/` | Unit checks and stored-reference regressions |
| `docs/` | This site's source |
