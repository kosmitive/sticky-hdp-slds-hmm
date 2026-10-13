# Install and usage

This project targets Python 3.11 and later. [uv](https://docs.astral.sh/uv/) installs the locked dependencies from `uv.lock`. The Gibbs samplers are JAX. NumPy is used for the price files, the baseline regressions, the pendulum integrator, and the plots.

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

The smoke script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three blocked Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, and a self-transition weight `rho` in `(0, 1)`.

## Examples

```bash
uv run python examples/pendulum_demo.py
uv run python examples/pendulum_forecast.py
uv run python examples/regime_volatility.py sp500
uv run python examples/regime_volatility.py btc
```

`pendulum_demo.py` and `pendulum_forecast.py` accept a sweep count as their only argument. The default is 300. The forecast fits the training window only. `regime_volatility.py` accepts one or more dataset names and an optional sweep count. The default is 4 chains of 2000 sweeps.

## Layout

| Path | Role |
| --- | --- |
| `slds/` | JAX Gibbs samplers for the regime HAR and the sticky HDP-SLDS |
| `examples/smoke_slds.py` | Short end-to-end run |
| `examples/pendulum_demo.py` | Noisy pendulum fit |
| `examples/pendulum_forecast.py` | 80-step forecast from the blocked sampler |
| `examples/regime_volatility.py` | Rolling-origin sticky regime HAR |
| `examples/market_data.py` | Close files and the GARCH baseline |
| `examples/data/` | S&P 500 and Bitcoin close snapshots |
| `tests/` | Unit checks and stored-reference regressions |
| `docs/` | This site's source |
