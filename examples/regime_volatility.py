"""Plot and run the rolling-origin sticky regime HAR.

uv run python -m examples.regime_volatility sp500
uv run python -m examples.regime_volatility btc
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from sticky_hdp_slds import evaluate_returns, load_closes
from sticky_hdp_slds.volatility import CHAINS, REFIT_YEARS, SWEEPS, datasets

_DATA_DIR = Path(__file__).resolve().parent / "data"
_IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
DATASETS = datasets(_DATA_DIR)


def _print_report(name: str, result: dict[str, object]) -> None:
    diagnostics = result["diagnostics"]
    print(f"{name} rolling-origin log mean r^2", flush=True)
    print(
        "  "
        + " ".join(
            f"{key} {diagnostics[key]}"
            for key in (
                "holdout",
                "train_days",
                "sweeps",
                "chains",
                "refit_every",
                "seed",
                "regimes",
                "self_stay_prior",
                "zero_returns",
                "garch_failures",
            )
        ),
        flush=True,
    )
    print(
        f"  rhat_max {diagnostics['rhat_max']:.4f} elapsed_sec {diagnostics['elapsed_sec']:.1f}",
        flush=True,
    )
    for item in diagnostics["refits"]:
        intercepts = " ".join(f"{value:.3f}" for value in item["intercepts"])
        slopes = " ".join(f"{value:.3f}" for value in item["slopes"])
        stay = " ".join(f"{value:.3f}" for value in item["self_stay_by_regime"])
        share = " ".join(f"{value:.3f}" for value in item["occupancy"])
        print(
            f"  refit {item['origin']} rhat {item['rhat_max']:.3f} stay {stay} "
            f"dur {' '.join(f'{value:.1f}' for value in item['duration_days'])} "
            f"c {intercepts} b {slopes} radius {item['spectral_radius']:.3f} "
            f"share {share} ll {item['loglik_second_mean']:.1f} "
            f"ll_std {item['loglik_second_std']:.1f}",
            flush=True,
        )
    for horizon, scores in result["scores"].items():
        parts = [f"h {horizon} n {scores['n']}"]
        for model in ("regime", "har", "ewma", "garch", "mean"):
            if model not in scores:
                continue
            parts.append(f"{model} {scores[model]['rmse']:.6f} bias {scores[model]['bias']:.6f}")
        print("  " + " ".join(parts), flush=True)
        for model, test in scores["dm"].items():
            print(
                f"    dm vs {model} stat {test['stat']:.4f} p {test['pvalue']:.4g} "
                f"fallback {test['nw_fallback']:.0f}",
                flush=True,
            )


def save_regime_plots(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    """Write the forecast, horizon, probability, and convergence figures."""

    destination.mkdir(parents=True, exist_ok=True)
    _plot_forecast(name, asset_name, result, destination)
    _plot_horizon(name, asset_name, result, destination)
    _plot_probabilities(name, asset_name, result, destination)
    _plot_convergence(name, asset_name, result, destination)


def _plot_forecast(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    horizon = 20 if 20 in result["actual"] else result["horizons"][-1]
    actual = result["actual"][horizon]
    dates = np.asarray(result["dates"])[: actual.size]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(dates, actual, color="black", lw=1.3, label="realized")
    series = [
        ("regime", "#1f4e79", 1.4),
        ("har", "#b85c38", 1.1),
        ("ewma", "#2a9d8f", 1.0),
        ("garch", "#6d597a", 1.0),
        ("mean", "#888888", 1.0),
    ]
    for key, color, width in series:
        if key == "regime":
            values = result["regime"][horizon]
        else:
            values = result["baseline"][horizon][key]
        ax.plot(dates, values, color=color, lw=width, label=key)
    ax.set_ylabel(f"log mean r^2 over {horizon} days")
    ax.set_xlabel("forecast origin")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{asset_name}: rolling-origin {horizon}-day log variance")
    if np.issubdtype(np.asarray(dates).dtype, np.datetime64) or _is_datetime(dates):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_forecast.png", dpi=140)
    plt.close(fig)


def _is_datetime(dates: np.ndarray) -> bool:
    if dates.size == 0:
        return False
    return hasattr(dates[0], "strftime")


def _plot_horizon(name: str, asset_name: str, result: dict[str, object], destination: Path) -> None:
    horizons = list(result["horizons"])
    models = ["regime", "har", "ewma", "garch", "mean"]
    colors = ["#1f4e79", "#b85c38", "#2a9d8f", "#6d597a", "#888888"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.15
    positions = np.arange(len(horizons), dtype=np.float64)
    for offset, (model, color) in enumerate(zip(models, colors, strict=True)):
        heights = []
        for horizon in horizons:
            if model not in result["scores"][horizon]:
                heights.append(np.nan)
            else:
                heights.append(result["scores"][horizon][model]["rmse"])
        ax.bar(positions + (offset - 2) * width, heights, width=width, color=color, label=model)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(horizon) for horizon in horizons])
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("RMSE of log mean r^2")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{asset_name}: rolling-origin RMSE")
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_horizon.png", dpi=140)
    plt.close(fig)


def _plot_probabilities(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    dates = np.asarray(result["dates"])
    probs = result["probabilities"]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    for regime in range(probs.shape[1]):
        axes[0].plot(dates, probs[:, regime], lw=1.1, label=f"regime {regime}")
    axes[0].set_ylabel("filtered probability")
    axes[0].legend(ncol=3, frameon=False)
    horizon = 1 if 1 in result["actual"] else result["horizons"][0]
    axes[1].plot(
        dates[: result["actual"][horizon].size], result["actual"][horizon], color="black", lw=1.0
    )
    axes[1].plot(
        dates[: result["regime"][horizon].size], result["regime"][horizon], color="#1f4e79", lw=1.0
    )
    axes[1].set_ylabel("log next-day r^2")
    axes[1].set_xlabel("forecast origin")
    axes[0].set_title(f"{asset_name}: filtered regimes, ordered from low to high intercept")
    if _is_datetime(dates):
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_probs.png", dpi=140)
    plt.close(fig)


def _plot_convergence(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    loglik = result["diagnostics"]["loglik_first"]
    if loglik is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    sweeps = np.arange(1, loglik.shape[1] + 1)
    half = loglik.shape[1] // 2
    for chain in range(loglik.shape[0]):
        ax.plot(sweeps, loglik[chain], lw=1.0, label=f"chain {chain + 1}")
    ax.axvline(half + 1, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("Gibbs sweep")
    ax.set_ylabel("complete-data log likelihood")
    ax.legend(frameon=False)
    rhat = float(result["diagnostics"]["refits"][0]["rhat_max"])
    ax.set_title(f"{asset_name}: first refit, max R-hat {rhat:.3f}")
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_convergence.png", dpi=140)
    plt.close(fig)


def run_dataset(
    name: str,
    sweeps: int | None = None,
    chains: int | None = None,
) -> dict[str, object]:
    """Fit one named dataset and write its figures and scores."""

    spec = DATASETS[name]
    snapshot = load_closes(spec["path"])
    closes = snapshot["closes"]
    returns = np.log(closes[1:] / closes[:-1])
    result = evaluate_returns(
        returns,
        holdout=int(spec["holdout"]),
        train_days=REFIT_YEARS * int(spec["annualization_days"]),
        sweeps=SWEEPS if sweeps is None else sweeps,
        chains=CHAINS if chains is None else chains,
        dates=np.asarray(snapshot["dates"]),
    )
    _print_report(name, result)
    save_regime_plots(spec["prefix"], spec["name"], result, _IMG_DIR)
    print(f"  figures {_IMG_DIR}", flush=True)
    return result


def main() -> None:
    """Run the dataset named in the first argument."""

    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        names = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Usage: python -m examples.regime_volatility <{names}> [sweeps]")
    sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_dataset(sys.argv[1], sweeps=sweeps)


if __name__ == "__main__":
    main()
