"""Close snapshots for the rolling-origin volatility study.

The CSV files stay in ``examples/data``. Callers pass a path. Nothing here
downloads a new series.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

DATASET_SPECS = {
    "sp500": {
        "filename": "sp500_daily.csv",
        "prefix": "sp500",
        "name": "S&P 500",
        "window": 20,
        "holdout": 250,
        "annualization_days": 252,
    },
    "btc": {
        "filename": "btc_daily.csv",
        "prefix": "btc",
        "name": "Bitcoin",
        "window": 30,
        "holdout": 365,
        "annualization_days": 365,
    },
}


def datasets(data_dir: Path) -> dict[str, dict[str, object]]:
    """Dataset protocol plus the close-file path under ``data_dir``."""

    return {
        name: {**spec, "path": Path(data_dir) / str(spec["filename"])}
        for name, spec in DATASET_SPECS.items()
    }


def load_closes(path: Path) -> dict[str, object]:
    """Read a committed close snapshot.

    Args:
        path: CSV with ``date,close`` rows and ``#`` comment lines. One
            comment records the download date.

    Returns:
        Dates, closes, the source line, and the download date.
    """

    dates: list[datetime] = []
    closes: list[float] = []
    source = ""
    downloaded = ""
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "downloaded:" in stripped:
                downloaded = stripped.split(":", 1)[1].strip()
            elif "source:" in stripped:
                source = stripped.split(":", 1)[1].strip()
            continue
        if stripped.lower().startswith("date"):
            continue
        day, price = stripped.split(",")
        dates.append(datetime.strptime(day, "%Y-%m-%d"))
        closes.append(float(price))
    if len(closes) < 3:
        raise ValueError(f"{path} does not have enough rows.")
    return {
        "dates": dates,
        "closes": np.asarray(closes, dtype=np.float64),
        "source": source,
        "downloaded": downloaded,
    }
