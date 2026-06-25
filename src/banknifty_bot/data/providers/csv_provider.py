from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataProvider


class CSVProvider(DataProvider):
    """Loads bulk historical OHLCV from local CSV files (plan §4 Tier C).

    Expects a `timestamp` column (parseable, naive-or-aware) plus
    open/high/low/close/volume columns (case-insensitive).
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def get_ohlcv(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        path = self.root_dir / f"{symbol}_{interval}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No CSV data file at {path}")

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        df = df.loc[(df.index >= pd.Timestamp(start, tz=df.index.tz)) &
                    (df.index <= pd.Timestamp(end, tz=df.index.tz))]

        return df[["open", "high", "low", "close", "volume"]]
