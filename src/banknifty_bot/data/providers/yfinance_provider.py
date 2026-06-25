from __future__ import annotations

import pandas as pd

from .base import DataProvider

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceProvider(DataProvider):
    """Free data via yfinance. Intraday history is rolling-window limited by
    Yahoo (1m ~ 7d, 5m/15m ~ 60d) — see plan §4 Tier A.
    """

    def get_ohlcv(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=_COLUMN_MAP)[["open", "high", "low", "close", "volume"]]
        df.index.name = "timestamp"

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Kolkata")

        return df
