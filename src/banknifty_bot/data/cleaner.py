from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.utils.calendar import is_trading_day

_OHLC = ["open", "high", "low", "close"]


def clean_ohlcv(
    df: pd.DataFrame,
    session_start: str = "09:15",
    session_end: str = "15:30",
    intraday: bool = True,
) -> pd.DataFrame:
    """Drop non-session bars/holidays, de-duplicate, sanity-check — never fabricate trades.

    For daily+ data (`intraday=False`) the intraday session-time filter is skipped,
    since daily bars are timestamped at midnight and would otherwise all be dropped.
    """
    if df.empty:
        return df

    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")

    df = df[~df.index.duplicated(keep="last")].sort_index()

    trading_mask = df.index.to_series(index=df.index).apply(
        lambda ts: is_trading_day(ts.date())
    )
    df = df.loc[trading_mask.values]

    if intraday:
        start_t = pd.to_datetime(session_start).time()
        end_t = pd.to_datetime(session_end).time()
        df = df.loc[(df.index.time >= start_t) & (df.index.time <= end_t)]

    # Sanity checks: drop rows with non-positive prices or impossible OHLC ordering.
    valid = (df[_OHLC] > 0).all(axis=1)
    valid &= df["high"] >= df[["open", "close", "low"]].max(axis=1)
    valid &= df["low"] <= df[["open", "close", "high"]].min(axis=1)
    df = df.loc[valid]

    # Flag (do not silently fill) extreme single-bar jumps for manual review.
    ret = df["close"].pct_change().abs()
    df["_suspect_jump"] = ret > 0.10

    return df


def coverage_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "start": None, "end": None, "trading_days": 0}
    return {
        "rows": len(df),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "trading_days": df.index.normalize().nunique(),
    }
