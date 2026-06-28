"""Composable, opt-in entry filters applied on top of any strategy's signals.

A strategy proposes entries; these gates suppress the ones taken in unfavourable
conditions (wrong-side of trend, dead/explosive volatility, chop, off-hours), so a
raw edge trades only where it actually has one. All filters are vectorized and use
only current/past bars — no lookahead. Disabled fields (None) are no-ops, so an
empty `FilterConfig` leaves signals untouched.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from banknifty_bot.features.indicators import adx as adx_indicator
from banknifty_bot.features.indicators import atr as atr_indicator
from banknifty_bot.features.indicators import sma


@dataclass
class FilterConfig:
    trend_sma: int | None = None            # long only above SMA, short only below
    atr_pct_min: float | None = None        # skip dead vol (ATR as % of price)
    atr_pct_max: float | None = None        # skip explosive vol
    atr_window: int = 14
    adx_min: float | None = None            # require trend strength
    adx_max: float | None = None            # require ranging market
    adx_window: int = 14
    time_start: str | None = None           # intraday: earliest entry time "HH:MM"
    time_end: str | None = None             # intraday: latest entry time "HH:MM"

    @property
    def enabled(self) -> bool:
        return any(v is not None for v in (
            self.trend_sma, self.atr_pct_min, self.atr_pct_max,
            self.adx_min, self.adx_max, self.time_start, self.time_end,
        ))


def apply_filters(signals: pd.DataFrame, df: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    """Return a copy of `signals` with entries that fail any active filter set to 0."""
    if not cfg.enabled:
        return signals

    entry = signals["entry"].copy()
    allow = pd.Series(True, index=df.index)       # direction-agnostic gates
    long_ok = pd.Series(True, index=df.index)
    short_ok = pd.Series(True, index=df.index)

    if cfg.atr_pct_min is not None or cfg.atr_pct_max is not None:
        atr_pct = atr_indicator(df, window=cfg.atr_window) / df["close"] * 100
        if cfg.atr_pct_min is not None:
            allow &= atr_pct >= cfg.atr_pct_min
        if cfg.atr_pct_max is not None:
            allow &= atr_pct <= cfg.atr_pct_max

    if cfg.adx_min is not None or cfg.adx_max is not None:
        adx_val = adx_indicator(df, window=cfg.adx_window)["adx"]
        if cfg.adx_min is not None:
            allow &= adx_val >= cfg.adx_min
        if cfg.adx_max is not None:
            allow &= adx_val <= cfg.adx_max

    if cfg.time_start is not None or cfg.time_end is not None:
        times = pd.Series(df.index.time, index=df.index)
        if cfg.time_start is not None:
            allow &= times >= dt.time.fromisoformat(cfg.time_start)
        if cfg.time_end is not None:
            allow &= times <= dt.time.fromisoformat(cfg.time_end)

    if cfg.trend_sma is not None:
        trend = sma(df["close"], cfg.trend_sma)
        long_ok &= df["close"] > trend
        short_ok &= df["close"] < trend

    new_entry = pd.Series(0, index=df.index)
    new_entry[(entry == 1) & allow & long_ok] = 1
    new_entry[(entry == -1) & allow & short_ok] = -1

    out = signals.copy()
    out["entry"] = new_entry
    return out
