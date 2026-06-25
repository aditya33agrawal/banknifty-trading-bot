from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import atr, opening_range

from .base import Strategy


class OpeningRangeBreakout(Strategy):
    """Break of the first N-minute range, ATR-based stop, R-multiple target.

    Params: range_minutes, buffer_pct, sl_atr_mult, target_r, atr_window, allow_reentry
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        range_minutes = p.get("range_minutes", 15)
        buffer_pct = p.get("buffer_pct", 0.05)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        target_r = p.get("target_r", 2.0)
        atr_window = p.get("atr_window", 14)
        allow_reentry = p.get("allow_reentry", False)

        orng = opening_range(df, minutes=range_minutes)
        atr_val = atr(df, window=atr_window)

        day = df.index.normalize()
        session_start = df.groupby(day).apply(lambda g: g.index.min())
        range_end = session_start.reindex(day).values + pd.Timedelta(minutes=range_minutes)
        past_range = df.index.values >= range_end

        high_break = df["close"] > orng["or_high"] * (1 + buffer_pct / 100)
        low_break = df["close"] < orng["or_low"] * (1 - buffer_pct / 100)

        entry = pd.Series(0, index=df.index)
        entry[past_range & high_break] = 1
        entry[past_range & low_break & ~(past_range & high_break)] = -1

        if not allow_reentry:
            day_series = pd.Series(day, index=df.index)
            already_traded = entry.ne(0).groupby(day_series).cumsum().groupby(day_series).shift(1).fillna(0)
            entry = entry.where(already_traded == 0, 0)

        stop_loss = pd.Series(np.nan, index=df.index)
        target = pd.Series(np.nan, index=df.index)

        long_mask = entry == 1
        short_mask = entry == -1
        stop_loss[long_mask] = df["close"][long_mask] - sl_atr_mult * atr_val[long_mask]
        stop_loss[short_mask] = df["close"][short_mask] + sl_atr_mult * atr_val[short_mask]

        risk_long = df["close"][long_mask] - stop_loss[long_mask]
        risk_short = stop_loss[short_mask] - df["close"][short_mask]
        target[long_mask] = df["close"][long_mask] + target_r * risk_long
        target[short_mask] = df["close"][short_mask] - target_r * risk_short

        return pd.DataFrame({"entry": entry, "stop_loss": stop_loss, "target": target})

    @property
    def param_space(self) -> dict:
        return {
            "range_minutes": [15, 30],
            "buffer_pct": [0.0, 0.05, 0.1],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "target_r": [1.5, 2.0, 3.0],
            "atr_window": [14],
        }
