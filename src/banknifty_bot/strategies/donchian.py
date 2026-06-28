from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import atr, donchian_channel, sma

from .base import Strategy


class DonchianBreakout(Strategy):
    """Turtle-style channel breakout swing trend-follower (designed for daily bars).

    Goes long on a close above the prior `channel` high, short below the prior low,
    optionally only in the direction of a long SMA trend filter. ATR stop, R target.

    Params: channel, trend_sma, use_trend_filter, sl_atr_mult, target_r, atr_window
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        channel = p.get("channel", 20)
        trend_sma = p.get("trend_sma", 100)
        use_trend_filter = p.get("use_trend_filter", True)
        sl_atr_mult = p.get("sl_atr_mult", 2.0)
        target_r = p.get("target_r", 3.0)
        atr_window = p.get("atr_window", 14)

        dc = donchian_channel(df, window=channel)
        atr_val = atr(df, window=atr_window)
        trend = sma(df["close"], trend_sma)

        long_ok = df["close"] > dc["dc_upper"]
        short_ok = df["close"] < dc["dc_lower"]
        if use_trend_filter:
            long_ok &= df["close"] > trend
            short_ok &= df["close"] < trend

        entry = pd.Series(0, index=df.index)
        entry[long_ok] = 1
        entry[short_ok & ~long_ok] = -1

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
            "channel": [10, 20, 55],
            "trend_sma": [50, 100, 200],
            "use_trend_filter": [True, False],
            "sl_atr_mult": [1.5, 2.0, 3.0],
            "target_r": [2.0, 3.0, 4.0],
        }
