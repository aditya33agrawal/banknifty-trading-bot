from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import adx, atr, ema

from .base import Strategy


class EMATrend(Strategy):
    """Fast/slow EMA crossover, trades only when ADX confirms a trend.

    Params: fast_span, slow_span, adx_window, adx_threshold, sl_atr_mult, target_r, atr_window
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fast_span = p.get("fast_span", 9)
        slow_span = p.get("slow_span", 21)
        adx_window = p.get("adx_window", 14)
        adx_threshold = p.get("adx_threshold", 20)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        target_r = p.get("target_r", 2.0)
        atr_window = p.get("atr_window", 14)

        fast = ema(df["close"], fast_span)
        slow = ema(df["close"], slow_span)
        adx_df = adx(df, window=adx_window)
        atr_val = atr(df, window=atr_window)

        bullish_cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        bearish_cross = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        trending = adx_df["adx"] > adx_threshold

        entry = pd.Series(0, index=df.index)
        entry[bullish_cross & trending] = 1
        entry[bearish_cross & trending] = -1

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
            "fast_span": [5, 9, 12],
            "slow_span": [21, 26, 34],
            "adx_threshold": [15, 20, 25],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "target_r": [1.5, 2.0, 3.0],
        }
