from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import adx, atr, rsi

from .base import Strategy


class RSIMeanReversion(Strategy):
    """Oversold/overbought RSI reversion, gated to a ranging regime only
    (ADX below threshold) — trend days are left to trend strategies.

    Params: rsi_window, oversold, overbought, adx_window, adx_max,
            sl_atr_mult, target_r, atr_window
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi_window = p.get("rsi_window", 14)
        oversold = p.get("oversold", 30)
        overbought = p.get("overbought", 70)
        adx_window = p.get("adx_window", 14)
        adx_max = p.get("adx_max", 20)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        target_r = p.get("target_r", 1.5)
        atr_window = p.get("atr_window", 14)

        rsi_val = rsi(df["close"], window=rsi_window)
        adx_df = adx(df, window=adx_window)
        atr_val = atr(df, window=atr_window)

        ranging = adx_df["adx"] < adx_max
        cross_up_from_oversold = (rsi_val > oversold) & (rsi_val.shift(1) <= oversold)
        cross_down_from_overbought = (rsi_val < overbought) & (rsi_val.shift(1) >= overbought)

        entry = pd.Series(0, index=df.index)
        entry[cross_up_from_oversold & ranging] = 1
        entry[cross_down_from_overbought & ranging] = -1

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
            "rsi_window": [10, 14, 21],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
            "adx_max": [15, 20, 25],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "target_r": [1.0, 1.5, 2.0],
        }
