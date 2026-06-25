from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import atr
from banknifty_bot.features.indicators import supertrend as supertrend_indicator

from .base import Strategy


class SupertrendStrategy(Strategy):
    """Trades flips of the Supertrend direction, filtered by a minimum
    volatility (ATR as % of price) so flat/illiquid chop is skipped.

    Params: period, multiplier, min_atr_pct, sl_atr_mult, target_r, atr_window
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        period = p.get("period", 10)
        multiplier = p.get("multiplier", 3.0)
        min_atr_pct = p.get("min_atr_pct", 0.05)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        target_r = p.get("target_r", 2.0)
        atr_window = p.get("atr_window", 14)

        st = supertrend_indicator(df, period=period, multiplier=multiplier)
        atr_val = atr(df, window=atr_window)
        atr_pct = atr_val / df["close"] * 100

        flip_up = (st["direction"] == 1) & (st["direction"].shift(1) == -1)
        flip_down = (st["direction"] == -1) & (st["direction"].shift(1) == 1)
        sufficient_vol = atr_pct > min_atr_pct

        entry = pd.Series(0, index=df.index)
        entry[flip_up & sufficient_vol] = 1
        entry[flip_down & sufficient_vol] = -1

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
            "period": [7, 10, 14],
            "multiplier": [2.0, 3.0, 4.0],
            "min_atr_pct": [0.02, 0.05, 0.1],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "target_r": [1.5, 2.0, 3.0],
        }
