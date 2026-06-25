from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import atr, session_vwap

from .base import Strategy


class VWAPStrategy(Strategy):
    """Trend-following (price vs VWAP + slope) and/or mean-reversion to VWAP bands.

    Params: mode ("trend" | "reversion"), band_atr_mult, slope_window,
            sl_atr_mult, target_r, atr_window
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        mode = p.get("mode", "trend")
        band_atr_mult = p.get("band_atr_mult", 1.0)
        slope_window = p.get("slope_window", 5)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        target_r = p.get("target_r", 2.0)
        atr_window = p.get("atr_window", 14)

        vwap = session_vwap(df)
        atr_val = atr(df, window=atr_window)
        vwap_slope = vwap.diff(slope_window)

        entry = pd.Series(0, index=df.index)

        if mode == "trend":
            entry[(df["close"] > vwap) & (vwap_slope > 0)] = 1
            entry[(df["close"] < vwap) & (vwap_slope < 0)] = -1
        else:  # reversion
            upper_band = vwap + band_atr_mult * atr_val
            lower_band = vwap - band_atr_mult * atr_val
            entry[df["close"] < lower_band] = 1   # buy the dip back toward VWAP
            entry[df["close"] > upper_band] = -1  # fade the rip back toward VWAP

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
            "mode": ["trend", "reversion"],
            "band_atr_mult": [0.5, 1.0, 1.5],
            "slope_window": [3, 5, 10],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "target_r": [1.5, 2.0, 3.0],
        }
