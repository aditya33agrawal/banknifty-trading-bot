from __future__ import annotations

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import atr

from .base import Strategy

# Default members: the four validated swing edges seeded with their walk-forward-best
# params (from scripts/wf_search_all.py). Each contributes a directional vote.
DEFAULT_MEMBERS = [
    {"name": "ema_trend", "params": {"fast_span": 12, "slow_span": 34, "adx_threshold": 20, "sl_atr_mult": 1.0, "target_r": 2.0}},
    {"name": "supertrend", "params": {"period": 10, "multiplier": 4.0, "min_atr_pct": 0.1, "sl_atr_mult": 1.0, "target_r": 2.0}},
    {"name": "rsi_reversion", "params": {"rsi_window": 10, "oversold": 35, "overbought": 70, "adx_max": 20, "sl_atr_mult": 1.5, "target_r": 2.0}},
    {"name": "donchian", "params": {"channel": 10, "trend_sma": 200, "use_trend_filter": True, "sl_atr_mult": 3.0, "target_r": 4.0}},
]


class Ensemble(Strategy):
    """Voting meta-strategy over several uncorrelated sub-strategies.

    Each member's entry signals are forward-filled into a persistent directional
    *stance* (long/flat/short), since the members rarely fire on the same bar.
    The ensemble goes long/short on the rising edge of agreement — i.e. when at
    least `min_votes` members share a stance and that side outvotes the other.
    Stops/targets are a unified ATR rule, independent of the members' own exits.

    Params: members (list of {name, params}), min_votes, sl_atr_mult, target_r, atr_window
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from .registry import get_strategy  # local import: avoid circular import

        p = self.params
        members = p.get("members", DEFAULT_MEMBERS)
        min_votes = p.get("min_votes", 2)
        sl_atr_mult = p.get("sl_atr_mult", 1.5)
        target_r = p.get("target_r", 2.0)
        atr_window = p.get("atr_window", 14)

        stances = []
        for m in members:
            sig = get_strategy(m["name"], m.get("params", {})).generate_signals(df)
            stance = sig["entry"].where(sig["entry"] != 0).ffill().fillna(0)
            stances.append(stance.rename(m["name"]))
        stance_df = pd.concat(stances, axis=1)

        vote_long = (stance_df == 1).sum(axis=1)
        vote_short = (stance_df == -1).sum(axis=1)
        long_agree = (vote_long >= min_votes) & (vote_long > vote_short)
        short_agree = (vote_short >= min_votes) & (vote_short > vote_long)

        entry = pd.Series(0, index=df.index)
        entry[long_agree & ~long_agree.shift(1, fill_value=False)] = 1
        entry[short_agree & ~short_agree.shift(1, fill_value=False)] = -1

        atr_val = atr(df, window=atr_window)
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
            "min_votes": [2, 3],
            "sl_atr_mult": [1.0, 1.5, 2.0],
            "target_r": [1.5, 2.0, 3.0],
        }
