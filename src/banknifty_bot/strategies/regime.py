from __future__ import annotations

import pandas as pd

from banknifty_bot.features.indicators import adx

from .base import Strategy


class RegimeFilter(Strategy):
    """Meta-strategy: routes to a trend sub-strategy on trending days (ADX
    above threshold) and a mean-reversion sub-strategy on ranging days.
    "Do nothing" is the valid default when neither regime's sub-strategy fires.

    Params:
      adx_window, adx_threshold
      trend_strategy, trend_params   (registry name + params dict)
      reversion_strategy, reversion_params
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from .registry import get_strategy  # local import: avoid circular import with registry

        p = self.params
        adx_window = p.get("adx_window", 14)
        adx_threshold = p.get("adx_threshold", 20)

        trend_strategy = get_strategy(p["trend_strategy"], p.get("trend_params", {}))
        reversion_strategy = get_strategy(p["reversion_strategy"], p.get("reversion_params", {}))

        trend_signals = trend_strategy.generate_signals(df)
        reversion_signals = reversion_strategy.generate_signals(df)

        trending = adx(df, window=adx_window)["adx"] > adx_threshold

        out = reversion_signals.where(~trending, trend_signals)
        out["entry"] = out["entry"].fillna(0).astype(int)
        return out

    @property
    def param_space(self) -> dict:
        return {"adx_threshold": [15, 20, 25], "adx_window": [10, 14, 21]}
