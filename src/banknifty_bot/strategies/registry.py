from __future__ import annotations

from .base import Strategy
from .orb import OpeningRangeBreakout
from .regime import RegimeFilter
from .rsi_reversion import RSIMeanReversion
from .supertrend import SupertrendStrategy
from .trend import EMATrend
from .vwap import VWAPStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "orb": OpeningRangeBreakout,
    "vwap": VWAPStrategy,
    "ema_trend": EMATrend,
    "supertrend": SupertrendStrategy,
    "rsi_reversion": RSIMeanReversion,
    "regime_filter": RegimeFilter,
}


def register(name: str, cls: type[Strategy]) -> None:
    REGISTRY[name] = cls


def get_strategy(name: str, params: dict) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](params)
