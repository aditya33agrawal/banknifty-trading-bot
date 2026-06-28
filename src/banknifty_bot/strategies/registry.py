from __future__ import annotations

from .base import Strategy
from .donchian import DonchianBreakout
from .ensemble import Ensemble
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
    "donchian": DonchianBreakout,
    "ensemble": Ensemble,
}

# Strategies that work on daily/swing bars (the rest are intraday-only, relying on
# session VWAP / opening-range mechanics).
SWING_COMPATIBLE = {"ema_trend", "supertrend", "rsi_reversion", "donchian", "ensemble"}


def register(name: str, cls: type[Strategy]) -> None:
    REGISTRY[name] = cls


def get_strategy(name: str, params: dict) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](params)
