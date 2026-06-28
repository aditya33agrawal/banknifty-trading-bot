"""Exit-management configuration for the backtest engine.

All features are opt-in (None/0 = disabled), so the default `ExitConfig()` leaves
the engine's original stop/target behaviour unchanged. Combine them per strategy:

- trailing_atr_mult : trail the stop by N*ATR behind the best price seen.
- breakeven_at_r    : once price reaches this R-multiple of profit, move stop to entry.
- partial_exit_r    : scale out `partial_exit_pct` of the position at this R-multiple,
                      and move the stop on the remainder to breakeven.
- time_stop_bars    : exit at market if neither stop nor target hits within N bars.

"R" = initial risk per unit = |entry_price - initial_stop|.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExitConfig:
    trailing_atr_mult: float | None = None
    breakeven_at_r: float | None = None
    partial_exit_r: float | None = None
    partial_exit_pct: float = 0.5
    time_stop_bars: int | None = None

    @property
    def enabled(self) -> bool:
        return any(
            v is not None
            for v in (self.trailing_atr_mult, self.breakeven_at_r, self.partial_exit_r, self.time_stop_bars)
        )
