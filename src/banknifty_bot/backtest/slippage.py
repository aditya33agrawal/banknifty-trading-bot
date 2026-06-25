from __future__ import annotations

from typing import Literal

SlippageModel = Literal["ticks", "pct", "atr_fraction"]


class Slippage:
    """Adverse price adjustment applied at fill time.

    - ticks: fixed price offset (e.g. 1.0 index point), scaled by `multiplier`.
    - pct: percentage of the fill price.
    - atr_fraction: fraction of the bar's ATR at the time of the fill.
    """

    def __init__(self, model: SlippageModel, value: float, multiplier: float = 1.0):
        self.model = model
        self.value = value
        self.multiplier = multiplier

    def adjust(self, price: float, side: Literal["buy", "sell"], atr_value: float | None = None) -> float:
        offset = self._offset(price, atr_value) * self.multiplier
        return price + offset if side == "buy" else price - offset

    def _offset(self, price: float, atr_value: float | None) -> float:
        if self.model == "ticks":
            return self.value
        if self.model == "pct":
            return price * self.value / 100
        if self.model == "atr_fraction":
            return (atr_value or 0.0) * self.value
        raise ValueError(f"Unknown slippage model: {self.model}")

    def stressed(self, multiplier: float) -> "Slippage":
        return Slippage(self.model, self.value, multiplier)
