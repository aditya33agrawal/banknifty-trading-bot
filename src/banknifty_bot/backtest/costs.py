"""Indian F&O cost model. The index itself isn't tradeable, so every backtest
models the chosen execution proxy (futures or options) — see plan §6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from banknifty_bot.config import load_yaml

Side = Literal["long", "short"]


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp_duty: float
    gst: float

    @property
    def total(self) -> float:
        return (
            self.brokerage + self.stt + self.exchange_txn
            + self.sebi + self.stamp_duty + self.gst
        )


class CostModel:
    def __init__(self, instrument: Literal["futures", "options"], params: dict):
        self.instrument = instrument
        self.params = params

    @classmethod
    def from_yaml(cls, path: str, instrument: Literal["futures", "options"]) -> "CostModel":
        all_params = load_yaml(path)
        return cls(instrument, all_params[instrument])

    def _leg_cost(self, value: float, is_buy: bool) -> CostBreakdown:
        p = self.params
        brokerage = p["brokerage_per_order"]

        if self.instrument == "options":
            stt = (value * p["stt_pct_on_sell_premium"] / 100) if not is_buy else 0.0
            exch = value * p["exchange_txn_pct_on_premium"] / 100
            sebi = value * p["sebi_pct_on_premium"] / 100
            stamp = (value * p["stamp_duty_pct_on_buy_premium"] / 100) if is_buy else 0.0
        else:
            stt = (value * p["stt_pct_on_sell_value"] / 100) if not is_buy else 0.0
            exch = value * p["exchange_txn_pct_on_value"] / 100
            sebi = value * p["sebi_pct_on_value"] / 100
            stamp = (value * p["stamp_duty_pct_on_buy_value"] / 100) if is_buy else 0.0

        gst = (brokerage + exch + sebi) * p["gst_pct"] / 100
        return CostBreakdown(brokerage, stt, exch, sebi, stamp, gst)

    def round_trip_cost(
        self, entry_price: float, exit_price: float, qty: int, side: Side
    ) -> CostBreakdown:
        """Total cost breakdown for opening + closing a position.

        `qty` is total units (lots * lot_size). `side` determines which leg
        is the buy and which is the sell.
        """
        entry_value = entry_price * qty
        exit_value = exit_price * qty

        if side == "long":
            entry_leg = self._leg_cost(entry_value, is_buy=True)
            exit_leg = self._leg_cost(exit_value, is_buy=False)
        else:
            entry_leg = self._leg_cost(entry_value, is_buy=False)
            exit_leg = self._leg_cost(exit_value, is_buy=True)

        return CostBreakdown(
            brokerage=entry_leg.brokerage + exit_leg.brokerage,
            stt=entry_leg.stt + exit_leg.stt,
            exchange_txn=entry_leg.exchange_txn + exit_leg.exchange_txn,
            sebi=entry_leg.sebi + exit_leg.sebi,
            stamp_duty=entry_leg.stamp_duty + exit_leg.stamp_duty,
            gst=entry_leg.gst + exit_leg.gst,
        )
