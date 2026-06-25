from __future__ import annotations

import datetime as dt

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.engine import OpenPosition
from banknifty_bot.backtest.portfolio import Portfolio, Trade
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.utils.logging import get_logger

from .broker_base import Broker, Side

log = get_logger(__name__)


class PaperBroker(Broker):
    """Simulated fills using the latest available bar + the *same* CostModel
    and Slippage objects used in backtesting, so paper P&L is directly
    comparable to backtest P&L (plan §10 — no logic drift).
    """

    def __init__(self, cost_model: CostModel, slippage: Slippage, contract_multiplier: float, initial_equity: float):
        self.cost_model = cost_model
        self.slippage = slippage
        self.contract_multiplier = contract_multiplier
        self.portfolio = Portfolio(initial_equity=initial_equity)
        self._position: OpenPosition | None = None

    def has_open_position(self) -> bool:
        return self._position is not None

    def open_position(self, side: Side, qty: int, price: float, ts: dt.datetime, stop_loss: float = float("nan"), target: float = float("nan")) -> bool:
        if self._position is not None or qty <= 0:
            return False
        fill_side = "buy" if side == "long" else "sell"
        fill_price = self.slippage.adjust(price, fill_side)
        self._position = OpenPosition(side=side, entry_time=ts, entry_price=fill_price, qty=qty, stop_loss=stop_loss, target=target)
        log.info("PAPER OPEN %s qty=%d @ %.2f (ts=%s)", side, qty, fill_price, ts)
        return True

    def close_position(self, price: float, ts: dt.datetime, reason: str) -> None:
        if self._position is None:
            return
        pos = self._position
        fill_side = "sell" if pos.side == "long" else "buy"
        exit_price = self.slippage.adjust(price, fill_side)

        if pos.side == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.qty * self.contract_multiplier
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.qty * self.contract_multiplier

        cost = self.cost_model.round_trip_cost(pos.entry_price, exit_price, pos.qty, pos.side).total
        net_pnl = gross_pnl - cost

        self.portfolio.record_trade(
            Trade(
                entry_time=pos.entry_time, exit_time=ts, side=pos.side,
                entry_price=pos.entry_price, exit_price=exit_price, qty=pos.qty,
                gross_pnl=gross_pnl, cost=cost, net_pnl=net_pnl, exit_reason=reason,
            )
        )
        log.info("PAPER CLOSE %s qty=%d @ %.2f net_pnl=%.2f reason=%s", pos.side, pos.qty, exit_price, net_pnl, reason)
        self._position = None

    def check_stop_target(self, bar) -> str | None:
        if self._position is None:
            return None
        pos = self._position
        if pos.side == "long":
            if bar["low"] <= pos.stop_loss:
                return "stop"
            if bar["high"] >= pos.target:
                return "target"
        else:
            if bar["high"] >= pos.stop_loss:
                return "stop"
            if bar["low"] <= pos.target:
                return "target"
        return None
