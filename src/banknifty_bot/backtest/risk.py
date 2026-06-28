from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    initial_equity: float
    risk_per_trade_pct: float
    daily_max_loss_pct: float
    max_trades_per_day: int
    max_open_positions: int
    lot_size: int
    contract_multiplier: float = 1.0
    square_off: dt.time = dt.time(15, 15)
    no_trade_first_minutes: int = 5
    no_trade_last_minutes: int = 10
    # Intraday: hard square-off + no-trade windows + flat overnight.
    # Swing (False): positions held across days, exit only on stop/target/signal.
    intraday: bool = True


@dataclass
class DailyState:
    date: dt.date
    trades_count: int = 0
    pnl: float = 0.0
    halted: bool = False


class RiskManager:
    """Enforces per-trade sizing and daily limits. Stateful across a backtest
    run; call `new_day` at each session boundary.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.equity = cfg.initial_equity
        self.day_state: DailyState | None = None
        self.open_positions = 0

    def new_day(self, date: dt.date) -> None:
        self.day_state = DailyState(date=date)

    def in_no_trade_window(self, ts: dt.datetime, session_start: dt.datetime, session_end: dt.datetime) -> bool:
        if not self.cfg.intraday:
            return False
        first_cutoff = session_start + dt.timedelta(minutes=self.cfg.no_trade_first_minutes)
        last_cutoff = session_end - dt.timedelta(minutes=self.cfg.no_trade_last_minutes)
        return ts < first_cutoff or ts > last_cutoff

    def past_square_off(self, ts: dt.datetime) -> bool:
        return self.cfg.intraday and ts.time() >= self.cfg.square_off

    def can_open_position(self) -> bool:
        if self.day_state is None or self.day_state.halted:
            return False
        if self.open_positions >= self.cfg.max_open_positions:
            return False
        if self.day_state.trades_count >= self.cfg.max_trades_per_day:
            return False
        return True

    def position_size(self, entry_price: float, stop_price: float) -> int:
        """Risk-per-trade sizing: lots derived from stop distance and equity risk budget."""
        risk_amount = self.equity * self.cfg.risk_per_trade_pct / 100
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0
        unit_risk = stop_distance * self.cfg.contract_multiplier
        lots = int(risk_amount / (unit_risk * self.cfg.lot_size)) if unit_risk > 0 else 0
        return max(lots, 0) * self.cfg.lot_size

    def register_entry(self) -> None:
        self.open_positions += 1
        self.day_state.trades_count += 1

    def register_exit(self, trade_pnl: float) -> None:
        self.open_positions -= 1
        self._book_pnl(trade_pnl)

    def register_partial(self, trade_pnl: float) -> None:
        """Scale-out fill: books P&L but keeps the position slot open."""
        self._book_pnl(trade_pnl)

    def _book_pnl(self, trade_pnl: float) -> None:
        self.equity += trade_pnl
        self.day_state.pnl += trade_pnl
        max_loss = self.cfg.initial_equity * self.cfg.daily_max_loss_pct / 100
        if self.day_state.pnl <= -max_loss:
            self.day_state.halted = True
