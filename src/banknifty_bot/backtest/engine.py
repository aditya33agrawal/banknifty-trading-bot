from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from banknifty_bot.features.indicators import atr as atr_indicator

from .costs import CostModel
from .exits import ExitConfig
from .portfolio import Portfolio, Trade
from .risk import RiskConfig, RiskManager
from .slippage import Slippage


@dataclass
class OpenPosition:
    side: str
    entry_time: dt.datetime
    entry_price: float
    qty: int
    stop_loss: float
    target: float
    initial_risk: float = 0.0
    best_price: float = 0.0
    bars_held: int = 0
    partial_done: bool = False
    breakeven_done: bool = False
    stop_moved: bool = False


class BacktestEngine:
    """Bar-by-bar simulation: signals -> risk/sizing -> fills w/ slippage ->
    costs -> portfolio/ledger -> daily limits & square-off.

    Intrabar mechanics: on each bar after entry, checks whether the bar's
    high/low would have hit the stop or target first (stop assumed to trigger
    first if both are touched in the same bar — conservative).

    Optional exit management (`ExitConfig`): ATR trailing stop, breakeven shift,
    R-multiple partial scale-out, and time stop.
    """

    def __init__(
        self,
        risk_cfg: RiskConfig,
        cost_model: CostModel,
        slippage: Slippage,
        atr_window: int = 14,
        exit_cfg: ExitConfig | None = None,
    ):
        self.risk_cfg = risk_cfg
        self.cost_model = cost_model
        self.slippage = slippage
        self.atr_window = atr_window
        self.exit_cfg = exit_cfg or ExitConfig()

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> Portfolio:
        portfolio = Portfolio(initial_equity=self.risk_cfg.initial_equity)
        risk = RiskManager(self.risk_cfg)
        atr_series = atr_indicator(df, window=self.atr_window)

        # Pre-extract numpy arrays — avoids per-bar pandas overhead (~10x faster than iterrows)
        closes      = df["close"].to_numpy(dtype=float)
        highs       = df["high"].to_numpy(dtype=float)
        lows        = df["low"].to_numpy(dtype=float)
        index       = df.index
        sig_entries = signals["entry"].to_numpy(dtype=float)
        sig_stops   = signals["stop_loss"].to_numpy(dtype=float)
        sig_targets = signals["target"].to_numpy(dtype=float)
        atr_vals    = atr_series.to_numpy(dtype=float)

        position: OpenPosition | None = None
        current_day: dt.date | None = None
        session_start_ts: dt.datetime | None = None
        session_end_ts: dt.datetime | None = None

        for i in range(len(df)):
            ts    = index[i]
            day   = ts.date()
            close = closes[i]
            high  = highs[i]
            low   = lows[i]
            atr   = atr_vals[i]

            if day != current_day:
                if self.risk_cfg.intraday and position is not None:
                    self._close(portfolio, risk, position, close, ts, "square_off")
                    position = None
                current_day = day
                risk.new_day(day)
                session_start_ts = ts
                session_end_ts = pd.Timestamp.combine(day, self.risk_cfg.square_off).tz_localize(ts.tzinfo)

            if position is not None:
                position, closed = self._check_exit(
                    portfolio, risk, position, high, low, close, ts, atr
                )
                if closed:
                    position = None

            if position is None and risk.past_square_off(ts):
                portfolio.mark(ts)
                continue

            if position is None and risk.can_open_position():
                entry_sig = sig_entries[i]
                if entry_sig != 0 and not risk.in_no_trade_window(ts, session_start_ts, session_end_ts):
                    position = self._open(
                        risk, entry_sig, sig_stops[i], sig_targets[i], close, ts, atr
                    )

            portfolio.mark(ts)

        if position is not None:
            last_i = len(df) - 1
            self._close(portfolio, risk, position, closes[last_i], index[last_i], "square_off")

        return portfolio

    def _open(
        self,
        risk: RiskManager,
        entry_sig: float,
        stop_loss: float,
        target: float,
        close: float,
        ts: dt.datetime,
        atr_val: float,
    ) -> OpenPosition | None:
        side = "long" if entry_sig == 1 else "short"
        fill_side = "buy" if side == "long" else "sell"
        entry_price = self.slippage.adjust(close, fill_side, atr_val)

        qty = risk.position_size(entry_price, stop_loss)
        if qty <= 0:
            return None

        risk.register_entry()
        return OpenPosition(
            side=side,
            entry_time=ts,
            entry_price=entry_price,
            qty=qty,
            stop_loss=stop_loss,
            target=target,
            initial_risk=abs(entry_price - stop_loss),
            best_price=entry_price,
        )

    def _check_exit(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        position: OpenPosition,
        high: float,
        low: float,
        close: float,
        ts: dt.datetime,
        atr_val: float,
    ) -> tuple[OpenPosition | None, bool]:
        if risk.past_square_off(ts):
            self._close(portfolio, risk, position, close, ts, "square_off")
            return None, True

        long = position.side == "long"

        # 1. Hard stop / target using the levels carried into this bar.
        if (long and low <= position.stop_loss) or (not long and high >= position.stop_loss):
            reason = "trail_stop" if position.stop_moved else "stop"
            self._close(portfolio, risk, position, close, ts, reason, exit_price=position.stop_loss)
            return None, True
        if position.target == position.target and (  # target not NaN
            (long and high >= position.target) or (not long and low <= position.target)
        ):
            self._close(portfolio, risk, position, close, ts, "target", exit_price=position.target)
            return None, True

        cfg = self.exit_cfg

        # 2. R-multiple partial scale-out.
        if cfg.partial_exit_r is not None and not position.partial_done and position.initial_risk > 0:
            partial_price = (
                position.entry_price + cfg.partial_exit_r * position.initial_risk if long
                else position.entry_price - cfg.partial_exit_r * position.initial_risk
            )
            touched = high >= partial_price if long else low <= partial_price
            if touched:
                part_qty = int(position.qty * cfg.partial_exit_pct)
                part_qty -= part_qty % self.risk_cfg.lot_size
                if 0 < part_qty < position.qty:
                    self._record_trade(portfolio, position, partial_price, part_qty, ts, "partial", risk, partial=True)
                    position.qty -= part_qty
                position.partial_done = True
                position.stop_loss = position.entry_price
                position.breakeven_done = True
                position.stop_moved = True

        # 3. Time stop.
        position.bars_held += 1
        if cfg.time_stop_bars is not None and position.bars_held >= cfg.time_stop_bars:
            self._close(portfolio, risk, position, close, ts, "time_stop")
            return None, True

        # 4. Update best price, then tighten the stop for next bars.
        position.best_price = max(position.best_price, high) if long else min(position.best_price, low)

        if cfg.breakeven_at_r is not None and not position.breakeven_done and position.initial_risk > 0:
            r_now = (
                (position.best_price - position.entry_price) if long
                else (position.entry_price - position.best_price)
            ) / position.initial_risk
            if r_now >= cfg.breakeven_at_r:
                new_stop = max(position.stop_loss, position.entry_price) if long else min(position.stop_loss, position.entry_price)
                if new_stop != position.stop_loss:
                    position.stop_loss = new_stop
                    position.stop_moved = True
                position.breakeven_done = True

        if cfg.trailing_atr_mult is not None and atr_val == atr_val:  # atr not NaN
            trail = (
                position.best_price - cfg.trailing_atr_mult * atr_val if long
                else position.best_price + cfg.trailing_atr_mult * atr_val
            )
            new_stop = max(position.stop_loss, trail) if long else min(position.stop_loss, trail)
            if new_stop != position.stop_loss:
                position.stop_loss = new_stop
                position.stop_moved = True

        return position, False

    def _close(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        position: OpenPosition,
        close: float,
        ts: dt.datetime,
        reason: str,
        exit_price: float | None = None,
    ) -> None:
        self._record_trade(portfolio, position, exit_price if exit_price is not None else close,
                           position.qty, ts, reason, risk, partial=False)

    def _record_trade(
        self,
        portfolio: Portfolio,
        position: OpenPosition,
        raw_exit_price: float,
        qty: int,
        ts: dt.datetime,
        reason: str,
        risk: RiskManager,
        partial: bool,
    ) -> None:
        fill_side = "sell" if position.side == "long" else "buy"
        exit_price_filled = self.slippage.adjust(raw_exit_price, fill_side)

        if position.side == "long":
            gross_pnl = (exit_price_filled - position.entry_price) * qty * self.risk_cfg.contract_multiplier
        else:
            gross_pnl = (position.entry_price - exit_price_filled) * qty * self.risk_cfg.contract_multiplier

        cost = self.cost_model.round_trip_cost(
            position.entry_price, exit_price_filled, qty, position.side
        ).total
        net_pnl = gross_pnl - cost

        portfolio.record_trade(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price_filled,
                qty=qty,
                gross_pnl=gross_pnl,
                cost=cost,
                net_pnl=net_pnl,
                exit_reason=reason,
            )
        )
        if partial:
            risk.register_partial(net_pnl)
        else:
            risk.register_exit(net_pnl)
