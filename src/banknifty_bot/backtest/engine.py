from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

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
    initial_risk: float = 0.0    # |entry - initial_stop| per unit, for R-multiples
    best_price: float = 0.0      # most favourable price seen (for trailing)
    bars_held: int = 0
    partial_done: bool = False
    breakeven_done: bool = False
    stop_moved: bool = False      # stop tightened from its initial level (trail/breakeven)


class BacktestEngine:
    """Bar-by-bar simulation: signals -> risk/sizing -> fills w/ slippage ->
    costs -> portfolio/ledger -> daily limits & square-off.

    Intrabar mechanics: on each bar after entry, checks whether the bar's
    high/low would have hit the stop or target first (stop assumed to trigger
    first if both are touched in the same bar — conservative).

    Optional exit management (`ExitConfig`): ATR trailing stop, breakeven shift,
    R-multiple partial scale-out, and time stop. The stop level used for a hit
    test is the one carried *into* the bar; trailing/breakeven updates apply to
    subsequent bars, so there is no intrabar lookahead.
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

        position: OpenPosition | None = None
        current_day: dt.date | None = None
        session_start_ts: dt.datetime | None = None

        for ts, bar in df.iterrows():
            day = ts.date()
            if day != current_day:
                # Intraday: flatten any carried position at the session boundary.
                # Swing (intraday=False): hold across days; exits are stop/target only.
                if self.risk_cfg.intraday and position is not None:
                    self._close(portfolio, risk, position, bar, ts, "square_off")
                    position = None
                current_day = day
                risk.new_day(day)
                session_start_ts = ts
                session_end_ts = pd.Timestamp.combine(day, self.risk_cfg.square_off).tz_localize(ts.tzinfo)

            if position is not None:
                position, closed = self._check_exit(portfolio, risk, position, bar, ts, atr_series.loc[ts])
                if closed:
                    position = None

            if position is None and risk.past_square_off(ts):
                portfolio.mark(ts)
                continue

            if position is None and risk.can_open_position():
                sig = signals.loc[ts]
                if sig["entry"] != 0 and not risk.in_no_trade_window(
                    ts, session_start_ts, session_end_ts
                ):
                    position = self._open(
                        risk, sig, bar, ts, atr_series.loc[ts]
                    )

            portfolio.mark(ts)

        if position is not None:
            last_ts = df.index[-1]
            self._close(portfolio, risk, position, df.iloc[-1], last_ts, "square_off")

        return portfolio

    def _open(self, risk: RiskManager, sig: pd.Series, bar: pd.Series, ts: dt.datetime, atr_val: float) -> OpenPosition | None:
        side = "long" if sig["entry"] == 1 else "short"
        fill_side = "buy" if side == "long" else "sell"
        entry_price = self.slippage.adjust(bar["close"], fill_side, atr_val)

        qty = risk.position_size(entry_price, sig["stop_loss"])
        if qty <= 0:
            return None

        risk.register_entry()
        return OpenPosition(
            side=side,
            entry_time=ts,
            entry_price=entry_price,
            qty=qty,
            stop_loss=sig["stop_loss"],
            target=sig["target"],
            initial_risk=abs(entry_price - sig["stop_loss"]),
            best_price=entry_price,
        )

    def _check_exit(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        position: OpenPosition,
        bar: pd.Series,
        ts: dt.datetime,
        atr_val: float,
    ) -> tuple[OpenPosition | None, bool]:
        if risk.past_square_off(ts):
            self._close(portfolio, risk, position, bar, ts, "square_off")
            return None, True

        long = position.side == "long"

        # 1. Hard stop / target using the levels carried into this bar.
        if (long and bar["low"] <= position.stop_loss) or (not long and bar["high"] >= position.stop_loss):
            reason = "trail_stop" if position.stop_moved else "stop"
            self._close(portfolio, risk, position, bar, ts, reason, exit_price=position.stop_loss)
            return None, True
        if position.target == position.target and (  # target not NaN
            (long and bar["high"] >= position.target) or (not long and bar["low"] <= position.target)
        ):
            self._close(portfolio, risk, position, bar, ts, "target", exit_price=position.target)
            return None, True

        cfg = self.exit_cfg

        # 2. R-multiple partial scale-out (treated like a target touch on part of the qty).
        if cfg.partial_exit_r is not None and not position.partial_done and position.initial_risk > 0:
            partial_price = (
                position.entry_price + cfg.partial_exit_r * position.initial_risk if long
                else position.entry_price - cfg.partial_exit_r * position.initial_risk
            )
            touched = bar["high"] >= partial_price if long else bar["low"] <= partial_price
            if touched:
                part_qty = int(position.qty * cfg.partial_exit_pct)
                part_qty -= part_qty % self.risk_cfg.lot_size  # keep whole lots
                if 0 < part_qty < position.qty:
                    self._record_trade(portfolio, position, partial_price, part_qty, ts, "partial", risk, partial=True)
                    position.qty -= part_qty
                position.partial_done = True
                position.stop_loss = position.entry_price  # protect the runner
                position.breakeven_done = True
                position.stop_moved = True

        # 3. Time stop.
        position.bars_held += 1
        if cfg.time_stop_bars is not None and position.bars_held >= cfg.time_stop_bars:
            self._close(portfolio, risk, position, bar, ts, "time_stop")
            return None, True

        # 4. Update best price, then tighten the stop (trailing / breakeven) for next bars.
        position.best_price = max(position.best_price, bar["high"]) if long else min(position.best_price, bar["low"])

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
        bar: pd.Series,
        ts: dt.datetime,
        reason: str,
        exit_price: float | None = None,
    ) -> None:
        raw_exit_price = exit_price if exit_price is not None else bar["close"]
        self._record_trade(portfolio, position, raw_exit_price, position.qty, ts, reason, risk, partial=False)

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
