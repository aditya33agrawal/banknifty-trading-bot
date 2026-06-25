from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from banknifty_bot.features.indicators import atr as atr_indicator

from .costs import CostModel
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


class BacktestEngine:
    """Bar-by-bar simulation: signals -> risk/sizing -> fills w/ slippage ->
    costs -> portfolio/ledger -> daily limits & square-off.

    Intrabar mechanics: on each bar after entry, checks whether the bar's
    high/low would have hit the stop or target first (stop assumed to trigger
    first if both are touched in the same bar — conservative).
    """

    def __init__(
        self,
        risk_cfg: RiskConfig,
        cost_model: CostModel,
        slippage: Slippage,
        atr_window: int = 14,
    ):
        self.risk_cfg = risk_cfg
        self.cost_model = cost_model
        self.slippage = slippage
        self.atr_window = atr_window

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
                if position is not None:
                    position = self._close(portfolio, risk, position, bar, ts, "square_off")
                current_day = day
                risk.new_day(day)
                session_start_ts = ts
                session_end_ts = pd.Timestamp.combine(day, self.risk_cfg.square_off).tz_localize(ts.tzinfo)

            if position is not None:
                position, closed = self._check_exit(portfolio, risk, position, bar, ts)
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
        )

    def _check_exit(
        self, portfolio: Portfolio, risk: RiskManager, position: OpenPosition, bar: pd.Series, ts: dt.datetime
    ) -> tuple[OpenPosition | None, bool]:
        if risk.past_square_off(ts):
            self._close(portfolio, risk, position, bar, ts, "square_off")
            return None, True

        if position.side == "long":
            stop_hit = bar["low"] <= position.stop_loss
            target_hit = bar["high"] >= position.target
        else:
            stop_hit = bar["high"] >= position.stop_loss
            target_hit = bar["low"] <= position.target

        if stop_hit:
            self._close(portfolio, risk, position, bar, ts, "stop", exit_price=position.stop_loss)
            return None, True
        if target_hit:
            self._close(portfolio, risk, position, bar, ts, "target", exit_price=position.target)
            return None, True

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
        fill_side = "sell" if position.side == "long" else "buy"
        exit_price_filled = self.slippage.adjust(raw_exit_price, fill_side)

        if position.side == "long":
            gross_pnl = (exit_price_filled - position.entry_price) * position.qty * self.risk_cfg.contract_multiplier
        else:
            gross_pnl = (position.entry_price - exit_price_filled) * position.qty * self.risk_cfg.contract_multiplier

        cost = self.cost_model.round_trip_cost(
            position.entry_price, exit_price_filled, position.qty, position.side
        ).total
        net_pnl = gross_pnl - cost

        portfolio.record_trade(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price_filled,
                qty=position.qty,
                gross_pnl=gross_pnl,
                cost=cost,
                net_pnl=net_pnl,
                exit_reason=reason,
            )
        )
        risk.register_exit(net_pnl)
