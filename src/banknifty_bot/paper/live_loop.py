"""Market-hours paper-trading loop skeleton (plan §10).

Each tick: fetch latest bars -> strategy.generate_signals -> risk checks ->
PaperBroker orders -> log every decision -> enforce daily loss limit + square-off.
Uses the *same* Strategy + CostModel objects as the backtest engine, so this is
not a separate code path to drift out of sync.

This is a skeleton: it polls on a fixed interval rather than using a real
scheduler/dependency, and assumes the configured DataProvider supports
near-real-time bars (yfinance intraday data is delayed/rate-limited — verify
before relying on it for paper trading decisions).
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.risk import RiskConfig, RiskManager
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.data.providers.base import DataProvider
from banknifty_bot.strategies.base import Strategy
from banknifty_bot.utils.calendar import is_trading_day
from banknifty_bot.utils.logging import get_logger

from .paper_broker import PaperBroker

log = get_logger(__name__)


class LiveLoop:
    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        symbol: str,
        interval: str,
        risk_cfg: RiskConfig,
        cost_model: CostModel,
        slippage: Slippage,
        poll_seconds: int = 60,
        lookback_days: int = 5,
    ):
        self.provider = provider
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.risk_cfg = risk_cfg
        self.poll_seconds = poll_seconds
        self.lookback_days = lookback_days

        self.broker = PaperBroker(cost_model, slippage, risk_cfg.contract_multiplier, risk_cfg.initial_equity)
        self.risk = RiskManager(risk_cfg)
        self._current_day: dt.date | None = None

    def _fetch_recent_bars(self) -> pd.DataFrame:
        end = dt.datetime.now(tz=dt.timezone.utc)
        start = end - dt.timedelta(days=self.lookback_days)
        return self.provider.get_ohlcv(self.symbol, start.isoformat(), end.isoformat(), self.interval)

    def run_once(self) -> None:
        now = dt.datetime.now(tz=dt.timezone.utc).astimezone()
        today = now.date()

        if not is_trading_day(today):
            log.info("Not a trading day (%s) — skipping.", today)
            return

        if today != self._current_day:
            self._current_day = today
            self.risk.new_day(today)
            log.info("New trading day: %s", today)

        df = self._fetch_recent_bars()
        if df.empty:
            log.warning("No bars returned from provider.")
            return

        last_ts = df.index[-1]
        last_bar = df.iloc[-1]

        if self.broker.has_open_position():
            reason = self.broker.check_stop_target(last_bar)
            if reason or self.risk.past_square_off(last_ts):
                self.broker.close_position(last_bar["close"], last_ts, reason or "square_off")
                self.risk.register_exit(self.broker.portfolio.trades[-1].net_pnl)
            return

        if self.risk.past_square_off(last_ts):
            return

        signals = self.strategy.generate_signals(df)
        sig = signals.iloc[-1]

        if sig["entry"] != 0 and self.risk.can_open_position():
            side = "long" if sig["entry"] == 1 else "short"
            qty = self.risk.position_size(last_bar["close"], sig["stop_loss"])
            if qty > 0:
                opened = self.broker.open_position(
                    side, qty, last_bar["close"], last_ts, sig["stop_loss"], sig["target"]
                )
                if opened:
                    self.risk.register_entry()
        else:
            log.info("No actionable signal at %s.", last_ts)

    def run_forever(self) -> None:
        log.info("Starting paper-trading loop (poll every %ds).", self.poll_seconds)
        while True:
            try:
                self.run_once()
            except Exception:
                log.exception("Error in live loop tick.")
            time.sleep(self.poll_seconds)
