from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class Trade:
    entry_time: dt.datetime
    exit_time: dt.datetime
    side: str  # "long" | "short"
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str  # "stop" | "target" | "square_off"


@dataclass
class Portfolio:
    initial_equity: float
    equity: float = field(init=False)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[dt.datetime, float]] = field(default_factory=list)

    def __post_init__(self):
        self.equity = self.initial_equity

    def record_trade(self, trade: Trade) -> None:
        self.equity += trade.net_pnl
        self.trades.append(trade)

    def mark(self, ts: dt.datetime) -> None:
        self.equity_curve.append((ts, self.equity))

    def to_trades_df(self):
        import pandas as pd

        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def to_equity_df(self):
        import pandas as pd

        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"]).set_index("timestamp")
        df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        return df.set_index("timestamp")
