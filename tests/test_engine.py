import datetime as dt

import numpy as np
import pandas as pd
import pytest

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.engine import BacktestEngine
from banknifty_bot.backtest.risk import RiskConfig
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.strategies.orb import OpeningRangeBreakout

FUTURES_PARAMS = {
    "brokerage_per_order": 20.0,
    "stt_pct_on_sell_value": 0.02,
    "exchange_txn_pct_on_value": 0.0019,
    "sebi_pct_on_value": 0.0001,
    "stamp_duty_pct_on_buy_value": 0.002,
    "gst_pct": 18.0,
}


def make_trending_day(date_str, start_price, drift, n_bars=75, freq="5min"):
    idx = pd.date_range(f"{date_str} 09:15", periods=n_bars, freq=freq, tz="Asia/Kolkata")
    close = start_price + np.arange(n_bars) * drift
    high = close + 5
    low = close - 5
    open_ = close - drift / 2
    volume = np.full(n_bars, 500)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


@pytest.fixture
def two_day_df():
    day1 = make_trending_day("2026-01-05", 50000, drift=5)
    day2 = make_trending_day("2026-01-06", 50500, drift=-5)
    return pd.concat([day1, day2])


@pytest.fixture
def engine():
    risk_cfg = RiskConfig(
        initial_equity=1_000_000.0,
        risk_per_trade_pct=1.0,
        daily_max_loss_pct=3.0,
        max_trades_per_day=4,
        max_open_positions=1,
        lot_size=15,
        contract_multiplier=1.0,
        square_off=dt.time(15, 15),
        no_trade_first_minutes=0,
        no_trade_last_minutes=0,
    )
    cost_model = CostModel("futures", FUTURES_PARAMS)
    slippage = Slippage("ticks", 0.0)
    return BacktestEngine(risk_cfg, cost_model, slippage, atr_window=5)


def test_single_position_at_a_time(two_day_df, engine):
    strategy = OpeningRangeBreakout(
        {"range_minutes": 15, "buffer_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5}
    )
    signals = strategy.generate_signals(two_day_df)
    portfolio = engine.run(two_day_df, signals)
    trades = portfolio.to_trades_df()

    if not trades.empty:
        # no overlapping trades
        for i in range(len(trades) - 1):
            assert trades.iloc[i]["exit_time"] <= trades.iloc[i + 1]["entry_time"]


def test_square_off_no_overnight_position(two_day_df, engine):
    strategy = OpeningRangeBreakout(
        {"range_minutes": 15, "buffer_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5}
    )
    signals = strategy.generate_signals(two_day_df)
    portfolio = engine.run(two_day_df, signals)
    trades = portfolio.to_trades_df()

    if not trades.empty:
        for _, t in trades.iterrows():
            assert t["entry_time"].date() == t["exit_time"].date()
            assert t["exit_time"].time() <= dt.time(15, 30)


def test_costs_reduce_net_pnl_vs_gross(two_day_df, engine):
    strategy = OpeningRangeBreakout(
        {"range_minutes": 15, "buffer_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5}
    )
    signals = strategy.generate_signals(two_day_df)
    portfolio = engine.run(two_day_df, signals)
    trades = portfolio.to_trades_df()

    if not trades.empty:
        assert (trades["net_pnl"] == trades["gross_pnl"] - trades["cost"]).all()
        assert (trades["cost"] > 0).all()


def test_equity_curve_matches_final_equity(two_day_df, engine):
    strategy = OpeningRangeBreakout(
        {"range_minutes": 15, "buffer_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5}
    )
    signals = strategy.generate_signals(two_day_df)
    portfolio = engine.run(two_day_df, signals)
    equity_df = portfolio.to_equity_df()

    assert not equity_df.empty
    assert equity_df["equity"].iloc[-1] == pytest.approx(portfolio.equity)
