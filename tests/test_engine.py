import datetime as dt

import numpy as np
import pandas as pd
import pytest

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.engine import BacktestEngine
from banknifty_bot.backtest.exits import ExitConfig
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


def test_swing_mode_holds_position_overnight():
    """With intraday=False, a position must be held across days until its stop/target
    hits — the daily square-off is disabled."""
    idx = pd.date_range("2026-01-05", periods=10, freq="1D", tz="Asia/Kolkata")
    close = 50000 + np.arange(10) * 100  # steady uptrend
    df = pd.DataFrame(
        {"open": close, "high": close + 50, "low": close - 50, "close": close,
         "volume": np.full(10, 500)}, index=idx
    )
    # Enter long on bar 0; wide stop, target only reached after several days.
    signals = pd.DataFrame(
        {"entry": [1] + [0] * 9,
         "stop_loss": [49900.0] + [np.nan] * 9,
         "target": [50500.0] + [np.nan] * 9},
        index=idx,
    )
    risk_cfg = RiskConfig(
        initial_equity=1_000_000.0, risk_per_trade_pct=1.0, daily_max_loss_pct=100.0,
        max_trades_per_day=10, max_open_positions=1, lot_size=15, contract_multiplier=1.0,
        intraday=False,
    )
    eng = BacktestEngine(risk_cfg, CostModel("futures", FUTURES_PARAMS), Slippage("ticks", 0.0), atr_window=5)
    trades = eng.run(df, signals).to_trades_df()

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_time"].date() > t["entry_time"].date()  # held overnight
    assert t["exit_reason"] == "target"


def _swing_engine(exit_cfg, atr_window=2):
    risk_cfg = RiskConfig(
        initial_equity=1_000_000.0, risk_per_trade_pct=1.0, daily_max_loss_pct=100.0,
        max_trades_per_day=10, max_open_positions=1, lot_size=15, contract_multiplier=1.0,
        intraday=False,
    )
    return BacktestEngine(risk_cfg, CostModel("futures", FUTURES_PARAMS),
                          Slippage("ticks", 0.0), atr_window=atr_window, exit_cfg=exit_cfg)


def _bars(rows):
    idx = pd.date_range("2026-01-05", periods=len(rows), freq="1D", tz="Asia/Kolkata")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 500
    return df


def _entry_signals(df, stop, target):
    n = len(df)
    return pd.DataFrame(
        {"entry": [1] + [0] * (n - 1),
         "stop_loss": [stop] + [np.nan] * (n - 1),
         "target": [target] + [np.nan] * (n - 1)},
        index=df.index,
    )


def test_time_stop_exits_after_n_bars():
    df = _bars([[100, 101, 99, 100]] * 6)  # flat — neither stop nor target hits
    sig = _entry_signals(df, stop=90.0, target=200.0)
    trades = _swing_engine(ExitConfig(time_stop_bars=3)).run(df, sig).to_trades_df()
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "time_stop"
    # entry on bar 0, exit on bar 3 (3 bars held)
    assert (trades.iloc[0]["exit_time"] - trades.iloc[0]["entry_time"]).days == 3


def test_breakeven_locks_entry_price():
    df = _bars([
        [100, 100, 100, 100],   # entry
        [100, 102, 100, 101],   # reaches +1R (risk=2) -> stop to breakeven 100
        [100, 100, 98, 99],     # pulls back through 100 -> exit at breakeven
        [99, 99, 99, 99],
    ])
    sig = _entry_signals(df, stop=98.0, target=200.0)  # initial risk = 2
    trades = _swing_engine(ExitConfig(breakeven_at_r=1.0)).run(df, sig).to_trades_df()
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "trail_stop"
    assert t["exit_price"] == pytest.approx(100.0)  # breakeven, not the 98 initial stop


def test_partial_exit_scales_out_then_runs():
    df = _bars([
        [100, 100, 100, 100],   # entry, risk = 1
        [100, 101, 100, 100.5], # touches +1R=101 -> partial out 50%, stop->breakeven 100
        [100, 100, 99, 99.5],   # back through 100 -> remainder exits at breakeven
        [99, 99, 99, 99],
    ])
    sig = _entry_signals(df, stop=99.0, target=200.0)
    trades = _swing_engine(ExitConfig(partial_exit_r=1.0, partial_exit_pct=0.5)).run(df, sig).to_trades_df()
    assert len(trades) == 2
    assert set(trades["exit_reason"]) == {"partial", "trail_stop"}
    partial = trades[trades["exit_reason"] == "partial"].iloc[0]
    runner = trades[trades["exit_reason"] == "trail_stop"].iloc[0]
    assert partial["qty"] + runner["qty"] == trades["qty"].sum()
    assert partial["exit_price"] == pytest.approx(101.0)


def test_trailing_stop_locks_profit():
    # range 2 each bar so ATR(2) -> 2; trailing mult 1 keeps stop 2 below the high.
    df = _bars([
        [100, 101, 99, 100],    # entry, initial stop 96
        [100, 102, 100, 101],   # high 102 -> trail 100
        [101, 103, 101, 102],   # high 103 -> trail 101
        [101, 101, 99, 99],     # low 99 < trail 101 -> exit at 101 (profit locked)
    ])
    sig = _entry_signals(df, stop=96.0, target=200.0)
    trades = _swing_engine(ExitConfig(trailing_atr_mult=1.0), atr_window=2).run(df, sig).to_trades_df()
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "trail_stop"
    assert t["exit_price"] > 96.0 and t["gross_pnl"] > 0


def test_equity_curve_matches_final_equity(two_day_df, engine):
    strategy = OpeningRangeBreakout(
        {"range_minutes": 15, "buffer_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5}
    )
    signals = strategy.generate_signals(two_day_df)
    portfolio = engine.run(two_day_df, signals)
    equity_df = portfolio.to_equity_df()

    assert not equity_df.empty
    assert equity_df["equity"].iloc[-1] == pytest.approx(portfolio.equity)
