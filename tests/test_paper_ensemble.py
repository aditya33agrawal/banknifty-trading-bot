import numpy as np
import pandas as pd

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.exits import ExitConfig
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.paper.portfolio_paper import PortfolioPaperTrader, Sleeve

FUTURES = {
    "brokerage_per_order": 20.0, "stt_pct_on_sell_value": 0.02,
    "exchange_txn_pct_on_value": 0.0019, "sebi_pct_on_value": 0.0001,
    "stamp_duty_pct_on_buy_value": 0.002, "gst_pct": 18.0,
}


def _trader(tmp_path, risk=5.0):
    sleeves = [
        Sleeve("ema_trend", {"fast_span": 5, "slow_span": 13, "adx_threshold": 10}, 0.5),
        Sleeve("donchian", {"channel": 20, "use_trend_filter": False}, 0.5),
    ]
    return PortfolioPaperTrader(
        sleeves=sleeves, cost_model=CostModel("futures", FUTURES), slippage=Slippage("ticks", 0.0),
        initial_equity=1_000_000, risk_per_trade_pct=risk, lot_size=15, contract_multiplier=1.0,
        exit_cfg=ExitConfig(trailing_atr_mult=2.0), atr_window=14,
        state_path=tmp_path / "paper.json",
    )


def _daily_df(n=400):
    idx = pd.date_range("2018-01-01", periods=n, freq="1D", tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    close = 8000 + np.cumsum(rng.normal(2, 40, n))   # low index so 5% risk affords lots
    return pd.DataFrame(
        {"open": close, "high": close + 30, "low": close - 30, "close": close, "volume": 500}, index=idx
    )


def test_backfill_builds_state_and_persists(tmp_path):
    trader = _trader(tmp_path)
    df = _daily_df()
    trader.backfill(df, 200)
    assert (tmp_path / "paper.json").exists()
    st = trader.status()
    assert st["last_ts"] == str(df.index[-1])
    assert set(st["sleeves"]) == {"ema_trend", "donchian"}
    # trades, if any, must obey net = gross - cost
    if not st["trades"].empty:
        t = st["trades"]
        assert np.allclose(t["net_pnl"], t["gross_pnl"] - t["cost"])


def test_backfill_is_idempotent_on_rerun(tmp_path):
    df = _daily_df()
    t1 = _trader(tmp_path)
    t1.backfill(df, 200)
    n_trades = len(t1.status()["trades"])
    # reload state and replay same bars -> no new trades (all bars already processed)
    t2 = _trader(tmp_path)
    t2.backfill(df, 200)
    assert len(t2.status()["trades"]) == n_trades


def test_step_advances_one_new_bar(tmp_path):
    df = _daily_df()
    trader = _trader(tmp_path)
    trader.backfill(df.iloc[:-1], 200)
    last_before = trader.state["last_ts"]
    trader.step({s.name: df for s in trader.sleeves}, df.index[-1])
    assert trader.state["last_ts"] == str(df.index[-1]) != last_before
