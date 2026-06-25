import numpy as np
import pandas as pd
import pytest

from banknifty_bot.evaluation import metrics as M


@pytest.fixture
def equity_df():
    idx = pd.date_range("2026-01-01", periods=30, freq="1D", tz="Asia/Kolkata")
    equity = 1_000_000 * (1.001 ** np.arange(30))
    return pd.DataFrame({"equity": equity}, index=idx)


@pytest.fixture
def trades_df():
    return pd.DataFrame(
        {
            "entry_time": pd.date_range("2026-01-01 09:30", periods=10, freq="1D", tz="Asia/Kolkata"),
            "exit_time": pd.date_range("2026-01-01 10:30", periods=10, freq="1D", tz="Asia/Kolkata"),
            "gross_pnl": [1000, -500, 800, -300, 1200, -400, 900, -200, 700, -600],
            "cost": [50] * 10,
            "net_pnl": [950, -550, 750, -350, 1150, -450, 850, -250, 650, -650],
        }
    )


def test_max_drawdown_zero_for_monotonic_equity(equity_df):
    dd = M.max_drawdown(equity_df)
    assert dd["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-6)


def test_trade_stats_win_rate(trades_df):
    stats = M.trade_stats(trades_df)
    assert stats["n_trades"] == 10
    assert stats["win_rate_pct"] == pytest.approx(50.0)


def test_cost_stats_positive(trades_df):
    stats = M.cost_stats(trades_df)
    assert stats["total_cost"] == pytest.approx(500.0)
    assert stats["cost_per_trade"] == pytest.approx(50.0)


def test_full_report_has_expected_keys(equity_df, trades_df):
    report = M.full_report(equity_df, trades_df)
    for key in ["cagr_pct", "sharpe", "sortino", "calmar", "max_drawdown_pct", "n_trades", "total_cost"]:
        assert key in report
