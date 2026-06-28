import numpy as np
import pandas as pd
import pytest

from banknifty_bot.backtest.portfolio_ensemble import combine_strategies
from banknifty_bot.config import load_config
from banknifty_bot.strategies.registry import get_strategy


@pytest.fixture
def daily_df():
    idx = pd.date_range("2018-01-01", periods=600, freq="1D", tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    close = 20000 + np.cumsum(rng.normal(4, 70, len(idx)))
    return pd.DataFrame(
        {"open": close, "high": close + 40, "low": close - 40, "close": close, "volume": 500},
        index=idx,
    )


def test_voting_ensemble_emits_valid_contract(daily_df):
    params = {
        "min_votes": 2, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 14,
        "members": [
            {"name": "ema_trend", "params": {"fast_span": 9, "slow_span": 21, "adx_threshold": 10}},
            {"name": "supertrend", "params": {"period": 10, "multiplier": 3.0, "min_atr_pct": 0.0}},
            {"name": "donchian", "params": {"channel": 20, "use_trend_filter": False}},
        ],
    }
    out = get_strategy("ensemble", params).generate_signals(daily_df)
    assert list(out.columns) == ["entry", "stop_loss", "target"]
    assert out["entry"].isin([-1, 0, 1]).all()


def test_portfolio_ensemble_blends_and_reports_corr(daily_df):
    cfg = load_config("config/config_daily.yaml")
    specs = [
        ("ema_trend", {"fast_span": 9, "slow_span": 21, "adx_threshold": 10}),
        ("donchian", {"channel": 20, "use_trend_filter": False}),
    ]
    res = combine_strategies(cfg, daily_df, specs)
    assert res.corr.shape[0] == res.corr.shape[1]            # square correlation matrix
    assert len(res.members) >= 1
    if res.metrics:
        assert "sharpe" in res.metrics and "max_drawdown_pct" in res.metrics
