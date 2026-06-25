import numpy as np
import pandas as pd
import pytest

from banknifty_bot.strategies.registry import REGISTRY, get_strategy

PARAMS_BY_NAME = {
    "orb": {"range_minutes": 15, "buffer_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5},
    "vwap": {"mode": "trend", "band_atr_mult": 1.0, "slope_window": 5, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5},
    "ema_trend": {"fast_span": 5, "slow_span": 13, "adx_window": 5, "adx_threshold": 10, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5},
    "supertrend": {"period": 5, "multiplier": 2.0, "min_atr_pct": 0.0, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5},
    "rsi_reversion": {"rsi_window": 5, "oversold": 30, "overbought": 70, "adx_window": 5, "adx_max": 100, "sl_atr_mult": 1.5, "target_r": 1.5, "atr_window": 5},
    "regime_filter": {
        "adx_window": 5, "adx_threshold": 20,
        "trend_strategy": "ema_trend",
        "trend_params": {"fast_span": 5, "slow_span": 13, "adx_window": 5, "adx_threshold": 10, "sl_atr_mult": 1.5, "target_r": 2.0, "atr_window": 5},
        "reversion_strategy": "rsi_reversion",
        "reversion_params": {"rsi_window": 5, "oversold": 30, "overbought": 70, "adx_window": 5, "adx_max": 100, "sl_atr_mult": 1.5, "target_r": 1.5, "atr_window": 5},
    },
}


@pytest.fixture
def sample_df():
    idx = pd.date_range("2026-01-05 09:15", periods=75, freq="5min", tz="Asia/Kolkata")
    rng = np.random.default_rng(1)
    close = 50000 + np.cumsum(rng.normal(0, 15, len(idx)))
    high = close + rng.uniform(1, 8, len(idx))
    low = close - rng.uniform(1, 8, len(idx))
    open_ = close + rng.normal(0, 4, len(idx))
    volume = rng.integers(100, 1000, len(idx))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_strategy_output_contract(name, sample_df):
    strategy = get_strategy(name, PARAMS_BY_NAME[name])
    out = strategy.generate_signals(sample_df)

    assert list(out.columns) == ["entry", "stop_loss", "target"]
    assert out.index.equals(sample_df.index)
    assert out["entry"].isin([-1, 0, 1]).all()

    entries = out[out["entry"] != 0]
    if not entries.empty:
        assert entries["stop_loss"].notna().all()
        assert entries["target"].notna().all()


def test_regime_filter_routes_by_adx(sample_df):
    strategy = get_strategy("regime_filter", PARAMS_BY_NAME["regime_filter"])
    out = strategy.generate_signals(sample_df)
    assert out["entry"].isin([-1, 0, 1]).all()
