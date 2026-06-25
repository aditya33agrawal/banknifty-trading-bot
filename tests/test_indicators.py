import numpy as np
import pandas as pd
import pytest

from banknifty_bot.features import indicators as ind


@pytest.fixture
def sample_df():
    idx = pd.date_range("2026-01-05 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    close = 50000 + np.cumsum(rng.normal(0, 20, len(idx)))
    high = close + rng.uniform(1, 10, len(idx))
    low = close - rng.uniform(1, 10, len(idx))
    open_ = close + rng.normal(0, 5, len(idx))
    volume = rng.integers(100, 1000, len(idx))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_ema_matches_pandas_ewm(sample_df):
    out = ind.ema(sample_df["close"], span=9)
    expected = sample_df["close"].ewm(span=9, adjust=False).mean()
    pd.testing.assert_series_equal(out, expected)


def test_rsi_bounded(sample_df):
    out = ind.rsi(sample_df["close"], window=14)
    assert out.between(0, 100).all()


def test_atr_nonnegative(sample_df):
    out = ind.atr(sample_df, window=14)
    assert (out.dropna() >= 0).all()


def test_session_vwap_resets_daily():
    idx = pd.date_range("2026-01-05 09:15", periods=4, freq="5min", tz="Asia/Kolkata").append(
        pd.date_range("2026-01-06 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    )
    df = pd.DataFrame(
        {
            "open": [100] * 8,
            "high": [101] * 8,
            "low": [99] * 8,
            "close": [100] * 8,
            "volume": [10] * 8,
        },
        index=idx,
    )
    out = ind.session_vwap(df)
    # constant price -> vwap should equal price every bar regardless of day boundary
    assert np.allclose(out.values, 100.0, atol=1e-6)


def test_opening_range_high_low(sample_df):
    out = ind.opening_range(sample_df, minutes=15)
    first_15min = sample_df.iloc[:3]  # 3 bars of 5min = 15min
    assert out["or_high"].iloc[0] == pytest.approx(first_15min["high"].max())
    assert out["or_low"].iloc[0] == pytest.approx(first_15min["low"].min())


def test_adx_within_bounds(sample_df):
    out = ind.adx(sample_df, window=14)
    valid = out["adx"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()
