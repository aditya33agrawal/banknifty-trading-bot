import numpy as np
import pandas as pd

from banknifty_bot.features.filters import FilterConfig, apply_filters


def _df(close, freq="1D", start="2026-01-05 09:15"):
    idx = pd.date_range(start, periods=len(close), freq=freq, tz="Asia/Kolkata")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close,
         "volume": np.full(len(close), 500)}, index=idx
    )


def _signals(df, entries):
    return pd.DataFrame(
        {"entry": entries,
         "stop_loss": np.where(np.array(entries) != 0, df["close"] - 1, np.nan),
         "target": np.where(np.array(entries) != 0, df["close"] + 1, np.nan)},
        index=df.index,
    )


def test_disabled_config_is_noop():
    df = _df(range(100, 110))
    sig = _signals(df, [1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
    out = apply_filters(sig, df, FilterConfig())
    assert out["entry"].tolist() == sig["entry"].tolist()


def test_trend_filter_removes_wrong_side_entries():
    # Rising series: above its SMA late, below early. A short late should be dropped,
    # a long late kept.
    df = _df(list(range(100, 130)))
    entries = [0] * 30
    entries[28] = -1   # short near the top of an uptrend -> should be filtered out
    entries[29] = 1    # long in the uptrend -> kept
    sig = _signals(df, entries)
    out = apply_filters(sig, df, FilterConfig(trend_sma=10))
    assert out["entry"].iloc[28] == 0
    assert out["entry"].iloc[29] == 1


def test_time_filter_keeps_only_window():
    df = _df([100] * 6, freq="5min")  # 09:15..09:40
    sig = _signals(df, [1, 1, 1, 1, 1, 1])
    out = apply_filters(sig, df, FilterConfig(time_start="09:25", time_end="09:35"))
    kept = [t.strftime("%H:%M") for t in out.index[out["entry"] == 1].time]
    assert kept == ["09:25", "09:30", "09:35"]
