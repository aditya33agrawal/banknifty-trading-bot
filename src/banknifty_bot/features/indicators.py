"""Vectorized technical indicators for intraday OHLCV bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to the start of each trading session (resets daily)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_basic = hl2 + multiplier * atr_val
    lower_basic = hl2 - multiplier * atr_val

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    close = df["close"]

    for i in range(1, len(df)):
        if close.iloc[i - 1] > upper.iloc[i - 1]:
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i - 1])

        if close.iloc[i - 1] < lower.iloc[i - 1]:
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i - 1])

    direction = pd.Series(index=df.index, dtype=int)
    direction.iloc[0] = 1
    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    trend_line = np.where(direction == 1, lower, upper)
    return pd.DataFrame(
        {"supertrend": trend_line, "direction": direction}, index=df.index
    )


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_val = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        / atr_val
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        / atr_val
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window).std()
    return pd.DataFrame(
        {"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std}
    )


def opening_range(df: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    """First-N-minute high/low of each session, broadcast across that day's bars."""
    day = df.index.normalize()
    session_start = df.groupby(day).apply(lambda g: g.index.min())
    cutoff = session_start.reindex(day).values + pd.Timedelta(minutes=minutes)

    in_range = df.index.values < cutoff
    or_high = df["high"].where(in_range).groupby(day).cummax().groupby(day).transform("max")
    or_low = df["low"].where(in_range).groupby(day).cummin().groupby(day).transform("min")

    return pd.DataFrame({"or_high": or_high, "or_low": or_low}, index=df.index)


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window).std()


def time_of_day(df: pd.DataFrame) -> pd.Series:
    return df.index.time


def prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.normalize()
    daily = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev = daily.shift(1)
    mapped = prev.reindex(day)
    mapped.index = df.index
    return mapped.rename(columns={"high": "prev_high", "low": "prev_low", "close": "prev_close"})
