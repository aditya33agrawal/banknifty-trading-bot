"""NSE trading calendar: holidays, session window, square-off cutoff.

Holiday list is maintained manually (NSE publishes it yearly) — verify before
relying on it for live/paper trading far in the future.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

TIMEZONE = "Asia/Kolkata"

SESSION_START = dt.time(9, 15)
SESSION_END = dt.time(15, 30)
DEFAULT_SQUARE_OFF = dt.time(15, 15)

# NSE equity/F&O holidays. Extend yearly from the official NSE circular.
NSE_HOLIDAYS: set[dt.date] = {
    # 2024
    dt.date(2024, 1, 22), dt.date(2024, 1, 26), dt.date(2024, 3, 8),
    dt.date(2024, 3, 25), dt.date(2024, 3, 29), dt.date(2024, 4, 11),
    dt.date(2024, 4, 17), dt.date(2024, 5, 1), dt.date(2024, 5, 20),
    dt.date(2024, 6, 17), dt.date(2024, 7, 17), dt.date(2024, 8, 15),
    dt.date(2024, 10, 2), dt.date(2024, 11, 1), dt.date(2024, 11, 15),
    dt.date(2024, 12, 25),
    # 2025
    dt.date(2025, 2, 26), dt.date(2025, 3, 14), dt.date(2025, 3, 31),
    dt.date(2025, 4, 10), dt.date(2025, 4, 14), dt.date(2025, 4, 18),
    dt.date(2025, 5, 1), dt.date(2025, 8, 15), dt.date(2025, 8, 27),
    dt.date(2025, 10, 2), dt.date(2025, 10, 21), dt.date(2025, 10, 22),
    dt.date(2025, 11, 5), dt.date(2025, 12, 25),
    # 2026 (placeholder — confirm against the official NSE circular)
    dt.date(2026, 1, 26), dt.date(2026, 3, 4), dt.date(2026, 8, 15),
    dt.date(2026, 10, 2), dt.date(2026, 12, 25),
}


def is_trading_day(date: dt.date) -> bool:
    if date.weekday() >= 5:  # Sat/Sun
        return False
    return date not in NSE_HOLIDAYS


def session_bounds(date: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime.combine(date, SESSION_START)
    end = dt.datetime.combine(date, SESSION_END)
    return start, end


def square_off_cutoff(date: dt.date, cutoff: dt.time = DEFAULT_SQUARE_OFF) -> dt.datetime:
    return dt.datetime.combine(date, cutoff)


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    days = pd.date_range(start, end, freq="D")
    return [d.date() for d in days if is_trading_day(d.date())]


def filter_session(df: pd.DataFrame, index_col: str | None = None) -> pd.DataFrame:
    """Keep only rows that fall on NSE trading days within the session window."""
    idx = df.index if index_col is None else pd.DatetimeIndex(df[index_col])
    if idx.tz is not None:
        idx = idx.tz_convert(TIMEZONE)
    mask = (
        idx.to_series(index=df.index).apply(lambda ts: is_trading_day(ts.date()))
        & (idx.time >= SESSION_START)
        & (idx.time <= SESSION_END)
    )
    return df.loc[mask.values]
