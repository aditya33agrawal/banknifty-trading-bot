"""Flexible loader for third-party bulk intraday CSVs (plan §4 Tier C / §A.2).

Multi-year BankNifty 1-min data from Kaggle / GitHub / vendors comes in many shapes:
a combined datetime column, or separate Date + Time columns; full or abbreviated
OHLC names; with or without volume. `load_bulk_csv` auto-detects the common cases
(with manual overrides) and returns a tz-aware OHLCV frame ready for `clean_ohlcv`
and the partitioned store — so importing a new source is one command, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Candidate source column names (lower-cased) for each canonical field.
_OHLCV_CANDIDATES = {
    "open": ["open", "o", "op"],
    "high": ["high", "h", "hi"],
    "low": ["low", "l", "lo"],
    "close": ["close", "c", "cl", "last", "ltp"],
    "volume": ["volume", "vol", "v", "qty", "turnover"],
}
_DATETIME_SINGLE = ["timestamp", "datetime", "date_time", "datetimestamp", "time_stamp", "date time"]


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def load_bulk_csv(
    path: str | Path,
    datetime_col: str | None = None,
    date_col: str | None = None,
    time_col: str | None = None,
    col_map: dict[str, str] | None = None,
    tz: str = "Asia/Kolkata",
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Load a vendor CSV into a tz-aware OHLCV DataFrame (no cleaning yet).

    Args mirror the common variations; anything omitted is auto-detected.
    `col_map` maps canonical field -> source column to force a specific mapping.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = list(df.columns)
    col_map = {k: v.lower() for k, v in (col_map or {}).items()}

    # --- timestamp ---------------------------------------------------------
    dt_single = datetime_col.lower() if datetime_col else _pick(cols, _DATETIME_SINGLE)
    d_col = date_col.lower() if date_col else ("date" if "date" in cols else None)
    t_col = time_col.lower() if time_col else ("time" if "time" in cols else None)

    if dt_single and dt_single in cols:
        ts = pd.to_datetime(df[dt_single], dayfirst=dayfirst, errors="coerce")
    elif d_col and t_col:
        ts = pd.to_datetime(df[d_col].astype(str) + " " + df[t_col].astype(str),
                            dayfirst=dayfirst, errors="coerce")
    elif d_col:
        ts = pd.to_datetime(df[d_col], dayfirst=dayfirst, errors="coerce")
    else:
        raise ValueError(
            f"Could not find a datetime column. Columns seen: {cols}. "
            "Pass --datetime-col, or --date-col and --time-col."
        )

    # --- OHLCV -------------------------------------------------------------
    out = pd.DataFrame(index=pd.DatetimeIndex(ts, name="timestamp"))
    for field, candidates in _OHLCV_CANDIDATES.items():
        src = col_map.get(field) or _pick(cols, candidates)
        if src and src in cols:
            out[field] = pd.to_numeric(df[src].values, errors="coerce")
        elif field == "volume":
            out["volume"] = 0.0  # many index datasets omit volume
        else:
            raise ValueError(f"Could not find a '{field}' column among {cols}. Use --col-map {field}=<name>.")

    out = out[out.index.notna()].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)

    return out[["open", "high", "low", "close", "volume"]]
