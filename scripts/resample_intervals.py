#!/usr/bin/env python
"""Resample the clean 1m partition into higher-timeframe intervals.

Reads the 1m data from the partitioned store and writes each derived interval
back as a first-class partition so the sweep engine can treat all timeframes equally.

Usage:
  python scripts/resample_intervals.py
  python scripts/resample_intervals.py --intervals 5m,15m,30m,60m
  python scripts/resample_intervals.py --intervals 3m,5m,15m,30m,60m --start 2020-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from banknifty_bot.config import load_config
from banknifty_bot.data.store import read_partitioned, write_partitioned
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[1]

RULE_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min", "60m": "60min", "1d": "1D",
}

SESSION_START = pd.Timedelta(hours=9, minutes=15)
SESSION_END = pd.Timedelta(hours=15, minutes=30)


def _session_mask(df: pd.DataFrame) -> pd.Series:
    """True for bars whose timestamp falls within 09:15–15:30 IST."""
    t = df.index.hour * 60 + df.index.minute
    start_min = 9 * 60 + 15
    end_min = 15 * 60 + 30
    return pd.Series((t >= start_min) & (t <= end_min), index=df.index)


def resample_one(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    mask = _session_mask(resampled)
    return resampled[mask]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", default="3m,5m,15m,30m,60m")
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        search_cfg = REPO / "config" / "config_search.yaml"
        cfg_path = str(search_cfg) if search_cfg.exists() else str(REPO / "config" / "config_intraday.yaml")

    cfg = load_config(cfg_path)
    start = args.start or cfg.data.start
    end = args.end or cfg.data.end
    symbol = cfg.data.symbol
    intervals = [i.strip() for i in args.intervals.split(",")]

    unknown = [iv for iv in intervals if iv not in RULE_MAP]
    if unknown:
        log.error("Unknown interval(s): %s. Supported: %s", unknown, list(RULE_MAP))
        return

    log.info("Loading 1m data for %s from %s to %s …", symbol, start, end)
    df_1m = read_partitioned(cfg.data.processed_dir, symbol, "1m", start, end)
    if df_1m.empty:
        log.error("No 1m data found. Run: make import-1m first.")
        return
    log.info("Loaded %d 1m bars", len(df_1m))

    for interval in intervals:
        rule = RULE_MAP[interval]
        log.info("Resampling 1m → %s …", interval)
        resampled = resample_one(df_1m, rule)
        if resampled.empty:
            log.warning("No bars produced for %s — skipping.", interval)
            continue
        write_partitioned(resampled, cfg.data.processed_dir, symbol, interval)
        print(f"Resampled 1m → {interval}: {len(resampled):,} bars written")

    log.info("Done.")


if __name__ == "__main__":
    main()
