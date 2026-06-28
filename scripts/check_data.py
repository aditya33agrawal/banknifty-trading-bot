#!/usr/bin/env python
"""Validate the imported 1m partition after import_bulk_csv.py.

Usage:
  python scripts/check_data.py
  python scripts/check_data.py --start 2015-01-09 --end 2026-04-22
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from banknifty_bot.config import load_config
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
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

    log.info("Loading 1m data for %s from %s to %s", symbol, start, end)
    df = read_partitioned(cfg.data.processed_dir, symbol, "1m", start, end)

    if df.empty:
        log.error("No 1m data found. Run: make import-1m")
        sys.exit(1)

    failures: list[str] = []

    # Monotonic check (deduplication already done in store)
    if not df.index.is_monotonic_increasing:
        failures.append("Timestamps are NOT monotonically increasing")

    # No NaN/zero OHLC
    for col in ["open", "high", "low", "close"]:
        if df[col].isna().any():
            failures.append(f"NaN values found in column '{col}'")
        if (df[col] == 0).any():
            failures.append(f"Zero values found in column '{col}'")

    # No weekend bars
    weekend_mask = df.index.weekday >= 5
    if weekend_mask.any():
        failures.append(f"Weekend bars found: {weekend_mask.sum()} rows")

    # Per-year coverage table
    print("\n--- Coverage table ---")
    print(f"{'Year':<6} {'Bars':>8} {'Trading days':>14} {'Avg bars/day':>14} {'Status':>8}")
    print("-" * 56)
    warned_days = []
    for year, ydf in df.groupby(df.index.year):
        trading_days = ydf.index.normalize().nunique()
        avg_bars = len(ydf) / trading_days if trading_days else 0
        # Flag days with suspiciously low or high bar counts
        day_counts = ydf.groupby(ydf.index.normalize()).size()
        full_days = day_counts[day_counts > 50]  # ignore very short sessions at year edges
        low_days = full_days[full_days < 300]
        high_days = full_days[full_days > 400]
        status = "OK"
        if len(low_days):
            status = f"WARN({len(low_days)} thin)"
            warned_days.extend(low_days.index.tolist())
        if len(high_days):
            status += f"+WARN({len(high_days)} fat)"
            warned_days.extend(high_days.index.tolist())
        print(f"{year:<6} {len(ydf):>8,} {trading_days:>14} {avg_bars:>14.1f} {status:>8}")

    print("-" * 56)
    print(f"{'TOTAL':<6} {len(df):>8,} {df.index.normalize().nunique():>14}")
    print()

    if warned_days:
        log.warning("Days with unusual bar counts (first 10): %s", warned_days[:10])

    if failures:
        for f in failures:
            log.error("FAIL: %s", f)
        sys.exit(1)
    else:
        log.info("All checks passed. %d bars from %s to %s.", len(df), df.index.min().date(), df.index.max().date())
        sys.exit(0)


if __name__ == "__main__":
    main()
