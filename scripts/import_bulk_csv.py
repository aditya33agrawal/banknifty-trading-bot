#!/usr/bin/env python
"""Import a third-party bulk intraday CSV into the partitioned parquet store.

Handles the common Indian-intraday CSV shapes automatically (combined datetime OR
separate Date/Time columns; full or short OHLC names; optional volume).

Examples:
  # Auto-detect everything (combined datetime or Date+Time columns):
  python scripts/import_bulk_csv.py --file data/raw/banknifty_1min.csv --interval 1m

  # Force mappings for an awkward schema:
  python scripts/import_bulk_csv.py --file feed.csv --interval 1m \
      --date-col Date --time-col Time --col-map close=ltp,volume=qty --dayfirst

Where to get multi-year BankNifty intraday data (free):
  • Kaggle: search "Nifty Bank 1 minute" / "BankNifty intraday" datasets
  • GitHub: public repos dumping NSE 1-min OHLC (e.g. *_1min.csv files)
  • Vendors (paid, cleanest): firstratedata / truedata / GDFL
Download the CSV, then point this script at it.
"""
from __future__ import annotations

import argparse

from banknifty_bot.config import is_intraday_interval, load_config
from banknifty_bot.data.bulk_import import load_bulk_csv
from banknifty_bot.data.cleaner import clean_ohlcv
from banknifty_bot.data.store import write_partitioned
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)


def _parse_col_map(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    return dict(pair.split("=", 1) for pair in s.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the bulk CSV.")
    parser.add_argument("--config", default="config/config_intraday.yaml")
    parser.add_argument("--symbol", default=None, help="Store symbol key (default from config).")
    parser.add_argument("--interval", default=None, help="e.g. 1m, 5m (default from config).")
    parser.add_argument("--datetime-col", default=None)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--col-map", default=None, help='e.g. "close=ltp,volume=qty"')
    parser.add_argument("--dayfirst", action="store_true", help="Dates are DD-MM-YYYY.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = args.symbol or cfg.data.symbol
    interval = args.interval or cfg.data.interval

    log.info("Loading %s as %s %s …", args.file, symbol, interval)
    raw = load_bulk_csv(
        args.file, datetime_col=args.datetime_col, date_col=args.date_col,
        time_col=args.time_col, col_map=_parse_col_map(args.col_map),
        tz=cfg.data.timezone, dayfirst=args.dayfirst,
    )
    log.info("Parsed %d rows [%s -> %s]", len(raw), raw.index.min(), raw.index.max())

    cleaned = clean_ohlcv(
        raw, cfg.session.start, cfg.session.end, intraday=is_intraday_interval(interval)
    )
    manifest = write_partitioned(cleaned, cfg.data.processed_dir, symbol, interval)
    log.info("Imported %d cleaned rows into the store. Coverage: %s", len(cleaned), manifest)


if __name__ == "__main__":
    main()
