#!/usr/bin/env python
"""Idempotent incremental fetch: pulls OHLCV from the configured provider,
cleans it, and appends it to the partitioned parquet store.

Usage: python scripts/fetch_data.py --config config/config.yaml
"""
from __future__ import annotations

import argparse

from banknifty_bot.config import load_config
from banknifty_bot.data.cleaner import clean_ohlcv
from banknifty_bot.data.providers import PROVIDERS
from banknifty_bot.data.store import write_partitioned
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    provider_cls = PROVIDERS[cfg.data.provider]
    provider = provider_cls() if cfg.data.provider == "yfinance" else provider_cls(cfg.data.raw_dir)

    log.info(
        "Fetching %s %s [%s, %s] via %s",
        cfg.data.symbol, cfg.data.interval, cfg.data.start, cfg.data.end, cfg.data.provider,
    )
    raw = provider.get_ohlcv(cfg.data.symbol, cfg.data.start, cfg.data.end, cfg.data.interval)
    if raw.empty:
        log.warning("Provider returned no data.")
        return

    cleaned = clean_ohlcv(raw, cfg.session.start, cfg.session.end)
    manifest = write_partitioned(cleaned, cfg.data.processed_dir, cfg.data.symbol, cfg.data.interval)
    log.info("Wrote %d rows. Coverage: %s", len(cleaned), manifest)


if __name__ == "__main__":
    main()
