#!/usr/bin/env python
"""Run the paper-trading loop using the same Strategy + CostModel as backtesting.

Usage: python scripts/paper_trade.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.risk import RiskConfig
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.config import load_config, load_yaml
from banknifty_bot.data.providers import PROVIDERS
from banknifty_bot.paper.live_loop import LiveLoop
from banknifty_bot.strategies.registry import get_strategy
from banknifty_bot.utils.seeds import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    provider_cls = PROVIDERS[cfg.data.provider]
    provider = provider_cls() if cfg.data.provider == "yfinance" else provider_cls(cfg.data.raw_dir)

    strategy_params = load_yaml(cfg.strategy.params_file)
    strategy = get_strategy(cfg.strategy.name, strategy_params)

    cost_model = CostModel.from_yaml(cfg.backtest.costs_file, cfg.execution_proxy.instrument)
    slippage = Slippage(cfg.backtest.slippage_model, cfg.backtest.slippage_value)
    risk_cfg = RiskConfig(
        initial_equity=cfg.risk.initial_equity,
        risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
        daily_max_loss_pct=cfg.risk.daily_max_loss_pct,
        max_trades_per_day=cfg.risk.max_trades_per_day,
        max_open_positions=cfg.risk.max_open_positions,
        lot_size=cfg.execution_proxy.lot_size,
        contract_multiplier=cfg.execution_proxy.contract_multiplier,
        square_off=dt.time.fromisoformat(cfg.session.square_off),
        no_trade_first_minutes=cfg.session.no_trade_first_minutes,
        no_trade_last_minutes=cfg.session.no_trade_last_minutes,
    )

    loop = LiveLoop(
        provider, strategy, cfg.data.symbol, cfg.data.interval,
        risk_cfg, cost_model, slippage, poll_seconds=args.poll_seconds,
    )
    loop.run_forever()


if __name__ == "__main__":
    main()
