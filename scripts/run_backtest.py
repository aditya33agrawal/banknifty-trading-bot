#!/usr/bin/env python
"""Run a single strategy backtest end-to-end against the processed data store.

Usage: python scripts/run_backtest.py --config config/config.yaml
"""
from __future__ import annotations

import argparse

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.engine import BacktestEngine
from banknifty_bot.backtest.runner import risk_config_from, run_backtest
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.config import load_config, load_yaml
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.strategies.registry import get_strategy
from banknifty_bot.utils.logging import get_logger
from banknifty_bot.utils.seeds import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    df = read_partitioned(
        cfg.data.processed_dir, cfg.data.symbol, cfg.data.interval, cfg.data.start, cfg.data.end
    )
    if df.empty:
        log.error("No processed data found. Run scripts/fetch_data.py first.")
        return
    log.info("Loaded %d bars [%s -> %s]", len(df), df.index.min(), df.index.max())

    strategy_params = load_yaml(cfg.strategy.params_file)
    result = run_backtest(cfg, df, cfg.strategy.name, strategy_params)

    trades = result.trades_df
    log.info("Mode: %s | Trades: %d | Final equity: %.2f",
             "intraday" if result.intraday else "swing", len(trades), result.portfolio.equity)
    if not trades.empty:
        log.info("Net P&L: %.2f | Total cost: %.2f", trades["net_pnl"].sum(), trades["cost"].sum())
        m = result.metrics
        log.info("Sharpe: %.2f | Calmar: %.2f | MaxDD: %.1f%% | PF: %.2f | Win%%: %.1f",
                 m.get("sharpe", 0), m.get("calmar", 0), m.get("max_drawdown_pct", 0),
                 m.get("profit_factor", 0), m.get("win_rate_pct", 0))

    # Slippage-stress sweep (fragility check) — re-run the same signals at higher slippage.
    cost_model = CostModel.from_yaml(cfg.backtest.costs_file, cfg.execution_proxy.instrument)
    risk_cfg = risk_config_from(cfg)
    signals = get_strategy(cfg.strategy.name, strategy_params).generate_signals(df)
    for mult in cfg.backtest.slippage_stress_multipliers:
        slippage = Slippage(cfg.backtest.slippage_model, cfg.backtest.slippage_value)
        stressed_engine = BacktestEngine(risk_cfg, cost_model, slippage.stressed(mult))
        stressed_trades = stressed_engine.run(df, signals).to_trades_df()
        net = stressed_trades["net_pnl"].sum() if not stressed_trades.empty else 0.0
        log.info("Slippage x%s -> net P&L: %.2f", mult, net)


if __name__ == "__main__":
    main()
