#!/usr/bin/env python
"""Run a backtest and produce a self-contained HTML report (plan §8).

Usage: python scripts/generate_report.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.engine import BacktestEngine
from banknifty_bot.backtest.risk import RiskConfig
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.config import load_config, load_yaml
from banknifty_bot.data.cleaner import coverage_summary
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.evaluation.report import generate_html_report
from banknifty_bot.strategies.registry import get_strategy
from banknifty_bot.utils.logging import get_logger
from banknifty_bot.utils.seeds import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    df = read_partitioned(
        cfg.data.processed_dir, cfg.data.symbol, cfg.data.interval, cfg.data.start, cfg.data.end
    )
    if df.empty:
        log.error("No processed data found. Run scripts/fetch_data.py first.")
        return

    strategy_params = load_yaml(cfg.strategy.params_file)
    strategy = get_strategy(cfg.strategy.name, strategy_params)
    signals = strategy.generate_signals(df)

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

    engine = BacktestEngine(risk_cfg, cost_model, slippage)
    portfolio = engine.run(df, signals)

    run_meta = {
        "strategy": cfg.strategy.name,
        "params": strategy_params,
        "symbol": cfg.data.symbol,
        "interval": cfg.data.interval,
        "execution_proxy": cfg.execution_proxy.instrument,
        "data_coverage": coverage_summary(df),
        "slippage_model": f"{cfg.backtest.slippage_model}={cfg.backtest.slippage_value}",
    }

    output_path = args.output or f"{cfg.run.output_dir}/{cfg.run.name}_report.html"
    out = generate_html_report(portfolio.to_equity_df(), portfolio.to_trades_df(), run_meta, output_path)
    log.info("Report written to %s", out)


if __name__ == "__main__":
    main()
