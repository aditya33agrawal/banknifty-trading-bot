"""High-level backtest orchestration shared by the CLI and the Streamlit dashboard.

Both entry points call `run_backtest` so what you see in the dashboard is exactly
what the scripts produce — no logic drift.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from banknifty_bot.config import AppConfig, is_intraday_interval
from banknifty_bot.evaluation.metrics import full_report
from banknifty_bot.features.filters import FilterConfig, apply_filters
from banknifty_bot.strategies.registry import get_strategy

from .costs import CostModel
from .engine import BacktestEngine
from .exits import ExitConfig
from .portfolio import Portfolio
from .risk import RiskConfig
from .slippage import Slippage


@dataclass
class BacktestResult:
    portfolio: Portfolio
    equity_df: pd.DataFrame
    trades_df: pd.DataFrame
    metrics: dict
    intraday: bool


def risk_config_from(cfg: AppConfig) -> RiskConfig:
    return RiskConfig(
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
        intraday=is_intraday_interval(cfg.data.interval),
    )


def exit_config_from(cfg: AppConfig) -> ExitConfig:
    return ExitConfig(**cfg.exits.model_dump())


def filter_config_from(cfg: AppConfig) -> FilterConfig:
    return FilterConfig(**cfg.filters.model_dump())


def run_backtest(
    cfg: AppConfig,
    df: pd.DataFrame,
    strategy_name: str | None = None,
    strategy_params: dict | None = None,
) -> BacktestResult:
    """Run one strategy end-to-end against `df` using the config's cost/risk model.

    `strategy_name` / `strategy_params` override the config when supplied (the
    dashboard uses this for live parameter tuning).
    """
    name = strategy_name or cfg.strategy.name
    strategy = get_strategy(name, strategy_params or {})
    signals = strategy.generate_signals(df)
    signals = apply_filters(signals, df, filter_config_from(cfg))

    cost_model = CostModel.from_yaml(cfg.backtest.costs_file, cfg.execution_proxy.instrument)
    slippage = Slippage(cfg.backtest.slippage_model, cfg.backtest.slippage_value)
    risk_cfg = risk_config_from(cfg)

    engine = BacktestEngine(risk_cfg, cost_model, slippage, exit_cfg=exit_config_from(cfg))
    portfolio = engine.run(df, signals)

    equity_df = portfolio.to_equity_df()
    trades_df = portfolio.to_trades_df()
    metrics = full_report(equity_df, trades_df) if not equity_df.empty else {}

    return BacktestResult(
        portfolio=portfolio,
        equity_df=equity_df,
        trades_df=trades_df,
        metrics=metrics,
        intraday=risk_cfg.intraday,
    )
