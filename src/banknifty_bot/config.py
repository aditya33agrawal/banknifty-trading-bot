from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "default_run"
    seed: int = 42
    output_dir: str = "outputs"


class DataConfig(BaseModel):
    symbol: str = "^NSEBANK"
    provider: Literal["yfinance", "csv"] = "yfinance"
    interval: str = "5m"
    start: str
    end: str
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    timezone: str = "Asia/Kolkata"


class SessionConfig(BaseModel):
    start: str = "09:15"
    end: str = "15:30"
    square_off: str = "15:15"
    no_trade_first_minutes: int = 5
    no_trade_last_minutes: int = 10


class ExecutionProxyConfig(BaseModel):
    instrument: Literal["futures", "options"] = "futures"
    lot_size: int = 15
    contract_multiplier: float = 1.0


class RiskConfig(BaseModel):
    initial_equity: float = 1_000_000.0
    risk_per_trade_pct: float = 1.0
    daily_max_loss_pct: float = 3.0
    max_trades_per_day: int = 4
    max_open_positions: int = 1


class StrategyConfig(BaseModel):
    name: str = "orb"
    params_file: str = "config/strategies/orb.yaml"


class BacktestConfig(BaseModel):
    costs_file: str = "config/costs.yaml"
    slippage_model: Literal["ticks", "pct", "atr_fraction"] = "ticks"
    slippage_value: float = 1.0
    slippage_stress_multipliers: list[float] = Field(default_factory=lambda: [1, 2, 4])


class PromotionGatesConfig(BaseModel):
    min_sharpe_oos: float = 1.0
    min_calmar_oos: float = 0.5
    min_profit_factor: float = 1.3
    max_drawdown_pct: float = 20.0
    max_cost_pct_of_gross: float = 40.0


class AppConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig
    session: SessionConfig = Field(default_factory=SessionConfig)
    execution_proxy: ExecutionProxyConfig = Field(default_factory=ExecutionProxyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    promotion_gates: PromotionGatesConfig = Field(default_factory=PromotionGatesConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(raw)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
