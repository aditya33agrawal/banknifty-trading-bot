"""Portfolio ensemble — diversify across strategies' *return streams*.

Unlike the voting `Ensemble` strategy (which forces several signals onto one shared
position and tends to dilute the strongest edge), this runs each strategy
independently with its own capital sleeve and combines their daily returns. When the
return streams are weakly correlated, the blended equity has a higher Sharpe/Calmar
than any single member — the actual diversification benefit (plan §B.4 "ensemble").
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from banknifty_bot.config import AppConfig
from banknifty_bot.evaluation.metrics import full_report

from .runner import run_backtest


@dataclass
class EnsembleResult:
    equity_df: pd.DataFrame                 # blended portfolio equity
    trades_df: pd.DataFrame                 # all members' trades concatenated (tagged)
    metrics: dict                           # full_report over the blended equity
    members: pd.DataFrame                   # per-member metrics
    corr: pd.DataFrame                      # correlation matrix of members' daily returns


def _daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    return equity_df["equity"].resample("1D").last().dropna().pct_change().dropna()


def combine_strategies(
    cfg: AppConfig,
    df: pd.DataFrame,
    specs: list[tuple[str, dict]],
    weights: list[float] | None = None,
) -> EnsembleResult:
    """Run each (strategy_name, params) on `df` and blend their daily returns.

    `weights` default to equal capital across members (they are normalised to sum 1).
    """
    n = len(specs)
    weights = weights or [1.0 / n] * n
    total = sum(weights)
    weights = [w / total for w in weights]

    returns_by_member: dict[str, pd.Series] = {}
    member_rows, trade_parts = [], []

    for (name, params), w in zip(specs, weights):
        res = run_backtest(cfg, df, name, params)
        if res.equity_df.empty:
            continue
        returns_by_member[name] = _daily_returns(res.equity_df)
        m = res.metrics
        member_rows.append({
            "strategy": name, "weight": round(w, 3), "trades": m.get("n_trades", 0),
            "CAGR%": round(m.get("cagr_pct", 0), 1), "Sharpe": round(m.get("sharpe", 0), 2),
            "Calmar": round(m.get("calmar", 0), 2), "MaxDD%": round(m.get("max_drawdown_pct", 0), 1),
            "PF": round(m.get("profit_factor", 0), 2),
        })
        if not res.trades_df.empty:
            tagged = res.trades_df.copy()
            tagged["strategy"] = name
            trade_parts.append(tagged)

    if not returns_by_member:
        empty = pd.DataFrame(columns=["equity"])
        return EnsembleResult(empty, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame())

    ret_df = pd.DataFrame(returns_by_member).fillna(0.0)
    spec_names = [name for name, _ in specs if name in returns_by_member]
    w_map = {name: w for (name, _), w in zip(specs, weights)}
    blended_ret = sum(ret_df[name] * w_map[name] for name in spec_names)

    equity = (1 + blended_ret).cumprod() * cfg.risk.initial_equity
    equity_df = equity.to_frame("equity")

    trades_df = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    metrics = full_report(equity_df, trades_df) if not equity_df.empty else {}

    return EnsembleResult(
        equity_df=equity_df,
        trades_df=trades_df,
        metrics=metrics,
        members=pd.DataFrame(member_rows),
        corr=ret_df.corr().round(2),
    )
