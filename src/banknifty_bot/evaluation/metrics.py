from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    daily_equity = equity_df["equity"].resample("1D").last().dropna()
    return daily_equity.pct_change().dropna()


def sharpe_ratio(equity_df: pd.DataFrame, risk_free: float = 0.0) -> float:
    rets = _daily_returns(equity_df)
    if rets.std() == 0 or len(rets) < 2:
        return 0.0
    excess = rets - risk_free / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(equity_df: pd.DataFrame, risk_free: float = 0.0) -> float:
    rets = _daily_returns(equity_df)
    downside = rets[rets < 0]
    if downside.std() == 0 or len(rets) < 2:
        return 0.0
    excess = rets.mean() - risk_free / TRADING_DAYS_PER_YEAR
    return float(excess / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_df: pd.DataFrame) -> dict:
    equity = equity_df["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() if len(drawdown) else 0.0

    trough_idx = drawdown.idxmin() if len(drawdown) else None
    duration = None
    if trough_idx is not None:
        peak_idx = equity.loc[:trough_idx].idxmax()
        duration = (trough_idx - peak_idx).total_seconds() / 86400

    return {"max_drawdown_pct": float(max_dd * 100), "duration_days": duration}


def calmar_ratio(equity_df: pd.DataFrame, cagr: float) -> float:
    dd = max_drawdown(equity_df)["max_drawdown_pct"]
    if dd == 0:
        return 0.0
    return float(cagr / abs(dd))


def cagr(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    start, end = equity_df["equity"].iloc[0], equity_df["equity"].iloc[-1]
    days = (equity_df.index[-1] - equity_df.index[0]).total_seconds() / 86400
    years = days / 365.25
    if years <= 0 or start <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1) * 100


def trade_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "n_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "payoff_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_consecutive_losses": 0,
        }

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = wins["net_pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0.0
    win_rate = len(wins) / len(trades_df)
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    is_loss = (trades_df["net_pnl"] <= 0).astype(int)
    max_consec_losses = int(
        (is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)).max()
    ) if len(is_loss) else 0

    return {
        "n_trades": len(trades_df),
        "win_rate_pct": float(win_rate * 100),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "payoff_ratio": float(payoff),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "max_consecutive_losses": max_consec_losses,
    }


def cost_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"total_cost": 0.0, "cost_pct_of_gross": 0.0, "cost_per_trade": 0.0}

    total_cost = trades_df["cost"].sum()
    gross = trades_df["gross_pnl"].sum()
    cost_pct = (total_cost / gross * 100) if gross != 0 else float("inf")
    return {
        "total_cost": float(total_cost),
        "cost_pct_of_gross": float(cost_pct),
        "cost_per_trade": float(total_cost / len(trades_df)),
    }


def full_report(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    c = cagr(equity_df)
    return {
        "cagr_pct": c,
        "sharpe": sharpe_ratio(equity_df),
        "sortino": sortino_ratio(equity_df),
        "calmar": calmar_ratio(equity_df, c),
        **max_drawdown(equity_df),
        **trade_stats(trades_df),
        **cost_stats(trades_df),
    }
