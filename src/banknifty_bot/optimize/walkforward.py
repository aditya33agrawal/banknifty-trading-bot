"""Walk-forward analysis: optimize parameters on in-sample data, judge on the
untouched out-of-sample window that follows, roll forward, repeat.

The headline result is the **stitched OOS** performance — params are always chosen
without seeing the data they're scored on, which is the honest test the plan demands
(§1, §7). Anchored (expanding) IS by default; set `anchored=False` for a rolling IS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from banknifty_bot.backtest.runner import run_backtest
from banknifty_bot.config import AppConfig
from banknifty_bot.evaluation.metrics import full_report
from banknifty_bot.strategies.registry import get_strategy
from banknifty_bot.utils.logging import get_logger

from .search import grid_from_space, score

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame                       # per-fold IS/OOS metrics + chosen params
    oos_equity: pd.DataFrame                  # stitched out-of-sample equity curve
    oos_trades: pd.DataFrame                  # all OOS trades concatenated
    oos_metrics: dict                         # full_report over the stitched OOS result
    best_params_overall: dict = field(default_factory=dict)


def _fold_bounds(index: pd.DatetimeIndex, n_folds: int, train_ratio: float) -> list[tuple]:
    """Yield (is_start, is_end, oos_end) timestamps for each fold.

    The timeline is split into `n_folds` equal OOS windows; each fold trains on the
    `train_ratio` fraction of history immediately preceding its OOS window.
    """
    days = index.normalize().unique()
    oos_size = len(days) // (n_folds + 1)
    bounds = []
    for k in range(1, n_folds + 1):
        oos_start_i = k * oos_size
        oos_end_i = min((k + 1) * oos_size, len(days)) - 1
        train_span = int(oos_size * train_ratio / (1 - train_ratio)) or oos_size
        is_start_i = max(0, oos_start_i - train_span)
        bounds.append((days[is_start_i], days[oos_start_i - 1], days[oos_end_i]))
    return bounds


def walk_forward(
    cfg: AppConfig,
    df: pd.DataFrame,
    strategy_name: str,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    objective: str = "calmar",
    max_combos: int | None = 200,
    min_trades: int = 5,
    progress=None,
) -> WalkForwardResult:
    space = get_strategy(strategy_name, {}).param_space
    grid = grid_from_space(space, max_combos=max_combos, seed=cfg.run.seed)
    bounds = _fold_bounds(df.index, n_folds, train_ratio)

    log.info("  %s | grid=%d combos  folds=%d  bars=%d",
             strategy_name, len(grid), len(bounds), len(df))

    fold_rows: list[dict] = []
    oos_equity_parts: list[pd.DataFrame] = []
    oos_trade_parts: list[pd.DataFrame] = []
    last_equity = cfg.risk.initial_equity
    cell_start = time.perf_counter()

    for i, (is_start, is_end, oos_end) in enumerate(bounds):
        fold_start = time.perf_counter()
        is_df = df.loc[is_start:is_end]
        oos_df = df.loc[is_end:oos_end].iloc[1:]  # exclude the IS boundary bar
        if is_df.empty or oos_df.empty:
            log.warning("  fold %d/%d — empty IS or OOS slice, skipping", i + 1, len(bounds))
            continue

        log.info("  fold %d/%d  IS %s→%s (%d bars)  OOS →%s (%d bars)  scanning %d combos …",
                 i + 1, len(bounds),
                 is_start.strftime("%Y-%m-%d"), is_end.strftime("%Y-%m-%d"), len(is_df),
                 oos_end.strftime("%Y-%m-%d"), len(oos_df),
                 len(grid))

        # Optimize on in-sample.
        best_params, best_score = None, float("-inf")
        cleared = 0
        for j, params in enumerate(grid):
            m = run_backtest(cfg, is_df, strategy_name, params).metrics
            if m.get("n_trades", 0) < min_trades:
                continue
            cleared += 1
            s = score(m, objective)
            if s > best_score:
                best_score, best_params = s, params
        if best_params is None:                      # nothing cleared the bar this fold
            best_params = grid[0]
            log.warning("  fold %d/%d — no combo met min_trades=%d; using grid[0]", i + 1, len(bounds), min_trades)
        else:
            log.info("  fold %d/%d — IS scan done: %d/%d combos cleared, best %s=%.3f  params=%s",
                     i + 1, len(bounds), cleared, len(grid), objective, best_score, best_params)

        # Evaluate the chosen params out-of-sample.
        oos = run_backtest(cfg, oos_df, strategy_name, best_params)
        is_m = run_backtest(cfg, is_df, strategy_name, best_params).metrics

        fold_elapsed = time.perf_counter() - fold_start
        log.info("  fold %d/%d — OOS  Calmar=%.2f  Sharpe=%.2f  trades=%d  MaxDD=%.1f%%  (%.1fs)",
                 i + 1, len(bounds),
                 oos.metrics.get("calmar", 0), oos.metrics.get("sharpe", 0),
                 oos.metrics.get("n_trades", 0), oos.metrics.get("max_drawdown_pct", 0),
                 fold_elapsed)

        fold_rows.append({
            "fold": i + 1,
            "IS_start": is_start.date(), "OOS_start": oos_df.index[0].date(), "OOS_end": oos_end.date(),
            "IS_Calmar": round(is_m.get("calmar", 0), 2),
            "OOS_trades": oos.metrics.get("n_trades", 0),
            "OOS_CAGR%": round(oos.metrics.get("cagr_pct", 0), 1),
            "OOS_Sharpe": round(oos.metrics.get("sharpe", 0), 2),
            "OOS_Calmar": round(oos.metrics.get("calmar", 0), 2),
            "OOS_MaxDD%": round(oos.metrics.get("max_drawdown_pct", 0), 1),
            "OOS_PF": round(oos.metrics.get("profit_factor", 0), 2),
            "params": best_params,
        })

        # Chain the OOS equity so the stitched curve compounds across folds.
        if not oos.equity_df.empty:
            scaled = oos.equity_df / cfg.risk.initial_equity * last_equity
            oos_equity_parts.append(scaled)
            last_equity = float(scaled["equity"].iloc[-1])
        if not oos.trades_df.empty:
            oos_trade_parts.append(oos.trades_df)
        if progress is not None:
            progress((i + 1) / len(bounds))

    oos_equity = pd.concat(oos_equity_parts) if oos_equity_parts else pd.DataFrame(columns=["equity"])
    oos_trades = pd.concat(oos_trade_parts, ignore_index=True) if oos_trade_parts else pd.DataFrame()
    oos_metrics = full_report(oos_equity, oos_trades) if not oos_equity.empty else {}

    folds_df = pd.DataFrame(fold_rows)
    best_overall = {}
    if fold_rows:
        keyed = pd.Series([str(r["params"]) for r in fold_rows])
        winner = keyed.mode().iloc[0]
        best_overall = next(r["params"] for r in fold_rows if str(r["params"]) == winner)

    total_elapsed = time.perf_counter() - cell_start
    log.info("  %s DONE — OOS Calmar=%.2f  Sharpe=%.2f  trades=%d  elapsed=%.1fs  best_params=%s",
             strategy_name,
             oos_metrics.get("calmar", 0), oos_metrics.get("sharpe", 0),
             oos_metrics.get("n_trades", 0), total_elapsed, best_overall)

    return WalkForwardResult(folds_df, oos_equity, oos_trades, oos_metrics, best_overall)
