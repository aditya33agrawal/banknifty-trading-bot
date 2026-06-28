#!/usr/bin/env python
"""Matrix sweep: find the best (interval × strategy × params) combination.

For each cell the sweep runs an honest walk-forward optimization (out-of-sample
params) plus a Monte Carlo bootstrap, then streams results to a JSON leaderboard.
The file is written after each cell so a long run survives interruption.

Usage:
  python scripts/run_sweep.py \
    --intervals 5m,15m,30m,60m,1d \
    --strategies ema_trend,supertrend,rsi_reversion,donchian,orb,vwap \
    --folds 5 --max-combos 60 --objective calmar \
    --workers auto \
    --out outputs/sweep_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from banknifty_bot.config import is_intraday_interval, load_config
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.optimize.robustness import monte_carlo_bootstrap
from banknifty_bot.optimize.walkforward import walk_forward
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[1]

INTRADAY_ONLY = {"orb", "vwap"}


def gate_check(metrics: dict, gates) -> int:
    checks = [
        metrics.get("sharpe", 0) >= gates.min_sharpe_oos,
        metrics.get("calmar", 0) >= gates.min_calmar_oos,
        metrics.get("profit_factor", 0) >= gates.min_profit_factor,
        abs(metrics.get("max_drawdown_pct", 100)) <= gates.max_drawdown_pct,
        metrics.get("cost_pct_of_gross", 100) <= gates.max_cost_pct_of_gross,
    ]
    return sum(checks)


def _load_results(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    return []


def _save_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str))


def _run_cell(args: tuple) -> dict | None:
    """Run one (interval × strategy) cell. Top-level function required for pickling."""
    cfg, df, interval, strategy, folds, max_combos, objective = args
    t0 = time.perf_counter()
    log.info(">>> START  %s × %s  (%d bars)", interval, strategy, len(df))
    try:
        wf = walk_forward(
            cfg, df, strategy,
            n_folds=folds,
            max_combos=max_combos,
            objective=objective,
        )
    except Exception as exc:
        log.error(">>> FAIL   %s × %s — walk_forward failed: %s", interval, strategy, exc)
        return None

    om = wf.oos_metrics
    mc: dict = {}
    if not wf.oos_trades.empty:
        try:
            mc = monte_carlo_bootstrap(wf.oos_trades, cfg.risk.initial_equity)
        except Exception as exc:
            log.warning("%s × %s — monte_carlo failed: %s", interval, strategy, exc)

    gates_val = gate_check(om, cfg.promotion_gates)
    elapsed = time.perf_counter() - t0
    log.info(">>> DONE   %s × %s  Calmar=%.2f  Sharpe=%.2f  gates=%d/5  trades=%d  (%.1fs)",
             interval, strategy,
             om.get("calmar", 0), om.get("sharpe", 0),
             gates_val, om.get("n_trades", 0), elapsed)

    return {
        "interval": interval,
        "strategy": strategy,
        "oos_cagr_pct": round(om.get("cagr_pct", 0), 2),
        "oos_sharpe": round(om.get("sharpe", 0), 2),
        "oos_calmar": round(om.get("calmar", 0), 2),
        "oos_sortino": round(om.get("sortino", 0), 2),
        "oos_max_dd_pct": round(om.get("max_drawdown_pct", 0), 2),
        "oos_profit_factor": round(om.get("profit_factor", 0), 2),
        "oos_n_trades": om.get("n_trades", 0),
        "cost_pct_of_gross": round(om.get("cost_pct_of_gross", 0), 2),
        "gates_passed": gates_val,
        "mc_prob_profit": round(mc.get("prob_profit", 0), 3),
        "mc_p5_return_pct": round(mc.get("p5_return_pct", 0), 2),
        "mc_median_return_pct": round(mc.get("median_return_pct", 0), 2),
        "best_params": wf.best_params_overall,
        "data_start": str(df.index.min().date()),
        "data_end": str(df.index.max().date()),
        "data_bars": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", default="5m,15m,30m,60m,1d")
    parser.add_argument("--strategies", default="ema_trend,supertrend,rsi_reversion,donchian,orb,vwap")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-combos", type=int, default=60)
    parser.add_argument("--objective", default="calmar", choices=["calmar", "sortino", "sharpe", "profit_factor"])
    parser.add_argument("--out", default="outputs/sweep_results.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--slippage", type=float, default=1.0)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--workers", default="auto",
                        help="Parallel workers: 'auto' uses all CPU cores, or an integer (default: auto)")
    args = parser.parse_args()

    if args.workers == "auto":
        n_workers = os.cpu_count() or 1
    else:
        n_workers = int(args.workers)

    cfg_path = args.config
    if cfg_path is None:
        search_cfg = REPO / "config" / "config_search.yaml"
        cfg_path = str(search_cfg) if search_cfg.exists() else str(REPO / "config" / "config_intraday.yaml")

    cfg = load_config(cfg_path)
    cfg.backtest.slippage_value = args.slippage
    cfg.risk.risk_per_trade_pct = args.risk_pct

    intervals = [i.strip() for i in args.intervals.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    out_path = REPO / args.out

    existing = _load_results(out_path)
    done = {(r["interval"], r["strategy"]) for r in existing}
    results = list(existing)

    cells = [(iv, st) for iv in intervals for st in strategies]
    total = len(cells)
    log.info("Sweep: %d intervals × %d strategies = %d cells. Already done: %d. Workers: %d.",
             len(intervals), len(strategies), total, len(done), n_workers)

    # Build task list — skip already-done and intraday-only on daily bars.
    pending: list[tuple] = []
    skipped: list[tuple[str, str]] = []
    for interval, strategy in cells:
        if (interval, strategy) in done:
            log.info("[skip] %s × %s — already in results", interval, strategy)
            continue
        if strategy in INTRADAY_ONLY and not is_intraday_interval(interval):
            log.info("[skip] %s × %s — strategy is intraday-only", interval, strategy)
            skipped.append((interval, strategy))
            continue
        pending.append((interval, strategy))

    # Pre-load data per interval (shared across strategies for the same interval).
    # On Linux (Colab) fork semantics mean child processes read from shared pages.
    interval_dfs: dict[str, object] = {}
    for interval, _ in pending:
        if interval not in interval_dfs:
            cfg.data.interval = interval
            df = read_partitioned(
                cfg.data.processed_dir, cfg.data.symbol, interval,
                cfg.data.start, cfg.data.end,
            )
            interval_dfs[interval] = df
            log.info("Loaded %s: %d bars", interval, len(df))

    # Build args tuples for each pending cell.
    task_args = []
    for interval, strategy in pending:
        df = interval_dfs[interval]
        if len(df) < 100:
            log.warning("Skip %s × %s — not enough data (%d rows)", interval, strategy, len(df))
            continue
        task_args.append((cfg, df, interval, strategy, args.folds, args.max_combos, args.objective))

    sweep_start = time.perf_counter()

    def _print_result(tag: str, row: dict, completed: int, total: int) -> None:
        elapsed = time.perf_counter() - sweep_start
        avg = elapsed / completed
        eta_s = avg * (total - completed)
        eta_min = eta_s / 60
        print(
            f"{tag} → Calmar={row['oos_calmar']:.2f}  Sharpe={row['oos_sharpe']:.2f}  "
            f"gates={row['gates_passed']}/5  trades={row['oos_n_trades']}  "
            f"| cell {completed}/{total}  elapsed={elapsed/60:.1f}m  ETA≈{eta_min:.1f}m",
            flush=True,
        )

    if n_workers == 1:
        # Single-process path: preserves streaming output and avoids fork overhead.
        for i, task in enumerate(task_args, 1):
            interval, strategy = task[2], task[3]
            tag = f"[{i}/{len(task_args)}] {interval} × {strategy}"
            log.info("%s — running …", tag)
            row = _run_cell(task)
            if row:
                results.append(row)
                _save_results(out_path, results)
                _print_result(tag, row, i, len(task_args))
            else:
                log.warning("%s — no result", tag)
    else:
        # Parallel path: submit all cells, write results as each completes.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_tag = {pool.submit(_run_cell, t): (t[2], t[3]) for t in task_args}
            completed = 0
            for future in as_completed(future_to_tag):
                interval, strategy = future_to_tag[future]
                completed += 1
                tag = f"[{completed}/{len(task_args)}] {interval} × {strategy}"
                try:
                    row = future.result()
                except Exception as exc:
                    log.error("%s — exception: %s", tag, exc)
                    row = None
                if row:
                    results.append(row)
                    _save_results(out_path, results)
                    _print_result(tag, row, completed, len(task_args))
                else:
                    log.warning("%s — no result", tag)

    log.info("Sweep complete. %d results written to %s", len(results), out_path)


if __name__ == "__main__":
    main()
