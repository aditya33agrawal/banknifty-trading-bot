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
    --out outputs/sweep_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from banknifty_bot.config import is_intraday_interval, load_config
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.optimize.robustness import monte_carlo_bootstrap
from banknifty_bot.optimize.walkforward import walk_forward
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[1]

# Strategies that require intraday bars (need VWAP / opening-range mechanics).
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
    args = parser.parse_args()

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

    # Load existing results for resumability.
    existing = _load_results(out_path)
    done = {(r["interval"], r["strategy"]) for r in existing}
    results = list(existing)

    cells = [(iv, st) for iv in intervals for st in strategies]
    total = len(cells)
    log.info("Sweep: %d intervals × %d strategies = %d cells. Already done: %d.",
             len(intervals), len(strategies), total, len(done))

    for idx, (interval, strategy) in enumerate(cells, 1):
        tag = f"[{idx}/{total}] {interval} × {strategy}"

        if (interval, strategy) in done:
            log.info("%s — skipping (already in results)", tag)
            continue

        # Skip intraday-only strategies on daily bars.
        if strategy in INTRADAY_ONLY and not is_intraday_interval(interval):
            log.info("%s — skipping (strategy is intraday-only)", tag)
            continue

        # Load data for this interval.
        cfg.data.interval = interval
        df = read_partitioned(
            cfg.data.processed_dir, cfg.data.symbol, interval,
            cfg.data.start, cfg.data.end,
        )

        if len(df) < 100:
            log.warning("%s — not enough data (%d rows), skipping.", tag, len(df))
            continue

        log.info("%s — %d bars. Running walk-forward (folds=%d, max_combos=%d) …",
                 tag, len(df), args.folds, args.max_combos)

        try:
            wf = walk_forward(
                cfg, df, strategy,
                n_folds=args.folds,
                max_combos=args.max_combos,
                objective=args.objective,
            )
        except Exception as exc:
            log.error("%s — walk_forward failed: %s", tag, exc)
            continue

        om = wf.oos_metrics
        mc: dict = {}
        if not wf.oos_trades.empty:
            try:
                mc = monte_carlo_bootstrap(wf.oos_trades, cfg.risk.initial_equity)
            except Exception as exc:
                log.warning("%s — monte_carlo failed: %s", tag, exc)

        gates = gate_check(om, cfg.promotion_gates)

        row: dict = {
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
            "gates_passed": gates,
            "mc_prob_profit": round(mc.get("prob_profit", 0), 3),
            "mc_p5_return_pct": round(mc.get("p5_return_pct", 0), 2),
            "mc_median_return_pct": round(mc.get("median_return_pct", 0), 2),
            "best_params": wf.best_params_overall,
            "data_start": str(df.index.min().date()),
            "data_end": str(df.index.max().date()),
            "data_bars": len(df),
        }

        results.append(row)
        _save_results(out_path, results)

        print(f"{tag} → Calmar={row['oos_calmar']:.2f} Sharpe={row['oos_sharpe']:.2f} "
              f"gates={gates}/5 trades={row['oos_n_trades']}")

    log.info("Sweep complete. %d results written to %s", len(results), out_path)


if __name__ == "__main__":
    main()
