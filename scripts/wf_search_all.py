#!/usr/bin/env python
"""Broad walk-forward search across every swing-compatible strategy.

For each strategy: optimize parameters in-sample, evaluate out-of-sample (per the
config's filters/exits), then rank by stitched-OOS Calmar and count promotion gates
cleared. The point is to find a config that survives OOS — not the best in-sample run.

Usage:
  python scripts/wf_search_all.py --config config/config_daily.yaml --folds 5 --max-combos 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from banknifty_bot.config import load_config
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.optimize.robustness import monte_carlo_bootstrap
from banknifty_bot.optimize.walkforward import walk_forward
from banknifty_bot.strategies.registry import REGISTRY, SWING_COMPATIBLE
from banknifty_bot.utils.logging import get_logger
from banknifty_bot.utils.seeds import set_seed

log = get_logger(__name__)


def gates_passed(m: dict, g) -> int:
    if not m:
        return 0
    return sum([
        m.get("sharpe", 0) >= g.min_sharpe_oos,
        m.get("calmar", 0) >= g.min_calmar_oos,
        m.get("profit_factor", 0) >= g.min_profit_factor,
        abs(m.get("max_drawdown_pct", 100)) <= g.max_drawdown_pct,
        m.get("cost_pct_of_gross", 100) <= g.max_cost_pct_of_gross,
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_daily.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-combos", type=int, default=60)
    parser.add_argument("--objective", default="calmar")
    parser.add_argument("--swing-only", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)

    df = read_partitioned(
        cfg.data.processed_dir, cfg.data.symbol, cfg.data.interval, cfg.data.start, cfg.data.end
    )
    if df.empty:
        log.error("No data. Fetch first.")
        return

    strategies = sorted(SWING_COMPATIBLE) if args.swing_only else sorted(REGISTRY)
    log.info("WF search over %s on %d bars (%d folds, <=%d combos each), exits=%s",
             strategies, len(df), args.folds, args.max_combos, cfg.exits.model_dump())

    results = []
    for name in strategies:
        t0 = time.time()
        try:
            res = walk_forward(cfg, df, name, n_folds=args.folds,
                               objective=args.objective, max_combos=args.max_combos)
            m = res.oos_metrics
            mc = monte_carlo_bootstrap(res.oos_trades, cfg.risk.initial_equity) if not res.oos_trades.empty else {}
            row = {
                "strategy": name,
                "OOS_trades": m.get("n_trades", 0),
                "OOS_CAGR%": round(m.get("cagr_pct", 0), 1),
                "OOS_Sharpe": round(m.get("sharpe", 0), 2),
                "OOS_Calmar": round(m.get("calmar", 0), 2),
                "OOS_MaxDD%": round(m.get("max_drawdown_pct", 0), 1),
                "OOS_PF": round(m.get("profit_factor", 0), 2),
                "gates": f"{gates_passed(m, cfg.promotion_gates)}/5",
                "MC_p5%": round(mc.get("p5_return_pct", 0), 1) if mc else None,
                "MC_prob_profit": round(mc.get("prob_profit", 0), 2) if mc else None,
                "best_params": res.best_params_overall,
                "secs": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            row = {"strategy": name, "error": str(e)[:80], "secs": round(time.time() - t0, 1)}
        results.append(row)
        log.info("done %s in %ss -> %s", name, row["secs"],
                 {k: row.get(k) for k in ("OOS_Calmar", "OOS_PF", "gates")})

    results.sort(key=lambda r: r.get("OOS_Calmar", -999), reverse=True)

    print("\n=== Walk-forward leaderboard (out-of-sample, ranked by Calmar) ===")
    cols = ["strategy", "OOS_trades", "OOS_CAGR%", "OOS_Sharpe", "OOS_Calmar",
            "OOS_MaxDD%", "OOS_PF", "gates", "MC_prob_profit"]
    print(" | ".join(f"{c:>12s}" for c in cols))
    for r in results:
        if "error" in r:
            print(f"{r['strategy']:>12s} | ERROR: {r['error']}")
        else:
            print(" | ".join(f"{str(r.get(c, '')):>12s}" for c in cols))

    print("\nBest params per strategy:")
    for r in results:
        if "best_params" in r:
            print(f"  {r['strategy']:14s} {r['best_params']}")

    out = Path(cfg.run.output_dir) / "wf_search_all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
