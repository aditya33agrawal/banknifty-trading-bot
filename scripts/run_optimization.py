#!/usr/bin/env python
"""Walk-forward optimization for a strategy: tune on in-sample, judge out-of-sample.

Usage:
  python scripts/run_optimization.py --config config/config_daily.yaml --strategy donchian --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from banknifty_bot.config import load_config
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.optimize.robustness import monte_carlo_bootstrap
from banknifty_bot.optimize.walkforward import walk_forward
from banknifty_bot.utils.logging import get_logger
from banknifty_bot.utils.seeds import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_daily.yaml")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--objective", default="calmar")
    parser.add_argument("--max-combos", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    strategy = args.strategy or cfg.strategy.name

    df = read_partitioned(
        cfg.data.processed_dir, cfg.data.symbol, cfg.data.interval, cfg.data.start, cfg.data.end
    )
    if df.empty:
        log.error("No data. Fetch it first.")
        return
    log.info("Walk-forward %s on %d bars, %d folds", strategy, len(df), args.folds)

    result = walk_forward(cfg, df, strategy, n_folds=args.folds,
                          objective=args.objective, max_combos=args.max_combos)

    print("\n=== Walk-forward folds (out-of-sample) ===")
    print(result.folds.drop(columns=["params"]).to_string(index=False))

    print("\n=== Stitched OOS performance ===")
    for k in ("cagr_pct", "sharpe", "calmar", "max_drawdown_pct", "profit_factor", "win_rate_pct", "n_trades"):
        if k in result.oos_metrics:
            print(f"  {k:18s}: {result.oos_metrics[k]:.2f}")

    if not result.oos_trades.empty:
        mc = monte_carlo_bootstrap(result.oos_trades, cfg.risk.initial_equity)
        print("\n=== Monte Carlo (OOS trade-order bootstrap) ===")
        print(f"  median return : {mc['median_return_pct']:.1f}%")
        print(f"  5th pct return: {mc['p5_return_pct']:.1f}%")
        print(f"  worst max DD  : {mc['worst_max_dd_pct']:.1f}%")
        print(f"  prob. profit  : {mc['prob_profit']:.0%}")

    print(f"\nMost-selected params across folds: {result.best_params_overall}")

    out = Path(cfg.run.output_dir) / f"wfa_{strategy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "strategy": strategy,
        "oos_metrics": result.oos_metrics,
        "best_params_overall": result.best_params_overall,
        "folds": result.folds.drop(columns=["params"]).to_dict(orient="records"),
    }, indent=2, default=str))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
