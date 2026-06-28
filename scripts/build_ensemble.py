#!/usr/bin/env python
"""Build a multi-timeframe ensemble from the top sweep results.

Loads the sweep leaderboard, picks the top-N uncorrelated cells, runs each
member's backtest independently, then blends their daily return streams.

Usage:
  python scripts/build_ensemble.py --top 4 --sweep outputs/sweep_results.json \
    --out outputs/ensemble_result.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from banknifty_bot.backtest.runner import run_backtest
from banknifty_bot.config import load_config
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.evaluation.metrics import full_report
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[1]

OBJ_FIELD = {
    "calmar": "oos_calmar",
    "sortino": "oos_sortino",
    "sharpe": "oos_sharpe",
    "profit_factor": "oos_profit_factor",
}


def _daily_equity(equity_df: pd.DataFrame) -> pd.Series:
    return equity_df["equity"].resample("1D").last().ffill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--sweep", default="outputs/sweep_results.json")
    parser.add_argument("--out", default="outputs/ensemble_result.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-calmar", type=float, default=0.3)
    parser.add_argument("--objective", default="calmar",
                        choices=["calmar", "sortino", "sharpe", "profit_factor"])
    args = parser.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        search_cfg = REPO / "config" / "config_search.yaml"
        cfg_path = str(search_cfg) if search_cfg.exists() else str(REPO / "config" / "config_intraday.yaml")

    cfg = load_config(cfg_path)
    sweep_path = REPO / args.sweep
    out_path = REPO / args.out

    if not sweep_path.exists():
        log.error("Sweep results not found at %s. Run run_sweep.py first.", sweep_path)
        return

    rows = json.loads(sweep_path.read_text())
    df_sweep = pd.DataFrame(rows)
    sort_field = OBJ_FIELD.get(args.objective, "oos_calmar")

    # Filter and sort.
    df_sweep = df_sweep[df_sweep.get("oos_calmar", pd.Series(dtype=float)) >= args.min_calmar]
    if df_sweep.empty:
        log.error("No rows pass min-calmar=%.2f. Lower the threshold or run more sweep cells.", args.min_calmar)
        return

    df_sweep = df_sweep.sort_values(sort_field, ascending=False).reset_index(drop=True)

    # Deduplicate: prefer highest-scoring entry per strategy, allow same strategy at different intervals.
    seen_strat: dict[str, str] = {}
    selected = []
    for _, row in df_sweep.iterrows():
        key = (row["strategy"], row["interval"])
        strat_key = row["strategy"]
        # Allow same strategy at a different interval, but not exact same (interval, strategy).
        if strat_key in seen_strat and seen_strat[strat_key] == row["interval"]:
            continue
        if key not in {(r["strategy"], r["interval"]) for r in selected}:
            selected.append(row.to_dict())
            seen_strat[strat_key] = row["interval"]
        if len(selected) >= args.top:
            break

    if not selected:
        log.error("Could not select any members.")
        return

    log.info("Selected %d members:", len(selected))
    for m in selected:
        log.info("  %s × %s  Calmar=%.2f", m["interval"], m["strategy"], m.get("oos_calmar", 0))

    # Run each member's backtest to get equity curves.
    weight = 1.0 / len(selected)
    daily_series: dict[str, pd.Series] = {}
    all_trades: list[pd.DataFrame] = []
    member_out: list[dict] = []

    for member in selected:
        interval = member["interval"]
        strategy = member["strategy"]
        params = member.get("best_params") or {}
        label = f"{interval}_{strategy}"

        cfg.data.interval = interval
        df = read_partitioned(
            cfg.data.processed_dir, cfg.data.symbol, interval,
            cfg.data.start, cfg.data.end,
        )
        if df.empty:
            log.warning("No data for %s %s — skipping member.", strategy, interval)
            continue

        try:
            res = run_backtest(cfg, df, strategy, params)
        except Exception as exc:
            log.error("Backtest failed for %s %s: %s", strategy, interval, exc)
            continue

        if res.equity_df.empty:
            log.warning("No equity for %s %s — skipping.", strategy, interval)
            continue

        daily_series[label] = _daily_equity(res.equity_df)
        if not res.trades_df.empty:
            tagged = res.trades_df.copy()
            tagged["member"] = label
            all_trades.append(tagged)

        m = res.metrics
        member_out.append({
            **member,
            "member_weight": round(weight, 4),
            "backtest_cagr_pct": round(m.get("cagr_pct", 0), 2),
            "backtest_sharpe": round(m.get("sharpe", 0), 2),
            "backtest_calmar": round(m.get("calmar", 0), 2),
            "backtest_n_trades": m.get("n_trades", 0),
        })

    if not daily_series:
        log.error("All members failed — no ensemble built.")
        return

    # Align on common date index, compute blended daily returns.
    equity_panel = pd.DataFrame(daily_series).ffill()
    daily_returns = equity_panel.pct_change().dropna(how="all")
    blended_returns = daily_returns.mean(axis=1)  # equal weight

    # Reconstruct blended equity from initial equity.
    blended_equity = (1 + blended_returns).cumprod() * cfg.risk.initial_equity
    blended_equity_df = blended_equity.rename("equity").to_frame()

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    blended_metrics = full_report(blended_equity_df, all_trades_df)

    corr = daily_returns.corr()

    out = {
        "members": member_out,
        "blended_metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in blended_metrics.items()},
        "correlation_matrix": corr.round(3).to_dict(),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Ensemble result written to %s", out_path)
    log.info("Blended Calmar=%.2f Sharpe=%.2f CAGR=%.1f%%",
             blended_metrics.get("calmar", 0), blended_metrics.get("sharpe", 0),
             blended_metrics.get("cagr_pct", 0))


if __name__ == "__main__":
    main()
