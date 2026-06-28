"""Robustness checks — how much of a result is skill vs luck of trade ordering.

`monte_carlo_bootstrap` resamples the realized trade P&Ls to build a distribution of
final equity and max drawdown, so you can read off a worst-case (e.g. 5th percentile)
outcome instead of trusting a single historical path (plan §6, §9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_bootstrap(
    trades_df: pd.DataFrame,
    initial_equity: float,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict:
    if trades_df.empty:
        return {}

    pnls = trades_df["net_pnl"].to_numpy()
    rng = np.random.default_rng(seed)

    final_equities = np.empty(n_sims)
    max_drawdowns = np.empty(n_sims)
    for i in range(n_sims):
        shuffled = rng.permutation(pnls)              # same trades, random order
        equity = initial_equity + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity)
        dd = ((equity - peak) / peak).min() * 100
        final_equities[i] = equity[-1]
        max_drawdowns[i] = dd

    ret_pct = (final_equities / initial_equity - 1) * 100
    return {
        "n_sims": n_sims,
        "median_return_pct": float(np.median(ret_pct)),
        "p5_return_pct": float(np.percentile(ret_pct, 5)),
        "p95_return_pct": float(np.percentile(ret_pct, 95)),
        "median_max_dd_pct": float(np.median(max_drawdowns)),
        "worst_max_dd_pct": float(max_drawdowns.min()),
        "prob_profit": float((ret_pct > 0).mean()),
        "final_equities": final_equities,
        "max_drawdowns": max_drawdowns,
    }
