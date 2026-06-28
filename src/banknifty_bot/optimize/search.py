"""Parameter-grid construction and the objective used to rank configurations.

The objective deliberately rewards robust, low-churn edges rather than peak return:
it uses a risk-adjusted metric and penalises cost drag, so the optimizer doesn't
chase fragile, over-traded configurations (plan §7).
"""
from __future__ import annotations

import itertools
import random
from typing import Callable

OBJECTIVES: dict[str, Callable[[dict], float]] = {
    # Calmar with a soft penalty for bleeding to costs.
    "calmar": lambda m: m.get("calmar", 0.0) - 0.01 * max(m.get("cost_pct_of_gross", 0.0), 0.0),
    "sortino": lambda m: m.get("sortino", 0.0) - 0.01 * max(m.get("cost_pct_of_gross", 0.0), 0.0),
    "sharpe": lambda m: m.get("sharpe", 0.0),
    "profit_factor": lambda m: m.get("profit_factor", 0.0),
}


def grid_from_space(param_space: dict, max_combos: int | None = None, seed: int = 42) -> list[dict]:
    """Cartesian product of a strategy's `param_space`. If it exceeds `max_combos`,
    a deterministic random subset is returned so large spaces stay tractable."""
    keys = list(param_space.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*param_space.values())]
    if max_combos is not None and len(combos) > max_combos:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_combos)
    return combos


def score(metrics: dict, objective: str = "calmar") -> float:
    if not metrics:
        return float("-inf")
    return OBJECTIVES[objective](metrics)
