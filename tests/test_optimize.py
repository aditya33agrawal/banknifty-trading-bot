import numpy as np
import pandas as pd
import pytest

from banknifty_bot.config import load_config
from banknifty_bot.optimize.robustness import monte_carlo_bootstrap
from banknifty_bot.optimize.search import grid_from_space, score
from banknifty_bot.optimize.walkforward import walk_forward


def test_grid_from_space_products_and_caps():
    space = {"a": [1, 2, 3], "b": [10, 20]}
    full = grid_from_space(space)
    assert len(full) == 6
    assert {"a": 2, "b": 20} in full
    capped = grid_from_space(space, max_combos=4, seed=1)
    assert len(capped) == 4


def test_score_handles_empty_metrics():
    assert score({}, "calmar") == float("-inf")
    assert score({"calmar": 1.0, "cost_pct_of_gross": 0.0}, "calmar") == pytest.approx(1.0)


def test_monte_carlo_bootstrap_shape():
    trades = pd.DataFrame({"net_pnl": [100.0, -50.0, 200.0, -30.0, 80.0]})
    mc = monte_carlo_bootstrap(trades, initial_equity=10_000, n_sims=200, seed=1)
    assert mc["n_sims"] == 200
    assert 0.0 <= mc["prob_profit"] <= 1.0
    assert mc["worst_max_dd_pct"] <= mc["median_max_dd_pct"]


@pytest.fixture
def daily_cfg(tmp_path):
    cfg = load_config("config/config_daily.yaml")
    return cfg


def test_walk_forward_runs_and_reports_oos(daily_cfg):
    # Synthetic trending daily series so at least some folds produce trades.
    idx = pd.date_range("2018-01-01", periods=600, freq="1D", tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    close = 20000 + np.cumsum(rng.normal(5, 80, len(idx)))
    df = pd.DataFrame(
        {"open": close, "high": close + 40, "low": close - 40, "close": close,
         "volume": 500}, index=idx
    )
    res = walk_forward(daily_cfg, df, "ema_trend", n_folds=2, max_combos=3, min_trades=1)
    assert not res.folds.empty
    assert "OOS_Calmar" in res.folds.columns
    assert isinstance(res.best_params_overall, dict)
