#!/usr/bin/env python
"""Daily paper trader for the portfolio ensemble.

Run once per trading day AFTER the close (manually, cron, or /schedule):
  python scripts/paper_trade_ensemble.py --config config/config_daily.yaml

First-time priming with recent history (builds an initial forward ledger):
  python scripts/paper_trade_ensemble.py --backfill 250

State persists to outputs/paper_ensemble_state.json; the dashboard 📡 Paper tab reads it.
"""
from __future__ import annotations

import argparse

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.exits import ExitConfig
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.config import load_config, load_yaml
from banknifty_bot.data.store import read_partitioned
from banknifty_bot.paper.portfolio_paper import PortfolioPaperTrader, Sleeve
from banknifty_bot.strategies.registry import SWING_COMPATIBLE
from banknifty_bot.utils.logging import get_logger
from banknifty_bot.utils.seeds import set_seed

log = get_logger(__name__)

STATE_PATH = "outputs/paper_ensemble_state.json"


def build_trader(cfg, risk_per_trade_pct: float, state_path=STATE_PATH) -> PortfolioPaperTrader:
    members = sorted(SWING_COMPATIBLE - {"ensemble"})
    weight = 1.0 / len(members)
    sleeves = [Sleeve(name, load_yaml(f"config/strategies/{name}.yaml"), weight) for name in members]
    cost_model = CostModel.from_yaml(cfg.backtest.costs_file, cfg.execution_proxy.instrument)
    slippage = Slippage(cfg.backtest.slippage_model, cfg.backtest.slippage_value)
    return PortfolioPaperTrader(
        sleeves=sleeves, cost_model=cost_model, slippage=slippage,
        initial_equity=cfg.risk.initial_equity, risk_per_trade_pct=risk_per_trade_pct,
        lot_size=cfg.execution_proxy.lot_size, contract_multiplier=cfg.execution_proxy.contract_multiplier,
        exit_cfg=ExitConfig(**cfg.exits.model_dump()), atr_window=14, state_path=state_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_daily.yaml")
    parser.add_argument("--backfill", type=int, default=0,
                        help="Prime the ledger by replaying the last N stored daily bars.")
    parser.add_argument("--risk-per-trade", type=float, default=5.0,
                        help="%% of sleeve equity risked per trade. Higher than the backtest's "
                             "1%% so a ¼-capital sleeve can afford a lot at current index levels.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    trader = build_trader(cfg, args.risk_per_trade)

    df = read_partitioned(cfg.data.processed_dir, cfg.data.symbol, cfg.data.interval, cfg.data.start, cfg.data.end)
    if df.empty:
        log.error("No data. Run scripts/fetch_data.py --config %s first.", args.config)
        return

    if args.backfill > 0:
        log.info("Backfilling %d trading days of paper decisions…", args.backfill)
        trader.backfill(df, args.backfill)
    else:
        trader.step({s.name: df for s in trader.sleeves}, df.index[-1])

    st = trader.status()
    print("\n=== Paper ensemble status ===")
    print(f"  as of bar     : {st['last_ts']}")
    print(f"  total equity  : ₹{st['total_equity']:,.0f}  ({st['return_pct']:+.2f}%)")
    print(f"  open positions: {len(st['open_positions'])}/{len(trader.sleeves)} sleeves")
    for name, pos in st["open_positions"].items():
        print(f"     {name:14s} {pos['side']:5s} qty={pos['qty']} entry={pos['entry_price']:.0f} stop={pos['stop_loss']:.0f}")
    if not st["trades"].empty:
        t = st["trades"]
        print(f"  closed trades : {len(t)}  net P&L ₹{t['net_pnl'].sum():,.0f}  win% {(t['net_pnl'] > 0).mean() * 100:.0f}")
    log.info("State at %s", trader.state_path)


if __name__ == "__main__":
    main()
