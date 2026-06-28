"""Stateful daily paper trader for the portfolio ensemble (plan §10).

Each member strategy runs as an independent capital *sleeve* (its own position and
equity), like `backtest/portfolio_ensemble.py` — so paper results stay comparable to
the blended backtest. State (open positions, equity, trade ledger) persists to JSON
between runs, so this is invoked **once per trading day after the close** (manually,
cron, or /schedule), not as a tight poll loop.

Fills use the same `CostModel` + `Slippage` as the backtest, and exits mirror the
engine's stop / target / ATR-trailing logic (the active live config). Breakeven /
partial / time-stop exits are not mirrored yet — they are off in config_daily.

Sizing note: at current index levels (~58k, ATR ~700) one BankNifty lot risks ~₹10k,
so a ¼-capital sleeve needs a few % risk-per-trade to afford a lot — hence the paper
risk level is set higher than the backtest's 1% (which only traded in the low-index
early years).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.exits import ExitConfig
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.features.indicators import atr as atr_indicator
from banknifty_bot.strategies.registry import get_strategy
from banknifty_bot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Sleeve:
    name: str
    params: dict
    weight: float


class PortfolioPaperTrader:
    def __init__(
        self,
        sleeves: list[Sleeve],
        cost_model: CostModel,
        slippage: Slippage,
        initial_equity: float,
        risk_per_trade_pct: float,
        lot_size: int,
        contract_multiplier: float,
        exit_cfg: ExitConfig,
        atr_window: int,
        state_path: str | Path,
    ):
        self.sleeves = sleeves
        self.cost_model = cost_model
        self.slippage = slippage
        self.initial_equity = initial_equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.lot_size = lot_size
        self.contract_multiplier = contract_multiplier
        self.exit_cfg = exit_cfg
        self.atr_window = atr_window
        self.state_path = Path(state_path)
        self.state = self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {
            "created": dt.datetime.now().isoformat(),
            "last_ts": None,
            "initial_equity": self.initial_equity,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "sleeves": {
                s.name: {"weight": s.weight, "equity": self.initial_equity * s.weight, "position": None}
                for s in self.sleeves
            },
            "trades": [],
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    # ------------------------------------------------------------------ sizing / fills
    def _position_size(self, sleeve_equity: float, entry: float, stop: float) -> int:
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0
        risk_amount = sleeve_equity * self.risk_per_trade_pct / 100
        unit_risk = stop_distance * self.contract_multiplier
        lots = int(risk_amount / (unit_risk * self.lot_size)) if unit_risk > 0 else 0
        return max(lots, 0) * self.lot_size

    def _close(self, sl_state: dict, pos: dict, raw_exit: float, ts, reason: str, sleeve_name: str) -> None:
        side = pos["side"]
        fill_side = "sell" if side == "long" else "buy"
        exit_price = self.slippage.adjust(raw_exit, fill_side)
        qty = pos["qty"]
        if side == "long":
            gross = (exit_price - pos["entry_price"]) * qty * self.contract_multiplier
        else:
            gross = (pos["entry_price"] - exit_price) * qty * self.contract_multiplier
        cost = self.cost_model.round_trip_cost(pos["entry_price"], exit_price, qty, side).total
        net = gross - cost
        sl_state["equity"] += net
        sl_state["position"] = None
        self.state["trades"].append({
            "sleeve": sleeve_name, "entry_time": pos["entry_time"], "exit_time": str(ts),
            "side": side, "entry_price": round(pos["entry_price"], 2), "exit_price": round(exit_price, 2),
            "qty": qty, "gross_pnl": round(gross, 2), "cost": round(cost, 2),
            "net_pnl": round(net, 2), "exit_reason": reason,
        })
        log.info("PAPER %s CLOSE %s qty=%d @ %.2f net=%.2f (%s)", sleeve_name, side, qty, exit_price, net, reason)

    # ------------------------------------------------------------------ per-bar core (shared by step + backfill)
    def _advance_sleeve(self, name: str, sig_row: pd.Series, atr_val: float, bar: pd.Series, ts) -> None:
        sl_state = self.state["sleeves"][name]
        pos = sl_state["position"]

        if pos is not None:
            long = pos["side"] == "long"
            if (long and bar["low"] <= pos["stop_loss"]) or (not long and bar["high"] >= pos["stop_loss"]):
                self._close(sl_state, pos, pos["stop_loss"], ts,
                            "trail_stop" if pos.get("stop_moved") else "stop", name)
                return
            if pos["target"] is not None and ((long and bar["high"] >= pos["target"]) or (not long and bar["low"] <= pos["target"])):
                self._close(sl_state, pos, pos["target"], ts, "target", name)
                return
            pos["best_price"] = max(pos["best_price"], bar["high"]) if long else min(pos["best_price"], bar["low"])
            if self.exit_cfg.trailing_atr_mult is not None and atr_val == atr_val:  # atr not NaN
                trail = (pos["best_price"] - self.exit_cfg.trailing_atr_mult * atr_val if long
                         else pos["best_price"] + self.exit_cfg.trailing_atr_mult * atr_val)
                new_stop = max(pos["stop_loss"], trail) if long else min(pos["stop_loss"], trail)
                if new_stop != pos["stop_loss"]:
                    pos["stop_loss"], pos["stop_moved"] = new_stop, True
            return

        if sig_row["entry"] != 0 and pd.notna(sig_row["stop_loss"]):
            side = "long" if sig_row["entry"] == 1 else "short"
            fill_side = "buy" if side == "long" else "sell"
            entry_price = self.slippage.adjust(float(bar["close"]), fill_side, float(atr_val) if atr_val == atr_val else None)
            qty = self._position_size(sl_state["equity"], entry_price, float(sig_row["stop_loss"]))
            if qty > 0:
                sl_state["position"] = {
                    "side": side, "entry_time": str(ts), "entry_price": entry_price, "qty": qty,
                    "stop_loss": float(sig_row["stop_loss"]),
                    "target": float(sig_row["target"]) if pd.notna(sig_row["target"]) else None,
                    "best_price": entry_price, "stop_moved": False,
                }
                log.info("PAPER %s OPEN %s qty=%d @ %.2f sl=%.0f", name, side, qty, entry_price, sig_row["stop_loss"])

    # ------------------------------------------------------------------ live: one new bar
    def step(self, bars_by_sleeve: dict[str, pd.DataFrame], ts, force: bool = False) -> None:
        if not force and self.state["last_ts"] is not None and str(ts) <= self.state["last_ts"]:
            log.info("No new bar (last processed %s) — nothing to do.", self.state["last_ts"])
            return
        for s in self.sleeves:
            df = bars_by_sleeve[s.name]
            signals = get_strategy(s.name, s.params).generate_signals(df)
            atr_series = atr_indicator(df, window=self.atr_window)
            self._advance_sleeve(s.name, signals.iloc[-1], float(atr_series.iloc[-1]), df.iloc[-1], ts)
        self.state["last_ts"] = str(ts)
        self._save_state()

    # ------------------------------------------------------------------ backfill: replay last n bars (fast)
    def backfill(self, df: pd.DataFrame, n: int) -> None:
        """Prime the ledger by replaying the last `n` bars. Signals/ATR are causal,
        so they are computed once over the full series and sliced per bar — far
        faster than recomputing each step."""
        n = min(n, len(df) - 1)
        precomputed = {
            s.name: (get_strategy(s.name, s.params).generate_signals(df), atr_indicator(df, window=self.atr_window))
            for s in self.sleeves
        }
        for i in range(len(df) - n, len(df)):
            ts = df.index[i]
            if self.state["last_ts"] is not None and str(ts) <= self.state["last_ts"]:
                continue
            bar = df.iloc[i]
            for s in self.sleeves:
                signals, atr_series = precomputed[s.name]
                self._advance_sleeve(s.name, signals.iloc[i], float(atr_series.iloc[i]), bar, ts)
            self.state["last_ts"] = str(ts)
        self._save_state()

    # ------------------------------------------------------------------ reporting
    def status(self) -> dict:
        sleeves = self.state["sleeves"]
        total_equity = sum(s["equity"] for s in sleeves.values())
        trades = pd.DataFrame(self.state["trades"])
        return {
            "total_equity": total_equity,
            "initial_equity": self.state["initial_equity"],
            "return_pct": (total_equity / self.state["initial_equity"] - 1) * 100,
            "last_ts": self.state["last_ts"],
            "open_positions": {k: v["position"] for k, v in sleeves.items() if v["position"]},
            "sleeves": sleeves,
            "trades": trades,
        }
