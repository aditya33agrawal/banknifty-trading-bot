# BankNifty Intraday Trading Bot — Implementation Plan

> **Status:** Draft v1 · **Owner:** @adityaagrawal · **Last updated:** 2026-06-23

---

## 0. Decisions locked in (from kickoff)

| Topic | Decision | Implication |
|---|---|---|
| **Broker / API** | None yet. Build an abstracted data + execution pipeline; plug a real broker only when going to real money. | Define `DataProvider` and `Broker` interfaces with swappable adapters. Start with free data + a `PaperBroker`. |
| **Instrument** | **Index signal research first** (BankNifty index, `^NSEBANK`). | Backtest entries/exits on the index. Model execution cost as if traded via the realistic proxy (futures/options) so P&L is honest. |
| **Strategy style** | **Rule-based + indicators** (SOTA basics), ML deferred. | ORB, VWAP, EMA/Supertrend, RSI, regime filters — explainable, fast to optimize, hard to overfit. |
| **Capital** | **Paper only for now.** | Size everything in **% of equity / risk-per-trade**, not fixed lots. Validation > returns. |

### Guiding principles
1. **Costs are not optional.** Every backtest includes brokerage, STT, exchange, GST, stamp, SEBI, and slippage. A strategy that only works gross is not a strategy.
2. **Out-of-sample or it didn't happen.** Optimize on in-sample, judge on untouched out-of-sample + walk-forward.
3. **Robustness over peak returns.** A stable Sharpe 1.2 beats a fragile Sharpe 3.0.
4. **Everything reproducible.** Config-driven, seeded, versioned data, logged runs.
5. **Promotion gates.** A strategy only advances backtest → paper → live by passing explicit quantitative criteria (§9).
6. **Compute-heavy work on Colab**, fast iteration + paper trading local (§10).

---

## 1. High-level architecture

```
            ┌────────────────────────────────────────────────────┐
            │                    CONFIG (yaml)                     │
            │   data · costs · strategies · risk · run settings    │
            └────────────────────────────────────────────────────┘
                                   │
   ┌──────────────┐   ┌────────────────────┐   ┌─────────────────────┐
   │ DATA PIPELINE│──▶│ FEATURES/INDICATORS │──▶│  STRATEGY (signals) │
   │ providers +  │   │  vectorized library │   │  rule-based modules │
   │ clean + store│   └────────────────────┘   └─────────────────────┘
   └──────────────┘                                       │
                                                          ▼
          ┌───────────────────────────────────────────────────────────┐
          │                    BACKTEST ENGINE                          │
          │  event/bar loop · cost model · slippage · risk · portfolio  │
          └───────────────────────────────────────────────────────────┘
                 │                          │                     │
                 ▼                          ▼                     ▼
        ┌────────────────┐      ┌────────────────────┐   ┌────────────────┐
        │  OPTIMIZATION  │      │ EVALUATION + VIZ    │   │  PAPER TRADING │
        │ walk-forward,  │      │ metrics, plots,     │   │ PaperBroker +  │
        │ grid/optuna    │      │ HTML/PDF report     │   │ live loop      │
        │ (Colab)        │      └────────────────────┘   └────────────────┘
        └────────────────┘                                  (→ real broker later)
```

**Key abstraction layers** (so nothing is locked to a vendor):
- `DataProvider` (interface) → `YFinanceProvider`, `CSVProvider`, *later* `KiteProvider` / `FyersProvider`.
- `Broker` (interface) → `PaperBroker`, *later* `KiteBroker`.
- `Strategy` (interface) → concrete strategies registered in a registry, selected by config.

The backtest engine and the paper-trading loop both consume the **same** `Strategy` and `CostModel` objects — so what you backtest is exactly what you paper-trade (no logic drift).

---

## 2. Repository structure

```
trading_bot/
├── docs/
│   └── banknifty_bot_plan.md          # this file
├── config/
│   ├── config.yaml                    # global run config
│   ├── costs.yaml                     # cost model params (see §6)
│   └── strategies/                    # one yaml per strategy preset
├── data/
│   ├── raw/                           # immutable downloaded data
│   └── processed/                     # cleaned, resampled, partitioned by date
├── src/banknifty_bot/
│   ├── config.py                      # typed config loader (pydantic)
│   ├── data/
│   │   ├── providers/                 # base.py, yfinance_provider.py, csv_provider.py
│   │   ├── cleaner.py                 # gaps, holidays, sessions, adjustments
│   │   └── store.py                   # parquet read/write, partitioning
│   ├── features/
│   │   └── indicators.py              # vectorized TA (vwap, ema, rsi, atr, supertrend, adx...)
│   ├── strategies/
│   │   ├── base.py                    # Strategy interface
│   │   ├── orb.py  vwap.py  trend.py  supertrend.py  rsi_reversion.py
│   │   └── registry.py
│   ├── backtest/
│   │   ├── engine.py                  # bar loop, position lifecycle
│   │   ├── costs.py                   # CostModel (Indian charges)
│   │   ├── slippage.py                # slippage models
│   │   ├── risk.py                    # sizing, stops, daily limits, square-off
│   │   └── portfolio.py              # equity, trades ledger
│   ├── optimize/
│   │   ├── walkforward.py             # rolling IS/OOS folds
│   │   ├── search.py                  # grid + optuna objective
│   │   └── robustness.py             # monte carlo, sensitivity, regime, deflated sharpe
│   ├── evaluation/
│   │   ├── metrics.py                 # all performance stats
│   │   ├── plots.py                   # matplotlib/plotly charts
│   │   └── report.py                  # assemble HTML/PDF report
│   ├── paper/
│   │   ├── broker_base.py paper_broker.py
│   │   └── live_loop.py               # scheduler, market-hours, square-off
│   └── utils/  (logging.py, calendar.py — NSE holidays/sessions, seeds.py)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_backtest.ipynb
│   ├── 03_optimization_colab.ipynb    # the heavy one (Colab)
│   └── 04_robustness_report.ipynb
├── scripts/
│   ├── fetch_data.py  run_backtest.py  run_optimization.py
│   ├── generate_report.py  paper_trade.py
├── tests/                             # pytest: costs, indicators, engine invariants
├── requirements.txt
└── README.md
```

---

## 3. Phase 0 — Project setup

- [ ] Python env (`venv`/`conda`), `requirements.txt` (pinned), `pyproject.toml` for the `banknifty_bot` package (installable so Colab can `pip install` it).
- [ ] `config.py` with **pydantic** typed config; load `config/*.yaml`.
- [ ] Logging (`utils/logging.py`), reproducibility seeds, NSE trading calendar (`utils/calendar.py`: holidays, 09:15–15:30 session, square-off cutoff).
- [ ] `pytest` skeleton + CI-style `make test` / `make backtest` targets.

**Stack:** Python 3.11+, `pandas`, `numpy`, `polars` (optional, fast), `pyarrow`/parquet, `pydantic`, `pyyaml`, `yfinance` / `nsepython` / `jugaad-data` (data), `vectorbt` *or* a custom engine (decide §5), `optuna` (optimization), `matplotlib` + `plotly` + `quantstats` (viz/metrics), `joblib` (parallel), `pytest`.

---

## 4. Phase 1 — Data pipeline

**The hardest part of intraday backtesting is good data.** We have no paid API, so a tiered approach:

| Tier | Source | Resolution / history | Use |
|---|---|---|---|
| A (now) | `yfinance` `^NSEBANK` | Daily: many years · Intraday: 1m≈7d, 5m/15m≈60d (rolling) | Daily-bar backtests + recent intraday |
| B (accumulate) | Scheduled daily fetch → append to local parquet store | Builds a growing intraday history going forward | Long-term local dataset |
| C (bulk history) | One-time multi-year 1-min dataset (Kaggle / public GitHub repos / one-off purchase) | Years of 1m/5m | Serious intraday robustness testing |
| D (later) | Real broker historical API (Kite/Fyers) once subscribed | Clean, authoritative | Final validation before live |

- [ ] `DataProvider` interface: `get_ohlcv(symbol, start, end, interval)`.
- [ ] `YFinanceProvider` + `CSVProvider` (for Tier C bulk files).
- [ ] **Cleaner:** drop non-session bars, handle holidays/half-days, forward-fill gaps carefully (never fabricate trades), de-duplicate, timezone → `Asia/Kolkata`, sanity checks (no negative/huge jumps).
- [ ] **Store:** partitioned parquet (`data/processed/interval=5m/year=YYYY/...`), with a manifest of coverage so we know exactly what data backs each backtest.
- [ ] `scripts/fetch_data.py` — idempotent incremental fetch (Tier B cron candidate).

> **Data caveat to record per run:** which tier/source + date coverage backs the result. A great backtest on 60 days of data is not evidence.

---

## 5. Phase 2 & 3 — Indicators + rule-based strategies

**Engine choice:** start with **`vectorbt`** for fast vectorized signal research and parameter sweeps; build a small **custom event/bar engine** for realistic intraday mechanics (intrabar stops, square-off, daily loss limits) used for the *final* validated backtests and shared with paper trading. (Vectorbt for breadth, custom engine for truth.)

### Indicator library (`features/indicators.py`) — vectorized
VWAP (session-anchored), EMA/SMA, RSI, ATR, Supertrend, ADX/DI, Bollinger, opening-range high/low, rolling volatility, time-of-day features, prev-day high/low/close.

### Strategy modules (BankNifty intraday classics, all config-parameterized)
1. **Opening Range Breakout (ORB)** — break of first N-min (15/30) range, ATR stop, target/trail. *Params:* range minutes, buffer, SL×ATR, target R, re-entry rule.
2. **VWAP strategy** — trend (price vs VWAP + slope) and/or mean-reversion to VWAP bands.
3. **EMA trend** — 9/21 (configurable) crossover, **ADX filter** to trade only in trends.
4. **Supertrend** — flip signals with volatility filter.
5. **RSI mean-reversion** — oversold/overbought in a range regime only.
6. **Regime filter (meta)** — trending vs ranging (ADX / realized vol) gates which sub-strategy is active; a "do-nothing" state is valid.

**Shared rules baked into every strategy (`risk.py`):**
- Intraday only — hard square-off at configurable cutoff (e.g. 15:15).
- No-trade windows (first/last few minutes, optional event days).
- Per-trade stop loss + target/trailing (points or ATR-based).
- **Daily max loss** → halt for the day. **Max trades/day.** **One position at a time** (initially).
- Position sizing in **% risk per trade** (e.g. risk 1% of equity; lots derived from stop distance & contract value of the chosen execution proxy).

### `Strategy` interface
```python
class Strategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame: ...  # entries/exits/size hints
    @property
    def param_space(self) -> dict: ...                                 # for optimization
```

---

## 6. Phase 4 — Backtest engine + **cost model**

### Engine (`backtest/engine.py`)
Bar-by-bar: consume signals → apply risk/sizing → simulate fills with slippage → apply costs → update portfolio/trade ledger → enforce daily limits & square-off. Outputs: per-trade ledger, equity curve, per-day P&L.

### Cost model (`backtest/costs.py`, params in `config/costs.yaml`)
Since the index isn't tradeable, model the **execution proxy**. Defaults below reflect typical NSE F&O charges — **treat as configurable and verify against current SEBI/exchange/broker circulars** (these rates change).

**Options (premium-based) — default template:**
| Charge | Basis | Typical default |
|---|---|---|
| Brokerage | per order | ₹20 (flat) |
| STT | on **sell** premium | 0.1% |
| Exchange txn | on premium | ~0.035% |
| SEBI | on premium | ₹10 / crore (0.0001%) |
| Stamp duty | on **buy** premium | 0.003% |
| GST | on (brokerage + exch + SEBI) | 18% |

**Futures — default template:**
| Charge | Basis | Typical default |
|---|---|---|
| Brokerage | per order | ₹20 (flat) |
| STT | on **sell** value | 0.02% |
| Exchange txn | on value | ~0.0019% |
| SEBI | on value | ₹10 / crore |
| Stamp duty | on **buy** value | 0.002% |
| GST | on (brokerage + exch + SEBI) | 18% |

- [ ] `CostModel.round_trip_cost(entry, exit, qty, side)` returning a full breakdown.
- [ ] Report **costs as % of gross P&L** and **₹/trade** — a core decision metric.
- [ ] Unit tests asserting cost math against hand-computed examples.

> **Contract spec caveat:** BankNifty lot size and expiry rules changed under SEBI (2024–25, e.g. weekly expiries / lot revisions). Keep `lot_size`, `expiry`, and the execution proxy **in config**, not hardcoded; verify current spec before paper/live.

### Slippage (`slippage.py`)
Configurable: fixed ticks, % of price, or ATR-fraction. Run every final backtest at **base / 2× / 4×** slippage to test fragility.

---

## 7. Phase 5 — Optimization (Colab-heavy)

- [ ] **Walk-forward analysis** (`optimize/walkforward.py`): rolling IS train → OOS test folds; report OOS-stitched equity (the headline result).
- [ ] **Search** (`optimize/search.py`): grid for small spaces, **Optuna** for larger; parallel via `joblib`. Objective = robust risk-adjusted metric (e.g. OOS Sortino or Calmar), **penalized for trade count & cost drag** to avoid overfit churn.
- [ ] Guard against multiple-testing: track # configs tried, report **Deflated/Probabilistic Sharpe**.

### Colab workflow
1. Push repo to GitHub (or mount Google Drive). Colab cell: `pip install -e` the package.
2. Upload/point to the processed parquet data (Drive) — keep data out of git.
3. Run `run_optimization.py` (Optuna study) on Colab CPU/GPU; persist the **study + best params + full results table** to Drive.
4. **Download artifacts locally** (`best_params.json`, `wfa_results.parquet`, plots). Local machine re-runs the *single* best config through the high-fidelity custom engine for the final report.
- [ ] `03_optimization_colab.ipynb` implements exactly this, parameterized.

---

## 8. Phase 6 — Evaluation, metrics & **visualization**

### Metrics (`evaluation/metrics.py`)
Returns: total, CAGR, monthly/annualized. Risk-adjusted: **Sharpe, Sortino, Calmar**. Drawdown: max DD, duration, recovery. Trade stats: **win rate, profit factor, expectancy, payoff ratio, avg/median win & loss, max consecutive losses, exposure %, turnover**. Cost stats: **total costs, costs % of gross, ₹/trade**. Distribution: skew, tail, worst day.

### Visualizations (`evaluation/plots.py` — matplotlib + interactive plotly)
- Equity curve (gross vs **net-of-costs**) with drawdown shading.
- Underwater (drawdown) curve.
- **Monthly returns heatmap.**
- Rolling Sharpe / rolling volatility.
- Trade P&L distribution histogram; MAE/MFE scatter.
- Intraday: avg P&L by **time-of-day** and by **weekday**.
- **Parameter sensitivity heatmaps** (is the optimum a plateau or a spike?).
- Walk-forward fold-by-fold bar chart (IS vs OOS).
- Monte Carlo equity cone (trade-order bootstrap).

### Report (`evaluation/report.py`)
- [ ] One command → **self-contained HTML report** (consider `quantstats` for the tearsheet + custom sections): all metrics tables + all charts + the run's data coverage, cost assumptions, and parameters. This is the artifact used to make go/no-go decisions.

---

## 9. Phase 7 — Promotion gates (backtest → paper → live)

A strategy is **promoted to paper trading** only if, on **out-of-sample / walk-forward** data, net of costs:

- [ ] Positive OOS net return across **the majority of WFA folds** (not one lucky fold).
- [ ] **Sharpe ≥ ~1.0** and **Calmar ≥ ~0.5** OOS (tune thresholds).
- [ ] **Profit factor ≥ ~1.3**, expectancy clearly positive after costs.
- [ ] **Max drawdown** within tolerance (e.g. ≤ 15–20%).
- [ ] **Costs < ~30–40% of gross** P&L (not bleeding to charges).
- [ ] Survives **2×/4× slippage** without flipping to a loss.
- [ ] **Parameter plateau** (neighbors of the optimum also profitable) — not a knife-edge.
- [ ] Monte Carlo: acceptable worst-case drawdown / ruin probability.

*(Exact numbers go in `config.yaml` and get tuned as we learn the data.)*

**Paper → live** gate (future): N weeks of paper results that **track the backtest distribution** (similar win rate, expectancy, drawdown), with live slippage/fills within modeled assumptions.

---

## 10. Phase 8 — Paper trading

- [ ] `Broker` interface + `PaperBroker` (simulated fills using live/last data + the **same CostModel**).
- [ ] `paper/live_loop.py`: market-hours scheduler → fetch latest bars → `strategy.generate_signals` → risk checks → `PaperBroker` orders → log trades → enforce daily loss limit + auto square-off.
- [ ] Persist a paper trade ledger; nightly auto-report comparing **paper vs backtest** expectations.
- [ ] Alerting/log of every decision for trust-building.

> Same `Strategy` + `CostModel` objects as the backtest → no logic drift between sim and paper.

---

## 11. Phase 9 — Live (future, out of scope now)

When promoted: implement a real `Broker` adapter (e.g. `KiteBroker`/`FyersBroker`), add order-state reconciliation, partial-fill handling, kill-switch, capital limits, secrets management, monitoring/alerting. Start with **1 lot** and the strictest daily loss limit.

---

## 12. Risk management (applies at every phase)
- % risk per trade; daily max loss halt; max trades/day; single open position initially.
- Hard intraday square-off; no overnight risk.
- Kill-switch / manual override.
- Never let live config diverge from the validated, backtested config.

---

## 13. Milestones / suggested order of work

1. **M0** Setup + config + calendar + logging *(Phase 0)*
2. **M1** Data pipeline + store + cleaner; first daily-bar dataset *(Phase 1)*
3. **M2** Indicator library + cost model + unit tests *(Phases 2, 6-costs)*
4. **M3** Backtest engine (custom) + 1 strategy (ORB) end-to-end with costs *(Phases 3–4)*
5. **M4** Evaluation + visualization + HTML report *(Phase 6)*
6. **M5** Remaining strategies + vectorbt sweep harness *(Phase 3/5)*
7. **M6** Walk-forward + Optuna + Colab notebook + robustness suite *(Phase 5)*
8. **M7** Apply promotion gates; pick winner(s) *(Phase 7)*
9. **M8** Paper broker + live loop + paper-vs-backtest tracking *(Phase 8)*
10. **M9** *(later)* Real broker adapter + go-live checklist *(Phase 9)*

---

## 14. Open items / to verify before money is at risk
- [ ] Confirm a real **multi-year intraday data** source (Tier C) — current free data limits serious intraday robustness.
- [ ] Verify **current** charge rates (STT/exchange/stamp/GST) and **BankNifty contract spec** (lot size, expiry) against latest circulars.
- [ ] Decide final **execution proxy** for cost modeling (futures vs options) before paper trading.
- [ ] Pick the **broker** when moving toward real money; implement its adapter behind the existing `Broker` interface.

---

## 15. Tech stack summary
`python 3.11+` · `pandas`/`numpy`/`polars` · `pyarrow` (parquet) · `pydantic` · `yfinance`/`nsepython` (data) · `vectorbt` (sweeps) + custom engine (truth) · `optuna` + `joblib` (optimization) · `matplotlib`/`plotly`/`quantstats` (viz) · `pytest` (tests) · **Colab** (heavy compute) · Google Drive/GitHub (artifact sync).
