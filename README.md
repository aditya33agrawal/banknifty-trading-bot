# banknifty-trading-bot

Rule-based BankNifty intraday trading bot: data pipeline → indicators → strategies →
backtest engine (with realistic Indian F&O costs) → evaluation/report → paper trading.

No real broker yet — paper only. See `docs/banknifty_bot_plan.md` for the full design
and `docs/banknifty_bot_plan.md` §9 for promotion gates before any live capital.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"
```

`vectorbt` (used for fast parameter sweeps, plan §5/§7) needs a numba/llvmlite
toolchain that may not build on every machine — install separately if you want
it: `pip install vectorbt`. It is not required for the core pipeline below.

## Workflow

```bash
make fetch-data      # pull + clean + store OHLCV (config/config.yaml)
make backtest        # run the configured strategy through the backtest engine
make report          # same, plus a self-contained HTML report in outputs/
make paper-trade     # paper-trading loop (same Strategy + CostModel as backtest)
make test            # pytest
```

Strategies (`config/strategies/*.yaml`, registered in
`src/banknifty_bot/strategies/registry.py`): `orb`, `vwap`, `ema_trend`,
`supertrend`, `rsi_reversion`, `regime_filter` (meta strategy that routes
between a trend and a mean-reversion sub-strategy by ADX regime).

## Layout

See `docs/banknifty_bot_plan.md` §2 for the full repository structure and the
rationale behind each module.
