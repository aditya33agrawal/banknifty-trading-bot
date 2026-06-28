.PHONY: install test fetch-data fetch-daily fetch-intraday import-bulk backtest optimize report paper-trade paper-ensemble paper-ensemble-backfill dashboard import-1m check-data resample sweep ensemble

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

fetch-data:
	python scripts/fetch_data.py --config config/config.yaml

fetch-daily:
	python scripts/fetch_data.py --config config/config_daily.yaml

fetch-intraday:
	python scripts/fetch_data.py --config config/config_intraday.yaml

import-bulk:
	@echo "Usage: python scripts/import_bulk_csv.py --file <path.csv> --interval 1m"

backtest:
	python scripts/run_backtest.py --config config/config.yaml

optimize:
	python scripts/run_optimization.py --config config/config_daily.yaml --strategy donchian

dashboard:
	streamlit run dashboard/app.py

report:
	python scripts/generate_report.py --config config/config.yaml

paper-trade:
	python scripts/paper_trade.py --config config/config.yaml

paper-ensemble:
	python scripts/paper_trade_ensemble.py --config config/config_daily.yaml

paper-ensemble-backfill:
	python scripts/paper_trade_ensemble.py --config config/config_daily.yaml --backfill 250

import-1m:
	python scripts/import_bulk_csv.py --file data/raw/bank-nifty-1m-data.csv --interval 1m --dayfirst

check-data:
	python scripts/check_data.py

resample:
	python scripts/resample_intervals.py --intervals 3m,5m,15m,30m,60m

sweep:
	python scripts/run_sweep.py --intervals 5m,15m,30m,60m,1d --strategies ema_trend,supertrend,rsi_reversion,donchian,orb,vwap --out outputs/sweep_results.json

ensemble:
	python scripts/build_ensemble.py --top 4 --out outputs/ensemble_result.json

