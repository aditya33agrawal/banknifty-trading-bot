.PHONY: install test fetch-data backtest report paper-trade

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

fetch-data:
	python scripts/fetch_data.py --config config/config.yaml

backtest:
	python scripts/run_backtest.py --config config/config.yaml

report:
	python scripts/generate_report.py --config config/config.yaml

paper-trade:
	python scripts/paper_trade.py --config config/config.yaml
