from .base import DataProvider
from .csv_provider import CSVProvider
from .yfinance_provider import YFinanceProvider

PROVIDERS = {
    "yfinance": YFinanceProvider,
    "csv": CSVProvider,
}

__all__ = ["DataProvider", "YFinanceProvider", "CSVProvider", "PROVIDERS"]
