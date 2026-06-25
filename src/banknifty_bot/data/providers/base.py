from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Source of OHLCV bars. Implementations must return a DataFrame indexed by
    a tz-aware DatetimeIndex (Asia/Kolkata) with columns: open, high, low, close, volume.
    """

    @abstractmethod
    def get_ohlcv(
        self, symbol: str, start: str, end: str, interval: str
    ) -> pd.DataFrame: ...
