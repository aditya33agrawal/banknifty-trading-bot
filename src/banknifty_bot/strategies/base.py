from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Rule-based strategy interface. `generate_signals` must be pure/vectorized
    given the input bars + params — no lookahead, no I/O.

    Output DataFrame (same index as `df`) columns:
      - entry: 1 (go long), -1 (go short), 0 (no entry signal this bar)
      - stop_loss: absolute price for the protective stop, set on entry bars
      - target: absolute price for the take-profit, set on entry bars (NaN = none)
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def param_space(self) -> dict: ...
