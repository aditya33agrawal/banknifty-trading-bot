from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Literal

Side = Literal["long", "short"]


class Broker(ABC):
    """Order/position interface. `PaperBroker` simulates fills now; a real
    `KiteBroker`/`FyersBroker` implements the same interface later (plan §9)
    so the live loop never changes when swapping execution.
    """

    @abstractmethod
    def open_position(self, side: Side, qty: int, price: float, ts: dt.datetime) -> bool: ...

    @abstractmethod
    def close_position(self, price: float, ts: dt.datetime, reason: str) -> None: ...

    @abstractmethod
    def has_open_position(self) -> bool: ...
