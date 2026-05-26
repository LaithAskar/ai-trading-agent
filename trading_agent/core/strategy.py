from __future__ import annotations

from abc import ABC, abstractmethod

from .events import Bar
from .orders import Order
from .portfolio import Portfolio


class Strategy(ABC):
    name: str = "unnamed"

    def on_start(self, symbols: list[str]) -> None:
        pass

    @abstractmethod
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        """Called once after each bar closes.

        Return orders to submit. They fill at the NEXT bar's open price.
        The bar passed in is the most recent closed bar — never a future bar.
        """

    def on_finish(self, portfolio: Portfolio) -> None:
        pass
