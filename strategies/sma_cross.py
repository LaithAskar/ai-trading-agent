from __future__ import annotations

from collections import deque

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


class SmaCross(Strategy):
    """Long-only SMA crossover: enter long when fast SMA crosses above slow SMA, exit when it crosses below.

    Position sizing: deploy ~100% of available cash on entry. Whole shares only.
    """

    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self._closes: deque[float] = deque(maxlen=slow)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    def _sma(self, n: int) -> float | None:
        if len(self._closes) < n:
            return None
        window = list(self._closes)[-n:]
        return sum(window) / n

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._closes.append(bar.close)
        fast = self._sma(self.fast)
        slow = self._sma(self.slow)

        orders: list[Order] = []
        if fast is not None and slow is not None and self._prev_fast is not None and self._prev_slow is not None:
            crossed_up = self._prev_fast <= self._prev_slow and fast > slow
            crossed_down = self._prev_fast >= self._prev_slow and fast < slow
            held = portfolio.position(bar.symbol)

            if crossed_up and held == 0:
                qty = int(portfolio.cash // bar.close)
                if qty > 0:
                    orders.append(Order(symbol=bar.symbol, side=Side.BUY, quantity=qty))
            elif crossed_down and held > 0:
                orders.append(Order(symbol=bar.symbol, side=Side.SELL, quantity=held))

        self._prev_fast = fast
        self._prev_slow = slow
        return orders
