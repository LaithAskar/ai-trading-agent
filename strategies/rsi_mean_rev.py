from __future__ import annotations

from collections import deque

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


class RsiMeanReversion(Strategy):
    """Long-only RSI mean-reversion.

    Uses a simple (non-Wilder) `period`-bar SMA of gains and losses on closes.
    Entry: RSI crosses DOWN through `oversold` (we became oversold; expect bounce).
    Exit:  RSI crosses UP through `overbought` (we became overbought; expect pullback).

    Sizing: deploy ~100% of available cash on entry, whole shares only.
    """

    name = "rsi_mean_rev"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        if period < 2:
            raise ValueError("period must be >= 2")
        if not (0 < oversold < overbought < 100):
            raise ValueError("require 0 < oversold < overbought < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._closes: deque[float] = deque(maxlen=period + 1)
        self._prev_rsi: float | None = None

    def _rsi(self) -> float | None:
        if len(self._closes) < self.period + 1:
            return None
        prices = list(self._closes)
        gains = losses = 0.0
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains += change
            else:
                losses += -change
        if losses == 0:
            return 100.0
        rs = (gains / self.period) / (losses / self.period)
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._closes.append(bar.close)
        rsi = self._rsi()

        orders: list[Order] = []
        if rsi is not None and self._prev_rsi is not None:
            held = portfolio.position(bar.symbol)
            crossed_into_oversold = self._prev_rsi >= self.oversold and rsi < self.oversold
            crossed_into_overbought = self._prev_rsi <= self.overbought and rsi > self.overbought

            if crossed_into_oversold and held == 0:
                qty = int(portfolio.cash // bar.close)
                if qty > 0:
                    orders.append(Order(symbol=bar.symbol, side=Side.BUY, quantity=qty))
            elif crossed_into_overbought and held > 0:
                orders.append(Order(symbol=bar.symbol, side=Side.SELL, quantity=held))

        self._prev_rsi = rsi
        return orders
