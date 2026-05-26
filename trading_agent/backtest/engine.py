from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.events import Bar
from ..core.orders import Order
from ..core.portfolio import Portfolio
from ..core.strategy import Strategy


@dataclass
class BacktestResult:
    portfolio: Portfolio
    strategy_name: str
    symbol: str


class BacktestEngine:
    """Single-symbol event-driven backtest.

    Order-of-operations per bar (the lookahead-safe contract):
      1. Fill orders submitted at the PREVIOUS bar at THIS bar's open price.
      2. Mark portfolio to market at THIS bar's close.
      3. Record equity.
      4. Strategy.on_bar() sees the closed bar and emits new orders.
      5. Those orders wait until the NEXT bar's open to fill.

    Orders still pending after the final bar are dropped (no fill).

    Slippage: applied symmetrically to the next-bar open. Buys fill at
    open * (1 + slippage_bps/10000); sells at open * (1 - slippage_bps/10000).
    A coarse model — real slippage is volume- and liquidity-dependent — but
    enough to keep optimistic backtests honest.

    Commission: a flat dollar fee per fill, deducted from cash post-fill.
    """

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        slippage_bps: float = 0.0,
        commission_per_trade: float = 0.0,
    ):
        if slippage_bps < 0 or commission_per_trade < 0:
            raise ValueError("slippage_bps and commission_per_trade must be >= 0")
        self.starting_cash = starting_cash
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade

    def _adjusted_fill_price(self, order: Order, raw_price: float) -> float:
        if self.slippage_bps == 0:
            return raw_price
        bps = self.slippage_bps / 10_000.0
        if order.side.value == "BUY":
            return raw_price * (1.0 + bps)
        return raw_price * (1.0 - bps)

    def run(
        self,
        strategy: Strategy,
        symbol: str,
        bars: Iterable[Bar],
    ) -> BacktestResult:
        portfolio = Portfolio(starting_cash=self.starting_cash)
        strategy.on_start([symbol])

        pending: list[Order] = []

        for bar in bars:
            for order in pending:
                fill_price = self._adjusted_fill_price(order, bar.open)
                try:
                    portfolio.fill_at(order, price=fill_price, timestamp=bar.timestamp)
                except ValueError:
                    continue
                if self.commission_per_trade > 0:
                    portfolio.cash -= self.commission_per_trade
            pending = []

            portfolio.mark(bar.symbol, bar.close)
            portfolio.record_equity(bar.timestamp)

            new_orders = strategy.on_bar(bar, portfolio)
            if new_orders:
                pending.extend(new_orders)

        strategy.on_finish(portfolio)
        return BacktestResult(
            portfolio=portfolio,
            strategy_name=strategy.name,
            symbol=symbol,
        )
