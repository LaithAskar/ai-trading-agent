from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .orders import Fill, Order, Side


@dataclass
class Portfolio:
    starting_cash: float
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    last_prices: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.starting_cash

    def apply_fill(self, fill: Fill) -> None:
        self.positions[fill.symbol] += fill.signed_quantity
        self.cash += fill.cash_delta
        if abs(self.positions[fill.symbol]) < 1e-9:
            del self.positions[fill.symbol]
        self.fills.append(fill)

    def fill_at(self, order: Order, price: float, timestamp: datetime) -> Fill:
        if order.side is Side.SELL:
            held = self.positions.get(order.symbol, 0.0)
            if order.quantity > held + 1e-9:
                raise ValueError(
                    f"Cannot sell {order.quantity} {order.symbol}; only {held} held"
                )
        else:
            cost = order.quantity * price
            if cost > self.cash + 1e-9:
                raise ValueError(
                    f"Insufficient cash for {order.quantity} {order.symbol} @ {price}: "
                    f"need {cost}, have {self.cash}"
                )

        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=timestamp,
        )
        self.apply_fill(fill)
        return fill

    def mark(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = price

    def equity(self) -> float:
        positions_value = sum(
            qty * self.last_prices.get(sym, 0.0)
            for sym, qty in self.positions.items()
        )
        return self.cash + positions_value

    def record_equity(self, timestamp: datetime) -> None:
        self.equity_curve.append((timestamp, self.equity()))

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)
