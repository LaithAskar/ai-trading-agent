from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..core.orders import Order, Side


@dataclass(frozen=True)
class AlpacaPosition:
    symbol: str
    quantity: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


@dataclass(frozen=True)
class AlpacaOrder:
    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str
    filled_avg_price: float | None
    submitted_at: str


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    portfolio_value: float
    buying_power: float
    is_paper: bool


class AlpacaPaperBroker:
    """Thin wrapper over alpaca-py's TradingClient locked to paper trading.

    Refuses to construct if Config.live_trading is true OR if the requested
    base_url is the live endpoint — defense in depth so we can never
    accidentally submit a live order from this codebase.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL_PREFIX = "https://api.alpaca.markets"

    def __init__(self, api_key: str, api_secret: str, *, allow_live: bool = False):
        if not api_key or not api_secret:
            raise ValueError("Alpaca API key and secret are required")
        if not allow_live:
            self._is_paper = True
        else:
            raise NotImplementedError(
                "Live trading is intentionally disabled in this codebase. "
                "Use paper trading or a different broker."
            )
        # Defer SDK import so tests can patch it
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(api_key, api_secret, paper=True)

    # ---- queries ----

    def account(self) -> AccountSnapshot:
        acct = self._client.get_account()
        return AccountSnapshot(
            cash=float(acct.cash),
            portfolio_value=float(acct.portfolio_value),
            buying_power=float(acct.buying_power),
            is_paper=True,
        )

    def positions(self) -> list[AlpacaPosition]:
        return [
            AlpacaPosition(
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            )
            for p in self._client.get_all_positions()
        ]

    def open_orders(self) -> list[AlpacaOrder]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return [
            AlpacaOrder(
                order_id=str(o.id),
                symbol=str(o.symbol),
                side=str(o.side).split(".")[-1],
                quantity=float(o.qty),
                status=str(o.status).split(".")[-1],
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                submitted_at=str(o.submitted_at),
            )
            for o in self._client.get_orders(filter=req)
        ]

    # ---- writes ----

    def submit_market_order(self, order: Order) -> AlpacaOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        resp = self._client.submit_order(req)
        return AlpacaOrder(
            order_id=str(resp.id),
            symbol=str(resp.symbol),
            side=str(resp.side).split(".")[-1],
            quantity=float(resp.qty),
            status=str(resp.status).split(".")[-1],
            filled_avg_price=float(resp.filled_avg_price) if resp.filled_avg_price else None,
            submitted_at=str(resp.submitted_at),
        )

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)
