from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from typing import Any

from .alpaca import AccountSnapshot


@dataclass(frozen=True)
class PublicPosition:
    symbol: str
    instrument_type: str
    quantity: float | None
    avg_entry_price: float | None
    market_value: float | None
    unrealized_pl: float | None


@dataclass(frozen=True)
class PublicOrder:
    order_id: str
    symbol: str
    instrument_type: str
    side: str
    quantity: float | None
    amount: float | None
    status: str
    filled_avg_price: float | None
    submitted_at: str | None


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Public portfolio {field} is unavailable") from exc
    if not math.isfinite(result):
        raise ValueError(f"Public portfolio {field} is not finite")
    return result


def _optional_finite_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field)


def _public_order_uuid(client_order_id: str) -> str:
    """Map a stable nonblank internal identity to Public's UUIDv4 shape.

    This helper exists only for read-only reconciliation of a previously known
    logical order. This adapter intentionally exposes no order-submission API.
    """
    normalized = client_order_id.strip()
    if not normalized:
        raise ValueError("client_order_id is required for Public reconciliation")
    digest = bytearray(hashlib.sha256(normalized.encode("utf-8")).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(digest)))


class PublicBroker:
    """Strictly read-only wrapper around Public's official Python SDK.

    Public has no paper sandbox. This class therefore exposes portfolio,
    instrument, and order-status reads only. It has no preflight, place, cancel,
    replace, or other order-mutating method. Live execution requires a separate
    reviewed implementation and explicit authorization.
    """

    def __init__(
        self,
        api_secret_key: str | None,
        account_number: str | None,
        *,
        client: Any | None = None,
    ):
        if not account_number:
            raise ValueError("Public account number is required")
        if client is None:
            if not api_secret_key:
                raise ValueError("Public API secret key is required")
            from public_api_sdk import (
                ApiKeyAuthConfig,
                PublicApiClient,
                PublicApiClientConfiguration,
            )

            client = PublicApiClient(
                ApiKeyAuthConfig(api_secret_key=api_secret_key, validity_minutes=15),
                config=PublicApiClientConfiguration(
                    default_account_number=account_number
                ),
            )
        self._client = client
        self.account_number = account_number

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def portfolio(self) -> Any:
        """Fetch one immutable-at-source portfolio response for consistent mapping."""
        return self._client.get_portfolio(account_id=self.account_number)

    def account(self, portfolio: Any | None = None) -> AccountSnapshot:
        portfolio = self.portfolio() if portfolio is None else portfolio
        cash = _finite_float(getattr(portfolio, "cash", None), "cash")
        total = getattr(portfolio, "total_account_value", None)
        if total is None:
            equity_rows = getattr(portfolio, "equity", None)
            if not equity_rows:
                raise ValueError("Public portfolio total account value is unavailable")
            values = [
                _finite_float(getattr(row, "value", None), "equity value")
                for row in equity_rows
            ]
            total = sum(values)
        portfolio_value = _finite_float(total, "total account value")
        buying_power_model = getattr(portfolio, "buying_power", None)
        cash_buying_power = getattr(buying_power_model, "cash_only_buying_power", None)
        buying_power = _finite_float(cash_buying_power, "cash-only buying power")

        # Public's portfolio response does not provide authoritative account-level
        # daily P/L. Summing current-position gains omits realized P/L after a
        # position is closed, so fail closed rather than reporting a false zero.
        return AccountSnapshot(
            cash,
            portfolio_value,
            buying_power,
            False,
            daily_pnl=None,
        )

    def positions(self, portfolio: Any | None = None) -> list[PublicPosition]:
        portfolio = self.portfolio() if portfolio is None else portfolio
        result: list[PublicPosition] = []
        for position in getattr(portfolio, "positions", []):
            instrument = getattr(position, "instrument", None)
            symbol = str(getattr(instrument, "symbol", "")).upper()
            instrument_type = str(
                getattr(
                    getattr(instrument, "type", ""),
                    "value",
                    getattr(instrument, "type", ""),
                )
            ).upper()
            cost_basis = getattr(position, "cost_basis", None)
            result.append(
                PublicPosition(
                    symbol=symbol,
                    instrument_type=instrument_type,
                    quantity=_optional_finite_float(
                        getattr(position, "quantity", None), f"{symbol} quantity"
                    ),
                    avg_entry_price=_optional_finite_float(
                        getattr(cost_basis, "unit_cost", None), f"{symbol} unit cost"
                    ),
                    market_value=_optional_finite_float(
                        getattr(position, "current_value", None),
                        f"{symbol} current value",
                    ),
                    unrealized_pl=_optional_finite_float(
                        getattr(cost_basis, "gain_value", None), f"{symbol} gain"
                    ),
                )
            )
        return result

    def open_orders(self, portfolio: Any | None = None) -> list[PublicOrder]:
        portfolio = self.portfolio() if portfolio is None else portfolio
        return [self._map_order(order) for order in getattr(portfolio, "orders", [])]

    def asset_class(self, symbol: str) -> str:
        from public_api_sdk import InstrumentType

        instrument = self._client.get_instrument(
            symbol.upper(), InstrumentType.EQUITY
        )
        kind = instrument.instrument.type
        if kind != InstrumentType.EQUITY:
            return str(getattr(kind, "value", kind)).upper()
        return "US_EQUITY"

    def order_by_client_order_id(self, client_order_id: str) -> PublicOrder | None:
        """Read broker state for a known stable identity; temporary 404 is unknown."""
        from public_api_sdk.exceptions import NotFoundError

        order_id = _public_order_uuid(client_order_id)
        try:
            order = self._client.get_order(
                order_id=order_id, account_id=self.account_number
            )
        except NotFoundError:
            return None
        return self._map_order(order)

    @staticmethod
    def _map_order(order: Any) -> PublicOrder:
        instrument = getattr(order, "instrument", None)
        kind = getattr(instrument, "type", "")
        side = getattr(order, "side", "")
        status = getattr(order, "status", "")
        created_at = getattr(order, "created_at", None)
        return PublicOrder(
            order_id=str(order.order_id),
            symbol=str(getattr(instrument, "symbol", "")).upper(),
            instrument_type=str(getattr(kind, "value", kind)).upper(),
            side=str(getattr(side, "value", side)).upper(),
            quantity=_optional_finite_float(
                getattr(order, "quantity", None), "order quantity"
            ),
            amount=_optional_finite_float(
                getattr(order, "notional_value", None), "order notional value"
            ),
            status=str(getattr(status, "value", status)).upper(),
            filled_avg_price=_optional_finite_float(
                getattr(order, "average_price", None), "order average price"
            ),
            submitted_at=None if created_at is None else str(created_at),
        )
