from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from ..core.orders import Order, Side
from .alpaca import AccountSnapshot


@dataclass(frozen=True)
class PublicPosition:
    symbol: str
    quantity: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


@dataclass(frozen=True)
class PublicOrder:
    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str
    filled_avg_price: float | None
    submitted_at: str


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Public portfolio {field} is unavailable") from exc
    if not math.isfinite(result):
        raise ValueError(f"Public portfolio {field} is not finite")
    return result


def _public_order_uuid(client_order_id: str) -> str:
    """Map an internal deterministic ID to Public's required UUIDv4 shape."""
    if not client_order_id:
        raise ValueError("client_order_id is required for Public order idempotency")
    digest = bytearray(hashlib.sha256(client_order_id.encode("utf-8")).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(digest)))


class PublicBroker:
    """Fail-closed wrapper around Public's official Python SDK.

    Public provides live-account infrastructure rather than a paper sandbox.
    The wrapper is read-only by default. Enabling order submission requires an
    explicit constructor flag and a positive per-order notional ceiling. Even
    then, only cash-only, core-session, single-leg US equity orders are exposed.
    """

    HARD_MAX_ORDER_NOTIONAL_USD = 20.0
    AUCTION_BUFFER_MINUTES = 15

    def __init__(
        self,
        api_secret_key: str | None,
        account_number: str | None,
        *,
        allow_order_submission: bool = False,
        max_order_notional_usd: float | None = None,
        client: Any | None = None,
        now_fn: Any | None = None,
    ):
        if not account_number:
            raise ValueError("Public account number is required")
        if allow_order_submission:
            if max_order_notional_usd is None or not math.isfinite(
                max_order_notional_usd
            ):
                raise ValueError(
                    "a finite max_order_notional_usd is required to enable Public orders"
                )
            if max_order_notional_usd <= 0:
                raise ValueError("max_order_notional_usd must be positive")
            if max_order_notional_usd > self.HARD_MAX_ORDER_NOTIONAL_USD:
                raise ValueError(
                    f"max_order_notional_usd cannot exceed ${self.HARD_MAX_ORDER_NOTIONAL_USD:.2f}"
                )
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
        self.allow_order_submission = allow_order_submission
        self.max_order_notional_usd = max_order_notional_usd
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _portfolio(self):
        return self._client.get_portfolio(account_id=self.account_number)

    def account(self) -> AccountSnapshot:
        portfolio = self._portfolio()
        cash = _finite_float(getattr(portfolio, "cash", None), "cash")
        total = getattr(portfolio, "total_account_value", None)
        if total is None:
            equity_rows = getattr(portfolio, "equity", None)
            if not equity_rows:
                raise ValueError("Public portfolio total account value is unavailable")
            total = sum((row.value for row in equity_rows), Decimal(0))
        portfolio_value = _finite_float(total, "total account value")
        buying_power_model = getattr(portfolio, "buying_power", None)
        cash_buying_power = getattr(buying_power_model, "cash_only_buying_power", None)
        buying_power = _finite_float(cash_buying_power, "cash-only buying power")

        daily_values: list[float] = []
        for position in getattr(portfolio, "positions", []):
            gain = getattr(position, "position_daily_gain", None)
            value = getattr(gain, "gain_value", None)
            if value is None:
                raise ValueError("Public portfolio daily loss state is incomplete")
            daily_values.append(_finite_float(value, "position daily gain"))
        daily_pnl = sum(daily_values)
        return AccountSnapshot(
            cash, portfolio_value, buying_power, False, daily_pnl=daily_pnl
        )

    def positions(self) -> list[PublicPosition]:
        result: list[PublicPosition] = []
        for position in getattr(self._portfolio(), "positions", []):
            instrument = position.instrument
            symbol = str(instrument.symbol).upper()
            quantity = _finite_float(position.quantity, f"{symbol} quantity")
            market_value = _finite_float(
                position.current_value, f"{symbol} current value"
            )
            cost_basis = getattr(position, "cost_basis", None)
            avg_entry = _finite_float(
                getattr(cost_basis, "unit_cost", None), f"{symbol} unit cost"
            )
            unrealized = _finite_float(
                getattr(cost_basis, "gain_value", None), f"{symbol} gain"
            )
            result.append(
                PublicPosition(symbol, quantity, avg_entry, market_value, unrealized)
            )
        return result

    def open_orders(self) -> list[PublicOrder]:
        return [
            self._map_order(order) for order in getattr(self._portfolio(), "orders", [])
        ]

    def asset_class(self, symbol: str) -> str:
        from public_api_sdk import InstrumentType

        instrument = self._client.get_instrument(symbol.upper(), InstrumentType.EQUITY)
        kind = instrument.instrument.type
        if kind != InstrumentType.EQUITY:
            return str(getattr(kind, "value", kind)).upper()
        return "US_EQUITY"

    def order_by_client_order_id(self, client_order_id: str) -> PublicOrder | None:
        from public_api_sdk.exceptions import NotFoundError

        order_id = _public_order_uuid(client_order_id)
        try:
            order = self._client.get_order(
                order_id=order_id, account_id=self.account_number
            )
        except NotFoundError:
            return None
        return self._map_order(order)

    def execution_window_open(self) -> bool:
        """Require an actual NYSE session and avoid both auction boundaries."""
        import exchange_calendars as xcals

        now = self._now_fn()
        if now.tzinfo is None:
            raise ValueError("execution clock must be timezone-aware")
        now_utc = now.astimezone(UTC)
        calendar = xcals.get_calendar("XNYS")
        session_date = now_utc.date()
        if not calendar.is_session(session_date):
            return False
        opened = calendar.session_open(session_date).to_pydatetime()
        closes = calendar.session_close(session_date).to_pydatetime()
        boundary = timedelta(minutes=self.AUCTION_BUFFER_MINUTES)
        return opened + boundary <= now_utc < closes - boundary

    def submit_market_order(
        self, order: Order, *, client_order_id: str | None = None
    ) -> PublicOrder:
        if not self.allow_order_submission:
            raise PermissionError(
                "Public order submission is disabled; adapter is read-only"
            )
        if client_order_id is None:
            raise ValueError("client_order_id is required for Public order idempotency")
        if order.quantity <= 0 or not math.isfinite(order.quantity):
            raise ValueError("order quantity must be finite and positive")
        if not self.execution_window_open():
            raise PermissionError(
                "Public order submission is outside the guarded NYSE execution window"
            )

        from public_api_sdk import (
            EquityMarketSession,
            InstrumentType,
            OpenCloseIndicator,
            OrderExpirationRequest,
            OrderInstrument,
            OrderRequest,
            OrderSide,
            OrderType,
            TimeInForce,
        )

        request = OrderRequest(
            order_id=_public_order_uuid(client_order_id),
            instrument=OrderInstrument(
                symbol=order.symbol.upper(), type=InstrumentType.EQUITY
            ),
            order_side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
            order_type=OrderType.MARKET,
            expiration=OrderExpirationRequest(
                time_in_force=TimeInForce.DAY, expiration_time=None
            ),
            quantity=Decimal(str(order.quantity)),
            amount=None,
            limit_price=None,
            stop_price=None,
            open_close_indicator=OpenCloseIndicator.OPEN
            if order.side is Side.BUY
            else OpenCloseIndicator.CLOSE,
            equity_market_session=EquityMarketSession.CORE,
            use_margin=False,
            tax_lot_matching_instructions=None,
        )
        preflight = self._client.perform_preflight_calculation(
            request, account_id=self.account_number
        )
        order_value = _finite_float(
            getattr(preflight, "order_value", None), "preflight order value"
        )
        if order_value <= 0:
            raise ValueError("Public preflight order value must be positive")
        if (
            self.max_order_notional_usd is None
            or order_value > self.max_order_notional_usd
        ):
            raise PermissionError(
                "Public preflight exceeds the adapter's max order notional"
            )

        submitted = self._client.place_order(request, account_id=self.account_number)
        return PublicOrder(
            order_id=str(submitted.order_id),
            symbol=order.symbol.upper(),
            side=order.side.value.upper(),
            quantity=float(order.quantity),
            status="SUBMITTED",
            filled_avg_price=None,
            submitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    @staticmethod
    def _map_order(order: Any) -> PublicOrder:
        instrument = getattr(order, "instrument", None)
        side = getattr(order, "side", "")
        status = getattr(order, "status", "")
        created_at = getattr(order, "created_at", None)
        return PublicOrder(
            order_id=str(order.order_id),
            symbol=str(getattr(instrument, "symbol", "")).upper(),
            side=str(getattr(side, "value", side)).upper(),
            quantity=_finite_float(getattr(order, "quantity", None), "order quantity"),
            status=str(getattr(status, "value", status)).upper(),
            filled_avg_price=(
                None
                if getattr(order, "average_price", None) is None
                else _finite_float(order.average_price, "order average price")
            ),
            submitted_at=str(created_at or ""),
        )
