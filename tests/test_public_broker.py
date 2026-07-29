from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_agent.broker.public import PublicBroker, _public_order_uuid
from trading_agent.core.orders import Order, Side


def _position(
    symbol: str = "AAPL",
    *,
    quantity: str = "1.5",
    value: str = "150",
    unit_cost: str = "90",
    gain: str = "15",
    daily_gain: str | None = "2",
):
    return SimpleNamespace(
        instrument=SimpleNamespace(symbol=symbol),
        quantity=Decimal(quantity),
        current_value=Decimal(value),
        cost_basis=SimpleNamespace(
            unit_cost=Decimal(unit_cost), gain_value=Decimal(gain)
        ),
        position_daily_gain=(
            SimpleNamespace(gain_value=Decimal(daily_gain))
            if daily_gain is not None
            else None
        ),
    )


def _portfolio(
    *, positions=None, cash="200", total="210", buying_power="180", orders=None
):
    return SimpleNamespace(
        cash=None if cash is None else Decimal(cash),
        total_account_value=None if total is None else Decimal(total),
        buying_power=SimpleNamespace(cash_only_buying_power=Decimal(buying_power)),
        positions=list(positions or []),
        orders=list(orders or []),
        equity=[],
    )


def _broker(client=None, **kwargs) -> PublicBroker:
    kwargs.setdefault("now_fn", lambda: datetime(2026, 7, 28, 16, 0, tzinfo=UTC))
    return PublicBroker(None, "ACC-123", client=client or MagicMock(), **kwargs)


def test_constructor_requires_account_and_secret_without_injected_client():
    with pytest.raises(ValueError, match="account number"):
        PublicBroker("secret", None)
    with pytest.raises(ValueError, match="secret key"):
        PublicBroker(None, "ACC-123")


def test_order_submission_requires_a_positive_finite_ceiling():
    with pytest.raises(ValueError, match="finite max_order_notional"):
        _broker(allow_order_submission=True)
    with pytest.raises(ValueError, match="positive"):
        _broker(allow_order_submission=True, max_order_notional_usd=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        _broker(allow_order_submission=True, max_order_notional_usd=20.01)


@pytest.mark.parametrize(
    "observed",
    [
        datetime(2026, 7, 25, 16, 0, tzinfo=UTC),  # Saturday
        datetime(2026, 7, 28, 13, 40, tzinfo=UTC),  # opening buffer
        datetime(2026, 7, 28, 19, 50, tzinfo=UTC),  # closing buffer
    ],
)
def test_exchange_calendar_blocks_non_session_and_auction_windows(observed):
    broker = _broker(now_fn=lambda: observed)
    assert broker.execution_window_open() is False


def test_internal_client_id_maps_to_stable_uuid4_shape():
    first = _public_order_uuid("ta-same-intent")
    assert first == _public_order_uuid("ta-same-intent")
    assert first != _public_order_uuid("ta-other-intent")
    assert uuid.UUID(first).version == 4


def test_account_uses_cash_only_buying_power_and_marks_live():
    client = MagicMock()
    client.get_portfolio.return_value = _portfolio(positions=[_position()])
    account = _broker(client).account()
    assert account.cash == 200
    assert account.portfolio_value == 210
    assert account.buying_power == 180
    assert account.daily_pnl == 2
    assert account.is_paper is False


def test_account_fails_closed_when_daily_loss_state_is_incomplete():
    client = MagicMock()
    client.get_portfolio.return_value = _portfolio(
        positions=[_position(daily_gain=None)]
    )
    with pytest.raises(ValueError, match="daily loss state"):
        _broker(client).account()


def test_positions_map_cost_and_gain_fields():
    client = MagicMock()
    client.get_portfolio.return_value = _portfolio(positions=[_position()])
    position = _broker(client).positions()[0]
    assert position.symbol == "AAPL"
    assert position.quantity == 1.5
    assert position.avg_entry_price == 90
    assert position.market_value == 150
    assert position.unrealized_pl == 15


def test_read_only_default_blocks_before_any_preflight_or_order_call():
    client = MagicMock()
    with pytest.raises(PermissionError, match="read-only"):
        _broker(client).submit_market_order(
            Order("AAPL", Side.BUY, 1), client_order_id="ta-intent"
        )
    client.perform_preflight_calculation.assert_not_called()
    client.place_order.assert_not_called()


def test_enabled_submission_requires_deterministic_client_id():
    client = MagicMock()
    broker = _broker(client, allow_order_submission=True, max_order_notional_usd=20)
    with pytest.raises(ValueError, match="client_order_id"):
        broker.submit_market_order(Order("AAPL", Side.BUY, 1))
    client.perform_preflight_calculation.assert_not_called()
    client.place_order.assert_not_called()


def test_preflight_over_adapter_cap_blocks_order_submission():
    client = MagicMock()
    client.perform_preflight_calculation.return_value = SimpleNamespace(
        order_value=Decimal("20.01")
    )
    broker = _broker(client, allow_order_submission=True, max_order_notional_usd=20)
    with pytest.raises(PermissionError, match="max order notional"):
        broker.submit_market_order(
            Order("AAPL", Side.BUY, 1), client_order_id="ta-over-cap"
        )
    client.place_order.assert_not_called()


def test_enabled_order_is_cash_only_core_equity_and_preflighted_first():
    client = MagicMock()
    client.perform_preflight_calculation.return_value = SimpleNamespace(
        order_value=Decimal("19.50")
    )
    client.place_order.return_value = SimpleNamespace(order_id="broker-order")
    broker = _broker(client, allow_order_submission=True, max_order_notional_usd=20)

    result = broker.submit_market_order(
        Order("aapl", Side.BUY, 1), client_order_id="ta-safe-intent"
    )

    preflight = client.perform_preflight_calculation.call_args.args[0]
    request = client.place_order.call_args.args[0]
    assert preflight.instrument.symbol == "AAPL"
    assert preflight.instrument.type.value == "EQUITY"
    assert preflight.order_side.value == "BUY"
    assert preflight.order_type.value == "MARKET"
    assert preflight.equity_market_session.value == "CORE"
    assert preflight.validate_order is True
    assert preflight.open_close_indicator.value == "OPEN"
    assert request.use_margin is False
    assert request.order_id == _public_order_uuid("ta-safe-intent")
    assert client.method_calls[0][0] == "perform_preflight_calculation"
    assert client.method_calls[1][0] == "place_order"
    assert result.order_id == "broker-order"


def test_sell_is_explicitly_close_not_short_open():
    client = MagicMock()
    client.perform_preflight_calculation.return_value = SimpleNamespace(
        order_value=Decimal(10)
    )
    client.place_order.return_value = SimpleNamespace(order_id="sell-order")
    broker = _broker(client, allow_order_submission=True, max_order_notional_usd=20)
    broker.submit_market_order(Order("AAPL", Side.SELL, 1), client_order_id="ta-sell")
    preflight = client.perform_preflight_calculation.call_args.args[0]
    request = client.place_order.call_args.args[0]
    assert preflight.order_side.value == "SELL"
    assert preflight.open_close_indicator.value == "CLOSE"
    assert preflight.validate_order is True
    assert request.use_margin is False


def test_missing_public_order_returns_none_for_reconciliation():
    from public_api_sdk.exceptions import NotFoundError

    client = MagicMock()
    client.get_order.side_effect = NotFoundError()
    assert _broker(client).order_by_client_order_id("ta-missing") is None
