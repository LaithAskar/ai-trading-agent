from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.broker.public import PublicBroker, _public_order_uuid


def _portfolio(
    *,
    cash: Any = Decimal("175"),
    total: Any = Decimal("200"),
    buying_power: Any = Decimal("175"),
    positions: Any = None,
    orders: Any = None,
):
    return SimpleNamespace(
        cash=cash,
        total_account_value=total,
        equity=[],
        buying_power=SimpleNamespace(cash_only_buying_power=buying_power),
        positions=[] if positions is None else positions,
        orders=[] if orders is None else orders,
    )


def _broker(client=None) -> PublicBroker:
    return PublicBroker(None, "ACC-123", client=client or MagicMock())


def test_requires_account_number_even_with_injected_client():
    with pytest.raises(ValueError, match="account number"):
        PublicBroker(None, None, client=MagicMock())


def test_requires_secret_when_constructing_real_sdk_client():
    with pytest.raises(ValueError, match="secret key"):
        PublicBroker(None, "ACC-123")


def test_adapter_exposes_no_order_mutation_surface():
    broker = _broker()
    for method in (
        "submit_market_order",
        "preflight_order",
        "place_order",
        "cancel_order",
        "replace_order",
    ):
        assert not hasattr(broker, method)


def test_public_order_uuid_is_deterministic_uuid4_and_rejects_blank():
    first = _public_order_uuid("ta-stable-intent")
    assert first == _public_order_uuid(" ta-stable-intent ")
    assert first != _public_order_uuid("ta-other-intent")
    assert first[14] == "4"
    with pytest.raises(ValueError, match="required"):
        _public_order_uuid("   ")


def test_account_maps_cash_values_but_daily_pnl_is_authoritatively_unavailable():
    client = MagicMock()
    portfolio = _portfolio(positions=[])
    account = _broker(client).account(portfolio)

    assert account.cash == 175.0
    assert account.portfolio_value == 200.0
    assert account.buying_power == 175.0
    assert account.is_paper is False
    assert account.daily_pnl is None
    client.get_portfolio.assert_not_called()


def test_account_can_sum_equity_rows_when_total_is_absent():
    portfolio = _portfolio(total=None)
    portfolio.equity = [
        SimpleNamespace(value=Decimal("150")),
        SimpleNamespace(value=Decimal("50")),
    ]
    assert _broker().account(portfolio).portfolio_value == 200.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cash", None),
        ("cash", Decimal("NaN")),
        ("total_account_value", None),
        ("buying_power", None),
    ],
)
def test_account_fails_closed_on_required_balance_state(field, value):
    portfolio = _portfolio()
    setattr(portfolio, field, value)
    if field == "total_account_value":
        portfolio.equity = []
    with pytest.raises(ValueError):
        _broker().account(portfolio)


def test_positions_tolerate_sdk_optional_values_without_aborting_status():
    position = SimpleNamespace(
        instrument=SimpleNamespace(
            symbol="aapl", type=SimpleNamespace(value="EQUITY")
        ),
        quantity=Decimal("1.5"),
        current_value=None,
        cost_basis=SimpleNamespace(unit_cost=None, gain_value=None),
    )
    mapped = _broker().positions(_portfolio(positions=[position]))[0]

    assert mapped.symbol == "AAPL"
    assert mapped.instrument_type == "EQUITY"
    assert mapped.quantity == 1.5
    assert mapped.market_value is None
    assert mapped.avg_entry_price is None
    assert mapped.unrealized_pl is None


def test_open_notional_order_maps_without_quantity():
    order = SimpleNamespace(
        order_id="order-1",
        instrument=SimpleNamespace(
            symbol="aapl", type=SimpleNamespace(value="EQUITY")
        ),
        side=SimpleNamespace(value="BUY"),
        quantity=None,
        amount=Decimal("10"),
        status=SimpleNamespace(value="NEW"),
        average_price=None,
        created_at=None,
    )
    mapped = _broker().open_orders(_portfolio(orders=[order]))[0]

    assert mapped.quantity is None
    assert mapped.amount == 10.0
    assert mapped.status == "NEW"
    assert mapped.submitted_at is None


def test_one_portfolio_snapshot_can_feed_all_read_models_without_refetching():
    client = MagicMock()
    portfolio = _portfolio()
    broker = _broker(client)

    broker.account(portfolio)
    broker.positions(portfolio)
    broker.open_orders(portfolio)

    client.get_portfolio.assert_not_called()


def test_missing_public_order_returns_none_for_read_only_reconciliation():
    from public_api_sdk.exceptions import NotFoundError

    client = MagicMock()
    client.get_order.side_effect = NotFoundError()
    assert _broker(client).order_by_client_order_id("ta-missing") is None


def test_close_delegates_to_sdk_client():
    client = MagicMock()
    _broker(client).close()
    client.close.assert_called_once()
