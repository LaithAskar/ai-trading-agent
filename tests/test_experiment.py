from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.core.orders import Order, Side
from trading_agent.experiment import (
    ExperimentContract,
    PortfolioSnapshot,
    evaluate_order,
    evaluate_order_batch,
)

NOW = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=1)


def snapshot(
    *,
    cash: float = 200.0,
    portfolio_value: float = 200.0,
    positions_market_value: dict[str, float] | None = None,
    positions_quantity: dict[str, float] | None = None,
    daily_pnl: float | None = 0.0,
    total_pnl: float | None = 0.0,
    trades_today: int | None = 0,
    market_is_open: bool | None = True,
    observed_at: datetime | None = NOW,
    session_open: datetime | None = NOW - timedelta(hours=2),
    session_close: datetime | None = NOW + timedelta(hours=2),
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        cash=cash,
        portfolio_value=portfolio_value,
        positions_market_value=positions_market_value or {},
        positions_quantity=positions_quantity or {},
        daily_pnl=daily_pnl,
        total_pnl=total_pnl,
        trades_today=trades_today,
        market_is_open=market_is_open,
        observed_at=observed_at,
        session_open=session_open,
        session_close=session_close,
    )


def decide(contract: ExperimentContract, order: Order, *, price: float, portfolio: PortfolioSnapshot, asset="US_EQUITY"):
    return evaluate_order(contract, order, price=price, portfolio=portfolio, asset_class=asset)


def test_contract_defaults_are_autonomous_and_trade_count_unbounded():
    contract = ExperimentContract(name="graph-trading-30d")
    assert contract.autonomous is True
    assert contract.trade_approval_required is False
    assert contract.mode == "paper"
    assert contract.max_trades_per_day is None


def test_live_contract_requires_explicit_live_enabled():
    with pytest.raises(ValueError, match="live mode requires"):
        ExperimentContract(name="live-test", mode="live")


def test_contract_requires_timezone_aware_creation_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        ExperimentContract(name="bad-time", created_at="2026-07-28T12:00:00")


@pytest.mark.parametrize(
    ("contract", "reason"),
    [
        (
            ExperimentContract(
                name="expired",
                created_at="2020-01-01T00:00:00+00:00",
                duration_days=1,
            ),
            "experiment contract has expired",
        ),
        (
            ExperimentContract(name="manual", autonomous=False),
            "contract does not authorize autonomous execution",
        ),
        (
            ExperimentContract(name="approval", trade_approval_required=True),
            "contract requires trade approval",
        ),
    ],
)
def test_gate_enforces_contract_time_and_approval_posture(contract, reason):
    decision = decide(
        contract,
        Order("AAPL", Side.BUY, 1),
        price=10,
        portfolio=snapshot(),
    )
    assert decision.allowed is False
    assert decision.reason == reason


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), float("-inf"), True])
def test_order_rejects_non_finite_or_boolean_quantity(quantity):
    with pytest.raises(ValueError, match="quantity must be positive"):
        Order("AAPL", Side.BUY, quantity)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_contract_rejects_non_finite_limits(value):
    with pytest.raises(ValueError, match="must be finite"):
        ExperimentContract(name="bad-limit", capital_cap_usd=value)


def test_gate_allows_order_inside_contract_without_trade_count_cap():
    decision = decide(
        ExperimentContract(name="paper-test", max_trades_per_day=None),
        Order("AAPL", Side.BUY, 1),
        price=19.0,
        portfolio=snapshot(trades_today=None),
    )
    assert decision.allowed is True
    assert decision.order_notional == 19.0


def test_gate_rejects_trade_above_notional_cap():
    decision = decide(
        ExperimentContract(name="paper-test"), Order("AAPL", Side.BUY, 2), price=15.0, portfolio=snapshot()
    )
    assert decision.reason == "trade exceeds max_trade_usd"


@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf")])
def test_gate_rejects_non_finite_price(price):
    decision = decide(
        ExperimentContract(name="paper-test"), Order("AAPL", Side.BUY, 1), price=price, portfolio=snapshot()
    )
    assert decision.allowed is False
    assert decision.reason == "price must be finite and positive"


@pytest.mark.parametrize(
    "changes",
    [
        {"cash": float("nan")},
        {"portfolio_value": float("inf")},
        {"positions_market_value": {"AAPL": float("nan")}},
        {"positions_quantity": {"AAPL": float("inf")}},
        {"daily_pnl": float("nan")},
        {"total_pnl": float("-inf")},
    ],
)
def test_gate_fails_closed_on_non_finite_snapshot_values(changes):
    decision = decide(
        ExperimentContract(name="paper-test"),
        Order("AAPL", Side.BUY, 1),
        price=10,
        portfolio=snapshot(**changes),
    )
    assert decision.allowed is False
    assert decision.reason == "portfolio snapshot contains non-finite values"


def test_gate_rejects_position_exposure_above_symbol_cap():
    decision = decide(
        ExperimentContract(name="paper-test"),
        Order("AAPL", Side.BUY, 1),
        price=15.0,
        portfolio=snapshot(positions_market_value={"AAPL": 30.0}),
    )
    assert decision.reason == "symbol exposure exceeds max_position_usd"
    assert decision.projected_symbol_exposure == 45.0


def test_gate_enforces_contract_capital_cap_across_symbols():
    contract = ExperimentContract(name="paper-test", capital_cap_usd=50, max_trade_usd=20, max_position_usd=40)
    decision = decide(
        contract,
        Order("MSFT", Side.BUY, 2),
        price=10.0,
        portfolio=snapshot(positions_market_value={"AAPL": 40.0}),
    )
    assert decision.reason == "projected exposure exceeds capital_cap_usd"
    assert decision.projected_total_exposure == 60.0


@pytest.mark.parametrize("asset", ["OPTIONS", "CRYPTO"])
def test_gate_rejects_blocked_assets(asset):
    decision = decide(ExperimentContract(name="paper-test"), Order("AAPL", Side.BUY, 1), price=10, portfolio=snapshot(), asset=asset)
    assert decision.reason == "asset class is blocked"


def test_gate_rejects_asset_not_on_allowlist_and_missing_class():
    contract = ExperimentContract(name="paper-test")
    assert decide(contract, Order("AAPL", Side.BUY, 1), price=10, portfolio=snapshot(), asset="FOREX").reason == "asset class is not allowed"
    assert decide(contract, Order("AAPL", Side.BUY, 1), price=10, portfolio=snapshot(), asset="").reason == "asset class unavailable"


def test_gate_never_allows_short_sale():
    order = Order("AAPL", Side.SELL, 2)
    decision = decide(
        ExperimentContract(name="paper-test"),
        order,
        price=10,
        portfolio=snapshot(positions_market_value={"AAPL": 10}, positions_quantity={"AAPL": 1}),
    )
    assert decision.reason == "sell would create a short position"


def test_gate_rejects_closed_market_and_auction_boundaries_but_allows_exact_open_boundary():
    contract = ExperimentContract(name="paper-test", no_open_close_auction_window_minutes=15)
    order = Order("AAPL", Side.BUY, 1)
    assert decide(contract, order, price=10, portfolio=snapshot(market_is_open=False)).reason == "market is closed"
    assert decide(
        contract, order, price=10,
        portfolio=snapshot(observed_at=NOW, session_open=NOW - timedelta(minutes=14, seconds=59)),
    ).reason == "inside market-open auction boundary"
    assert decide(
        contract, order, price=10,
        portfolio=snapshot(observed_at=NOW, session_close=NOW + timedelta(minutes=15)),
    ).reason == "inside market-close auction boundary"
    assert decide(
        contract, order, price=10,
        portfolio=snapshot(observed_at=NOW, session_open=NOW - timedelta(minutes=15), session_close=NOW + timedelta(hours=1)),
    ).allowed is True


def test_batch_gate_reserves_position_cash_and_trade_count_cumulatively():
    contract = ExperimentContract(
        name="paper-test", capital_cap_usd=100, max_trade_usd=20, max_position_usd=20, max_trades_per_day=2
    )
    orders = [Order("AAPL", Side.BUY, 1), Order("AAPL", Side.BUY, 1), Order("MSFT", Side.BUY, 1)]
    decisions = evaluate_order_batch(
        contract,
        orders,
        prices={"AAPL": 15, "MSFT": 10},
        portfolio=snapshot(cash=100, trades_today=0),
        asset_classes={"AAPL": "US_EQUITY", "MSFT": "US_EQUITY"},
    )
    assert [decision.allowed for decision in decisions] == [True, False, True]
    assert decisions[1].reason == "symbol exposure exceeds max_position_usd"
    assert decisions[2].projected_total_exposure == 25


def test_gate_rejects_daily_and_total_loss_stops():
    contract = ExperimentContract(name="paper-test")
    order = Order("AAPL", Side.BUY, 1)
    assert decide(contract, order, price=10, portfolio=snapshot(daily_pnl=-4.0)).reason == "daily loss stop reached"
    assert decide(contract, order, price=10, portfolio=snapshot(total_pnl=-30.0)).reason == "total loss stop reached"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"daily_pnl": None}, "daily loss state unavailable"),
        ({"total_pnl": None}, "total loss state unavailable"),
        ({"trades_today": None}, "trade-count state unavailable"),
    ],
)
def test_gate_fails_closed_when_required_risk_state_is_unavailable(changes, reason):
    contract = ExperimentContract(name="paper-test", max_trades_per_day=5)
    assert decide(contract, Order("AAPL", Side.BUY, 1), price=10, portfolio=snapshot(**changes)).reason == reason


def test_gate_respects_optional_trade_count_cap_when_set():
    decision = decide(
        ExperimentContract(name="paper-test", max_trades_per_day=5),
        Order("AAPL", Side.BUY, 1), price=10.0, portfolio=snapshot(trades_today=5),
    )
    assert decision.reason == "max trades per day reached"


def test_research_mode_never_executes_orders():
    decision = decide(
        ExperimentContract(name="research-test", mode="research"),
        Order("AAPL", Side.BUY, 1), price=10.0, portfolio=snapshot(),
    )
    assert decision.reason == "research mode cannot execute orders"
