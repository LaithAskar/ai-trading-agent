from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from strategies.ai_oligopoly_leaders import AiOligopolyLeaders
from trading_agent.core.events import Bar
from trading_agent.core.orders import Side
from trading_agent.core.portfolio import Portfolio


def bar(price: float, index: int, symbol: str = "NVDA") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2026, 1, 1) + timedelta(days=index),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def test_only_user_curated_leader_universe_can_generate_entries():
    strategy = AiOligopolyLeaders(
        fast=2,
        slow=3,
        momentum_lookback=2,
        min_momentum_pct=5,
        leaders="NVDA,MSFT",
        target_notional=20,
    )
    portfolio = Portfolio(starting_cash=200)

    orders = []
    for index, price in enumerate([10, 11, 12]):
        orders.extend(strategy.on_bar(bar(price, index, symbol="OTHER"), portfolio))

    assert orders == []


def test_confirmed_leader_trend_enters_with_exact_bounded_notional():
    strategy = AiOligopolyLeaders(
        fast=2,
        slow=3,
        momentum_lookback=2,
        min_momentum_pct=5,
        leaders="NVDA,MSFT",
        target_notional=20,
    )
    portfolio = Portfolio(starting_cash=200)

    assert strategy.on_bar(bar(10, 0), portfolio) == []
    assert strategy.on_bar(bar(11, 1), portfolio) == []
    orders = strategy.on_bar(bar(12, 2), portfolio)

    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].symbol == "NVDA"
    assert orders[0].quantity * 12 == pytest.approx(20)


def test_entry_never_exceeds_available_cash():
    strategy = AiOligopolyLeaders(
        fast=2,
        slow=3,
        momentum_lookback=2,
        min_momentum_pct=5,
        leaders="NVDA",
        target_notional=20,
    )
    portfolio = Portfolio(starting_cash=15)
    strategy.on_bar(bar(10, 0), portfolio)
    strategy.on_bar(bar(11, 1), portfolio)

    order = strategy.on_bar(bar(12, 2), portfolio)[0]
    assert order.quantity * 12 == pytest.approx(15)


def test_trend_break_exits_entire_position_without_adding_to_winner():
    strategy = AiOligopolyLeaders(
        fast=2,
        slow=3,
        momentum_lookback=2,
        min_momentum_pct=5,
        leaders="NVDA",
        target_notional=20,
    )
    portfolio = Portfolio(starting_cash=200)
    strategy.on_bar(bar(10, 0), portfolio)
    strategy.on_bar(bar(11, 1), portfolio)
    buy = strategy.on_bar(bar(12, 2), portfolio)[0]
    portfolio.fill_at(buy, price=12, timestamp=bar(12, 3).timestamp)

    assert strategy.on_bar(bar(13, 3), portfolio) == []
    sell = strategy.on_bar(bar(8, 4), portfolio)

    assert len(sell) == 1
    assert sell[0].side is Side.SELL
    assert sell[0].quantity == pytest.approx(buy.quantity)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast": 3, "slow": 3},
        {"fast": 0},
        {"fast": 2.5},
        {"momentum_lookback": 0},
        {"momentum_lookback": 2.5},
        {"target_notional": 0},
        {"leaders": ""},
    ],
)
def test_invalid_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        AiOligopolyLeaders(**kwargs)
