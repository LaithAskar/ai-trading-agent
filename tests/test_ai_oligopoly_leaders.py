from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast
from unittest.mock import patch

import pytest

from strategies.ai_oligopoly_leaders import AiOligopolyLeaders
from trading_agent.autonomous import (
    DEFAULT_PARAMS,
    DEFAULT_STRATEGIES,
    RESEARCH_ONLY_STRATEGIES,
    run_autonomous_daily,
)
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.backtest.runner import load_strategy
from trading_agent.broker.alpaca import AlpacaPaperBroker
from trading_agent.broker.paper_runner import paper_tick
from trading_agent.core.events import Bar
from trading_agent.core.orders import Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.experiment import ExperimentContract
from trading_agent.experiment_graph import ExperimentGraph


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


def test_fractional_order_fills_next_bar_and_target_is_not_a_hard_fill_cap():
    strategy = load_strategy(
        "ai_oligopoly_leaders",
        {
            "fast": 2,
            "slow": 3,
            "momentum_lookback": 2,
            "min_momentum_pct": 5,
            "leaders": "NVDA",
            "target_notional": 20,
        },
    )
    bars = [bar(10, 0), bar(11, 1), bar(12, 2), bar(25, 3)]

    result = BacktestEngine(starting_cash=200, slippage_bps=5).run(
        strategy, "NVDA", bars
    )

    assert len(result.portfolio.fills) == 1
    fill = result.portfolio.fills[0]
    assert fill.quantity == pytest.approx(20 / 12)
    assert fill.price == pytest.approx(25 * 1.0005)
    assert fill.quantity * fill.price > 40


def test_strategy_is_registered_for_explicit_research_but_not_active_soak_default():
    assert "ai_oligopoly_leaders" in DEFAULT_PARAMS
    assert "ai_oligopoly_leaders" in RESEARCH_ONLY_STRATEGIES
    assert DEFAULT_STRATEGIES == ["sma_cross", "rsi_mean_rev"]


def test_explicit_strategy_execution_fails_closed_before_backtesting(tmp_path):
    with pytest.raises(ValueError, match="research-only strategies cannot execute"):
        run_autonomous_daily(
            contract=ExperimentContract(name="research-only-test"),
            symbols=["NVDA"],
            strategies=["ai_oligopoly_leaders"],
            execute=True,
            broker=object(),
            graph=ExperimentGraph(tmp_path / "graph.sqlite3"),
        )


def test_direct_paper_tick_execution_fails_closed_before_data_or_broker_access():
    with patch("trading_agent.broker.paper_runner.load_bars") as load_bars:
        result = paper_tick(
            strategy_name="ai_oligopoly_leaders",
            symbol="NVDA",
            dry_run=False,
            broker=cast(AlpacaPaperBroker, object()),
            contract=ExperimentContract(name="direct-paper-boundary-test"),
        )

    load_bars.assert_not_called()
    assert result.proposed_orders == []
    assert result.submitted == []
    assert result.dry_run is False
    assert result.skipped_reason is not None
    assert "research-only" in result.skipped_reason


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
