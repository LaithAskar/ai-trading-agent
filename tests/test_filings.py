"""Tests for the EDGAR/filings stack.

The data layer is integration-tested manually (it hits the SEC). Here we
test the pieces we control: sentiment scoring, the signal logic, and the
strategy interface contract under a stubbed score history.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from strategies.filings_sentiment import FilingsSentiment, _score_text
from trading_agent.core.events import Bar
from trading_agent.core.orders import Side
from trading_agent.core.portfolio import Portfolio


def _bar(ts: datetime, close: float = 100.0) -> Bar:
    return Bar(symbol="TEST", timestamp=ts, open=close, high=close, low=close, close=close, volume=1_000_000)


def test_score_text_positive_when_positive_words_dominate():
    text = "Revenue growth was strong this quarter, with momentum across all segments. Margins improved."
    assert _score_text(text) > 0.3


def test_score_text_negative_when_negative_words_dominate():
    text = "Significant headwinds led to a decline in margins. We missed guidance, with weak demand."
    assert _score_text(text) < -0.3


def test_score_text_neutral_empty():
    assert _score_text("") == 0.0
    assert _score_text("the quick brown fox") == 0.0


def test_strategy_emits_no_signal_with_fewer_than_two_filings():
    s = FilingsSentiment()
    s._scores = [(datetime(2023, 5, 1), -0.3)]
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2024, 1, 1)), pf)
    assert orders == []


def test_strategy_goes_long_on_improving_sentiment():
    s = FilingsSentiment(threshold=0.0)
    s._scores = [
        (datetime(2023, 5, 1), -0.40),
        (datetime(2023, 8, 1), -0.25),  # improved
    ]
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2023, 9, 1), close=50.0), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 2000  # 100k / 50


def test_strategy_exits_when_sentiment_deteriorates():
    s = FilingsSentiment(threshold=0.0)
    s._scores = [
        (datetime(2023, 5, 1), -0.25),
        (datetime(2023, 8, 1), -0.40),  # deteriorated
    ]
    pf = Portfolio(starting_cash=100_000.0)
    pf.positions["TEST"] = 100  # already long
    orders = s.on_bar(_bar(datetime(2023, 9, 1)), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].quantity == 100


def test_strategy_holds_when_no_change_below_threshold():
    s = FilingsSentiment(threshold=0.10)
    s._scores = [
        (datetime(2023, 5, 1), -0.30),
        (datetime(2023, 8, 1), -0.25),  # delta=0.05, below threshold of 0.10
    ]
    pf = Portfolio(starting_cash=100_000.0)
    pf.positions["TEST"] = 100
    orders = s.on_bar(_bar(datetime(2023, 9, 1)), pf)
    # Already long and signal doesn't flip => no new orders
    assert orders == []


def test_strategy_uses_most_recent_two_filings_before_bar_date():
    s = FilingsSentiment(threshold=0.0)
    s._scores = [
        (datetime(2023, 5, 1), -0.50),
        (datetime(2023, 8, 1), -0.20),  # improved (this would be signal)
        (datetime(2024, 2, 1), -0.30),  # but this comes AFTER our bar
    ]
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2023, 11, 1), close=50.0), pf)
    # Should see the improvement from -0.50 to -0.20 and go long
    assert len(orders) == 1
    assert orders[0].side is Side.BUY


def test_strategy_rejects_bad_params():
    with pytest.raises(ValueError):
        FilingsSentiment(threshold=2.0)
    with pytest.raises(ValueError):
        FilingsSentiment(form="not-a-form")
