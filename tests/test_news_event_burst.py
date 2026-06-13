"""Tests for the NewsEventBurst strategy.

We bypass on_start (no AlphaVantage) and inject a synthetic event series
[(date, ticker_sentiment)] of articles already past the relevance floor, then
assert the burst-count entry/exit logic.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from strategies.news_event_burst import NewsEventBurst
from trading_agent.core.events import Bar
from trading_agent.core.orders import Side
from trading_agent.core.portfolio import Portfolio


def _bar(ts: datetime, close: float = 100.0) -> Bar:
    return Bar(
        symbol="TEST", timestamp=ts, open=close, high=close, low=close,
        close=close, volume=1_000_000,
    )


def _inject(s: NewsEventBurst, events: list[tuple[date, float]]) -> None:
    s._events = sorted(events, key=lambda e: e[0])


def test_rejects_invalid_params():
    with pytest.raises(ValueError):
        NewsEventBurst(relevance_floor=1.5)
    with pytest.raises(ValueError):
        NewsEventBurst(score_threshold=-0.1)
    with pytest.raises(ValueError):
        NewsEventBurst(burst_count=0)
    with pytest.raises(ValueError):
        NewsEventBurst(burst_window=0)
    with pytest.raises(ValueError):
        NewsEventBurst(hold_days=0)
    with pytest.raises(ValueError):
        NewsEventBurst(cash_buffer_pct=1.0)


def test_entry_reserves_cash_buffer():
    """Sizing must leave headroom so a next-bar gap-up fill isn't rejected.

    With all-in sizing (cash // close) the order notional == cash and any
    upward gap or slippage on the next-bar fill pushes it over the cash on
    hand. The buffer keeps the BUY quantity strictly below the all-in count.
    """
    s = NewsEventBurst(score_threshold=0.15, burst_count=3, burst_window=3, cash_buffer_pct=0.01)
    _inject(s, [(date(2024, 1, 1), 0.2), (date(2024, 1, 2), 0.3), (date(2024, 1, 3), 0.25)])
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2024, 1, 3), close=100.0), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 990   # int(100000 * 0.99 // 100), not the all-in 1000


def test_enters_on_positive_burst():
    s = NewsEventBurst(score_threshold=0.15, burst_count=3, burst_window=3)
    _inject(
        s,
        [
            (date(2024, 1, 1), 0.20),
            (date(2024, 1, 2), 0.30),
            (date(2024, 1, 3), 0.25),  # 3 positive, relevant articles in window
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2024, 1, 3), close=50.0), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.BUY


def test_no_entry_below_burst_count():
    s = NewsEventBurst(score_threshold=0.15, burst_count=3, burst_window=3)
    _inject(s, [(date(2024, 1, 2), 0.30), (date(2024, 1, 3), 0.25)])  # only 2
    pf = Portfolio(starting_cash=100_000.0)
    assert s.on_bar(_bar(datetime(2024, 1, 3)), pf) == []


def test_articles_outside_window_not_counted():
    s = NewsEventBurst(score_threshold=0.15, burst_count=3, burst_window=3)
    _inject(
        s,
        [
            (date(2024, 1, 1), 0.20),  # window is Jan 3-5, so this is too old
            (date(2024, 1, 4), 0.30),
            (date(2024, 1, 5), 0.25),
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    # As of Jan 5: window = Jan 3..5 → only 2 positives → no entry.
    assert s.on_bar(_bar(datetime(2024, 1, 5)), pf) == []


def test_future_articles_not_counted_lookahead():
    s = NewsEventBurst(score_threshold=0.15, burst_count=3, burst_window=5)
    _inject(
        s,
        [
            (date(2024, 1, 1), 0.20),
            (date(2024, 1, 2), 0.30),
            (date(2024, 1, 5), 0.90),  # AFTER the bar — must not be counted
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    # Bar is Jan 2: only 2 positives are visible → no burst yet.
    assert s.on_bar(_bar(datetime(2024, 1, 2)), pf) == []


def test_exits_on_negative_burst():
    s = NewsEventBurst(score_threshold=0.15, burst_count=2, burst_window=3, hold_days=99)
    _inject(
        s,
        [
            (date(2024, 1, 4), -0.30),
            (date(2024, 1, 5), -0.25),  # 2 negatives in window
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    pf.positions["TEST"] = 100
    orders = s.on_bar(_bar(datetime(2024, 1, 5)), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL


def test_exits_on_time_stop():
    s = NewsEventBurst(burst_count=2, hold_days=3)
    _inject(s, [])  # no news → only the time stop can fire
    pf = Portfolio(starting_cash=100_000.0)
    pf.positions["TEST"] = 100
    # Three held bars: counter reaches hold_days on the 3rd → SELL.
    assert s.on_bar(_bar(datetime(2024, 1, 1)), pf) == []
    assert s.on_bar(_bar(datetime(2024, 1, 2)), pf) == []
    orders = s.on_bar(_bar(datetime(2024, 1, 3)), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL


def test_holds_when_no_signal():
    s = NewsEventBurst(score_threshold=0.15, burst_count=3, burst_window=3, hold_days=99)
    _inject(s, [(date(2024, 1, 5), 0.20)])  # one mild positive, no burst either way
    pf = Portfolio(starting_cash=100_000.0)
    pf.positions["TEST"] = 100
    assert s.on_bar(_bar(datetime(2024, 1, 5)), pf) == []
