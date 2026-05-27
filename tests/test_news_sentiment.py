"""Tests for the news sentiment data source + strategy.

We don't hit AlphaVantage in tests. We verify:
  1. The day-aggregator correctly groups articles by date and averages
     ticker-specific scores.
  2. The SQLite cache round-trips and is read-back correctly.
  3. The strategy's rolling-mean + hysteresis logic emits the right signals
     given a stubbed sentiment series.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from strategies.news_sentiment import NewsSentiment
from trading_agent.core.events import Bar
from trading_agent.core.orders import Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.data.news_source import (
    DailySentiment,
    _aggregate_by_day,
    _load_cached,
    _save_cached,
)


# ---------- Aggregator ----------

def _make_av_article(ticker: str, score: float, time_published: str) -> dict:
    return {
        "time_published": time_published,
        "ticker_sentiment": [
            {"ticker": ticker, "ticker_sentiment_score": str(score)},
            {"ticker": "OTHER", "ticker_sentiment_score": "0.99"},
        ],
    }


def test_aggregate_groups_by_date_and_picks_target_ticker():
    articles = [
        _make_av_article("AAPL", 0.20, "20240101T1000"),
        _make_av_article("AAPL", 0.40, "20240101T1500"),
        _make_av_article("AAPL", -0.10, "20240102T0900"),
    ]
    daily = _aggregate_by_day("AAPL", articles)
    by_date = {d.date: d for d in daily}
    assert "2024-01-01" in by_date
    assert "2024-01-02" in by_date
    # Day 1: mean of 0.20 and 0.40
    assert by_date["2024-01-01"].avg_sentiment == pytest.approx(0.30)
    assert by_date["2024-01-01"].num_articles == 2
    # Day 2: single article
    assert by_date["2024-01-02"].avg_sentiment == pytest.approx(-0.10)


def test_aggregate_skips_articles_missing_target_ticker():
    articles = [
        {
            "time_published": "20240101T0000",
            "ticker_sentiment": [{"ticker": "MSFT", "ticker_sentiment_score": "0.5"}],
        },
        _make_av_article("AAPL", 0.20, "20240101T1000"),
    ]
    daily = _aggregate_by_day("AAPL", articles)
    assert len(daily) == 1
    assert daily[0].avg_sentiment == 0.20


def test_aggregate_ignores_garbage_articles():
    articles = [
        {"time_published": "20240101T0000"},  # no ticker_sentiment
        {"ticker_sentiment": []},              # no time_published
        _make_av_article("AAPL", 0.30, "20240105T1000"),
    ]
    daily = _aggregate_by_day("AAPL", articles)
    assert len(daily) == 1
    assert daily[0].date == "2024-01-05"


# ---------- Cache ----------

def test_cache_roundtrip(tmp_path):
    db = tmp_path / "news.sqlite3"
    rows = [
        DailySentiment("AAPL", "2024-01-01", 3, 0.25),
        DailySentiment("AAPL", "2024-01-02", 1, -0.10),
        DailySentiment("AAPL", "2024-01-03", 5, 0.40),
    ]
    _save_cached(rows, db_path=db)
    loaded = _load_cached("AAPL", "2024-01-01", "2024-01-31", db_path=db)
    assert len(loaded) == 3
    assert loaded[0].date == "2024-01-01"
    assert loaded[0].avg_sentiment == pytest.approx(0.25)
    assert loaded[2].num_articles == 5


def test_cache_filters_by_date_range(tmp_path):
    db = tmp_path / "news.sqlite3"
    _save_cached(
        [
            DailySentiment("AAPL", "2024-01-01", 1, 0.1),
            DailySentiment("AAPL", "2024-06-15", 1, 0.2),
            DailySentiment("AAPL", "2024-12-31", 1, 0.3),
        ],
        db_path=db,
    )
    loaded = _load_cached("AAPL", "2024-06-01", "2024-06-30", db_path=db)
    assert len(loaded) == 1
    assert loaded[0].date == "2024-06-15"


# ---------- Strategy ----------

def _bar(ts: datetime, close: float = 100.0) -> Bar:
    return Bar(symbol="TEST", timestamp=ts, open=close, high=close, low=close, close=close, volume=1_000_000)


def _stub_strategy_with_series(s: NewsSentiment, series: list[tuple[datetime, float]]):
    """Bypass on_start; inject a pre-built series."""
    s._series = [(d, score, 5) for d, score in series]
    s._scored_dates = {d.date(): score for d, score in series}


def test_strategy_holds_until_window_fills():
    s = NewsSentiment(window=5)
    _stub_strategy_with_series(
        s,
        [(datetime(2024, 1, i + 1), 0.5) for i in range(4)],  # only 4 < window=5
    )
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2024, 1, 10)), pf)
    assert orders == []


def test_strategy_goes_long_when_rolling_mean_above_enter_threshold():
    s = NewsSentiment(window=3, enter_threshold=0.15, exit_threshold=0.05)
    _stub_strategy_with_series(
        s,
        [
            (datetime(2024, 1, 1), 0.10),
            (datetime(2024, 1, 2), 0.30),
            (datetime(2024, 1, 3), 0.40),  # rolling mean = 0.267 > 0.15
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    orders = s.on_bar(_bar(datetime(2024, 1, 4), close=50.0), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.BUY


def test_strategy_exits_when_rolling_mean_below_exit_threshold():
    s = NewsSentiment(window=3, enter_threshold=0.15, exit_threshold=0.05)
    _stub_strategy_with_series(
        s,
        [
            (datetime(2024, 1, 1), -0.10),
            (datetime(2024, 1, 2), -0.20),
            (datetime(2024, 1, 3), -0.30),  # rolling mean = -0.20 < 0.05
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    pf.positions["TEST"] = 100
    orders = s.on_bar(_bar(datetime(2024, 1, 4)), pf)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL


def test_strategy_holds_inside_hysteresis_band():
    s = NewsSentiment(window=3, enter_threshold=0.15, exit_threshold=0.05)
    # rolling mean in the band [0.05, 0.15]: should not enter, should not exit
    _stub_strategy_with_series(
        s,
        [
            (datetime(2024, 1, 1), 0.08),
            (datetime(2024, 1, 2), 0.10),
            (datetime(2024, 1, 3), 0.12),  # mean = 0.10
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    # No position, no entry
    assert s.on_bar(_bar(datetime(2024, 1, 4)), pf) == []
    # With a position, no exit
    pf.positions["TEST"] = 100
    assert s.on_bar(_bar(datetime(2024, 1, 4)), pf) == []


def test_strategy_uses_only_sentiment_with_date_le_bar_date():
    """Lookahead safety: rolling mean must not include future sentiment scores."""
    s = NewsSentiment(window=2, enter_threshold=0.10, exit_threshold=0.05)
    _stub_strategy_with_series(
        s,
        [
            (datetime(2024, 1, 1), -0.50),
            (datetime(2024, 1, 2), -0.50),
            (datetime(2024, 1, 3), 0.90),   # this is AFTER our bar — must NOT be used
        ],
    )
    pf = Portfolio(starting_cash=100_000.0)
    # Bar is Jan 2 — only the first two sentiment days should contribute.
    # Rolling mean (window=2) over [-0.50, -0.50] = -0.50 → flat.
    orders = s.on_bar(_bar(datetime(2024, 1, 2)), pf)
    assert orders == []


def test_strategy_rejects_invalid_params():
    with pytest.raises(ValueError):
        NewsSentiment(window=0)
    with pytest.raises(ValueError):
        NewsSentiment(enter_threshold=0.05, exit_threshold=0.10)
    with pytest.raises(ValueError):
        NewsSentiment(enter_threshold=2.0)
