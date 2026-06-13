"""Tests for raw news-headline capture (Article rows) and the agent tool.

We never hit AlphaVantage here. We verify:
  1. `_extract_articles` parses the raw AV feed into per-ticker Article rows,
     picking THIS ticker's sentiment/relevance and skipping junk.
  2. The `news_articles` SQLite cache round-trips, filters by date range and
     min_relevance, and honors `limit`.
  3. The `get_news_headlines` agent tool shapes the data source output.
"""
from __future__ import annotations

import pytest

from trading_agent.data.news_source import (
    Article,
    _extract_articles,
    _load_articles,
    _save_articles,
)


def _make_av_article(
    ticker: str,
    score: float,
    time_published: str,
    *,
    relevance: float = 0.5,
    url: str | None = None,
    title: str = "Headline",
) -> dict:
    return {
        "url": url if url is not None else f"https://news.example/{time_published}",
        "time_published": time_published,
        "title": title,
        "summary": "Some summary text.",
        "source": "Example Wire",
        "overall_sentiment_score": "0.12",
        "ticker_sentiment": [
            {
                "ticker": ticker,
                "ticker_sentiment_score": str(score),
                "relevance_score": str(relevance),
            },
            {"ticker": "OTHER", "ticker_sentiment_score": "0.99", "relevance_score": "0.9"},
        ],
    }


# ---------- _extract_articles ----------

def test_extract_picks_target_ticker_fields():
    feed = [_make_av_article("AAPL", 0.30, "20240101T103000", relevance=0.7, title="Apple beats")]
    arts = _extract_articles("AAPL", feed)
    assert len(arts) == 1
    a = arts[0]
    assert a.ticker == "AAPL"
    assert a.title == "Apple beats"
    assert a.source == "Example Wire"
    assert a.ticker_sentiment == pytest.approx(0.30)
    assert a.relevance == pytest.approx(0.7)
    assert a.overall_sentiment == pytest.approx(0.12)
    assert a.date == "2024-01-01"
    assert a.time_published == "20240101T103000"


def test_extract_skips_articles_missing_target_ticker():
    feed = [
        {
            "url": "https://news.example/x",
            "time_published": "20240101T0000",
            "ticker_sentiment": [
                {"ticker": "MSFT", "ticker_sentiment_score": "0.5", "relevance_score": "0.5"}
            ],
        },
        _make_av_article("AAPL", 0.20, "20240101T1000"),
    ]
    arts = _extract_articles("AAPL", feed)
    assert len(arts) == 1
    assert arts[0].ticker_sentiment == pytest.approx(0.20)


def test_extract_skips_articles_missing_url_or_timestamp():
    feed = [
        {"time_published": "20240101T1000", "ticker_sentiment": []},  # no url
        {"url": "https://news.example/y", "ticker_sentiment": []},     # no timestamp
        _make_av_article("AAPL", 0.40, "20240105T1000"),
    ]
    arts = _extract_articles("AAPL", feed)
    assert len(arts) == 1
    assert arts[0].date == "2024-01-05"


# ---------- cache ----------

def _art(url: str, date: str, *, relevance: float = 0.5, score: float = 0.1) -> Article:
    return Article(
        ticker="AAPL",
        url=url,
        time_published=date.replace("-", "") + "T120000",
        date=date,
        title="t",
        summary="s",
        source="src",
        overall_sentiment=0.0,
        ticker_sentiment=score,
        relevance=relevance,
    )


def test_article_cache_roundtrip(tmp_path):
    db = tmp_path / "news.sqlite3"
    _save_articles([_art("u1", "2024-01-01"), _art("u2", "2024-01-02")], db_path=db)
    loaded = _load_articles("AAPL", "2024-01-01", "2024-01-31", db_path=db)
    assert [a.url for a in loaded] == ["u1", "u2"]  # sorted by time_published


def test_article_cache_filters_date_range(tmp_path):
    db = tmp_path / "news.sqlite3"
    _save_articles(
        [_art("u1", "2024-01-01"), _art("u2", "2024-06-15"), _art("u3", "2024-12-31")],
        db_path=db,
    )
    loaded = _load_articles("AAPL", "2024-06-01", "2024-06-30", db_path=db)
    assert [a.url for a in loaded] == ["u2"]


def test_article_cache_filters_min_relevance(tmp_path):
    db = tmp_path / "news.sqlite3"
    _save_articles(
        [_art("lo", "2024-01-01", relevance=0.1), _art("hi", "2024-01-02", relevance=0.8)],
        db_path=db,
    )
    loaded = _load_articles("AAPL", "2024-01-01", "2024-01-31", min_relevance=0.5, db_path=db)
    assert [a.url for a in loaded] == ["hi"]


def test_article_cache_honors_limit(tmp_path):
    db = tmp_path / "news.sqlite3"
    _save_articles([_art(f"u{i}", f"2024-01-0{i}") for i in range(1, 6)], db_path=db)
    loaded = _load_articles("AAPL", "2024-01-01", "2024-01-31", limit=2, db_path=db)
    assert len(loaded) == 2
    assert [a.url for a in loaded] == ["u1", "u2"]


def test_article_upsert_dedups_on_url(tmp_path):
    db = tmp_path / "news.sqlite3"
    _save_articles([_art("dup", "2024-01-01", score=0.1)], db_path=db)
    _save_articles([_art("dup", "2024-01-01", score=0.9)], db_path=db)  # same PK
    loaded = _load_articles("AAPL", "2024-01-01", "2024-01-31", db_path=db)
    assert len(loaded) == 1
    assert loaded[0].ticker_sentiment == pytest.approx(0.9)  # latest write wins


# ---------- agent tool ----------

def test_news_headlines_tool_shapes_output(monkeypatch):
    from trading_agent.agent import tools

    monkeypatch.setattr(
        "trading_agent.data.news_source.get_headlines",
        lambda **kw: [_art("u1", "2024-01-01", relevance=0.6, score=0.25)],
    )
    out = tools._get_news_headlines_tool(
        {"ticker": "aapl", "start": "2024-01-01", "end": "2024-01-31"}
    )
    assert out["ticker"] == "AAPL"
    assert out["count"] == 1
    h = out["headlines"][0]
    assert h["relevance"] == pytest.approx(0.6)
    assert h["ticker_sentiment"] == pytest.approx(0.25)
    assert h["url"] == "u1"


def test_news_headlines_tool_handles_empty(monkeypatch):
    from trading_agent.agent import tools

    monkeypatch.setattr(
        "trading_agent.data.news_source.get_headlines", lambda **kw: []
    )
    out = tools._get_news_headlines_tool(
        {"ticker": "aapl", "start": "2024-01-01", "end": "2024-01-31"}
    )
    assert out["count"] == 0
    assert "no headlines" in out["summary"]


def test_news_headlines_tool_registered():
    from trading_agent.agent.tools import ALL_TOOLS

    names = {t.name for t in ALL_TOOLS}
    assert "get_news_headlines" in names
