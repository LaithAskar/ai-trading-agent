"""News sentiment data via AlphaVantage NEWS_SENTIMENT.

Returns a daily-aggregated time series of ticker-specific sentiment scores,
cached in SQLite so backtests are reproducible and don't burn API quota.

Free tier:    500 requests/day, 5 requests/minute, ~2 years of history.
Endpoint:     https://www.alphavantage.co/query?function=NEWS_SENTIMENT
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests

from ..config import DATA_DIR, Config


CACHE_DB = DATA_DIR / "news_cache.sqlite3"
AV_BASE = "https://www.alphavantage.co/query"


SCHEMA = """
CREATE TABLE IF NOT EXISTS news_daily (
    ticker        TEXT NOT NULL,
    date          TEXT NOT NULL,
    num_articles  INTEGER NOT NULL,
    avg_sentiment REAL NOT NULL,
    cached_at     TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_daily(ticker);

CREATE TABLE IF NOT EXISTS news_articles (
    ticker            TEXT NOT NULL,
    url               TEXT NOT NULL,
    time_published    TEXT NOT NULL,   -- AV raw: YYYYMMDDTHHMMSS
    date              TEXT NOT NULL,   -- YYYY-MM-DD (derived, for range queries)
    title             TEXT NOT NULL,
    summary           TEXT NOT NULL,
    source            TEXT NOT NULL,
    overall_sentiment REAL NOT NULL,   -- article-level tone (all tickers)
    ticker_sentiment  REAL NOT NULL,   -- THIS ticker's sentiment score [-1, 1]
    relevance         REAL NOT NULL,   -- THIS ticker's relevance [0, 1]
    cached_at         TEXT NOT NULL,
    PRIMARY KEY (ticker, url)
);
CREATE INDEX IF NOT EXISTS idx_articles_ticker_date ON news_articles(ticker, date);
"""


@contextmanager
def _conn(db_path: Path = CACHE_DB) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


@dataclass(frozen=True)
class DailySentiment:
    ticker: str
    date: str          # YYYY-MM-DD
    num_articles: int
    avg_sentiment: float


@dataclass(frozen=True)
class Article:
    """One news headline, as it relates to a specific ticker.

    `ticker_sentiment` / `relevance` are AlphaVantage's per-ticker scores for
    THIS article (an article can mention many tickers with different scores).
    `time_published` keeps AV's full intraday timestamp so a strategy can map a
    headline to the correct tradeable bar; `date` is the derived day for range
    queries.
    """

    ticker: str
    url: str
    time_published: str  # YYYYMMDDTHHMMSS (AV raw)
    date: str            # YYYY-MM-DD
    title: str
    summary: str
    source: str
    overall_sentiment: float
    ticker_sentiment: float
    relevance: float


def _fetch_av_news(ticker: str, time_from: str, time_to: str, api_key: str) -> list[dict]:
    """Hit AlphaVantage's NEWS_SENTIMENT endpoint and return the `feed` array.

    AV's time_from / time_to format is YYYYMMDDTHHMM (e.g., 20240101T0000).
    """
    resp = requests.get(
        AV_BASE,
        params={
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "time_from": time_from,
            "time_to": time_to,
            "limit": 1000,
            "sort": "LATEST",
            "apikey": api_key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "Note" in data:
        raise RuntimeError(f"AlphaVantage rate-limited: {data['Note']}")
    if "Information" in data and "feed" not in data:
        raise RuntimeError(f"AlphaVantage refused: {data['Information']}")
    if "feed" not in data:
        return []
    return data["feed"]


def _aggregate_by_day(ticker: str, articles: list[dict]) -> list[DailySentiment]:
    """Group AV articles by date (UTC) and compute mean ticker-specific sentiment.

    Each article has a `ticker_sentiment` array; we look up the row for our
    ticker and average its `ticker_sentiment_score`.
    """
    by_date: dict[str, list[float]] = {}
    for art in articles:
        time_published = art.get("time_published", "")
        if len(time_published) < 8:
            continue
        date_str = f"{time_published[:4]}-{time_published[4:6]}-{time_published[6:8]}"

        ticker_scores = art.get("ticker_sentiment", [])
        score = None
        for ts in ticker_scores:
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    score = float(ts.get("ticker_sentiment_score", 0))
                except (TypeError, ValueError):
                    score = None
                break
        if score is None:
            continue
        by_date.setdefault(date_str, []).append(score)

    return [
        DailySentiment(
            ticker=ticker.upper(),
            date=d,
            num_articles=len(scores),
            avg_sentiment=sum(scores) / len(scores) if scores else 0.0,
        )
        for d, scores in sorted(by_date.items())
    ]


def _extract_articles(ticker: str, articles: list[dict]) -> list[Article]:
    """Parse the raw AV feed into per-ticker Article rows.

    Skips articles that don't carry a sentiment entry for our ticker (same
    rule as the day-aggregator) or that lack a usable timestamp/url.
    """
    out: list[Article] = []
    for art in articles:
        time_published = art.get("time_published", "")
        if len(time_published) < 8:
            continue
        url = art.get("url", "")
        if not url:
            continue
        date_str = f"{time_published[:4]}-{time_published[4:6]}-{time_published[6:8]}"

        ticker_score = None
        relevance = 0.0
        for ts in art.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    ticker_score = float(ts.get("ticker_sentiment_score", 0))
                except (TypeError, ValueError):
                    ticker_score = None
                try:
                    relevance = float(ts.get("relevance_score", 0))
                except (TypeError, ValueError):
                    relevance = 0.0
                break
        if ticker_score is None:
            continue

        try:
            overall = float(art.get("overall_sentiment_score", 0))
        except (TypeError, ValueError):
            overall = 0.0

        out.append(
            Article(
                ticker=ticker.upper(),
                url=url,
                time_published=time_published,
                date=date_str,
                title=art.get("title", ""),
                summary=art.get("summary", ""),
                source=art.get("source", ""),
                overall_sentiment=overall,
                ticker_sentiment=ticker_score,
                relevance=relevance,
            )
        )
    return out


def _save_articles(rows: list[Article], db_path: Path = CACHE_DB) -> None:
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn(db_path) as c:
        c.executemany(
            """
            INSERT OR REPLACE INTO news_articles
            (ticker, url, time_published, date, title, summary, source,
             overall_sentiment, ticker_sentiment, relevance, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.ticker, r.url, r.time_published, r.date, r.title, r.summary,
                    r.source, r.overall_sentiment, r.ticker_sentiment, r.relevance, now,
                )
                for r in rows
            ],
        )


def _load_articles(
    ticker: str,
    start: str,
    end: str,
    *,
    min_relevance: float = 0.0,
    limit: int | None = None,
    db_path: Path = CACHE_DB,
) -> list[Article]:
    sql = """
        SELECT ticker, url, time_published, date, title, summary, source,
               overall_sentiment, ticker_sentiment, relevance
        FROM news_articles
        WHERE ticker = ? AND date >= ? AND date <= ? AND relevance >= ?
        ORDER BY time_published ASC
    """
    params: list = [ticker.upper(), start, end, min_relevance]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _conn(db_path) as c:
        rows = c.execute(sql, params).fetchall()
    return [
        Article(
            ticker=r["ticker"],
            url=r["url"],
            time_published=r["time_published"],
            date=r["date"],
            title=r["title"],
            summary=r["summary"],
            source=r["source"],
            overall_sentiment=r["overall_sentiment"],
            ticker_sentiment=r["ticker_sentiment"],
            relevance=r["relevance"],
        )
        for r in rows
    ]


def _save_cached(rows: list[DailySentiment], db_path: Path = CACHE_DB) -> None:
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn(db_path) as c:
        c.executemany(
            """
            INSERT OR REPLACE INTO news_daily
            (ticker, date, num_articles, avg_sentiment, cached_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(r.ticker, r.date, r.num_articles, r.avg_sentiment, now) for r in rows],
        )


def _load_cached(
    ticker: str, start: str, end: str, db_path: Path = CACHE_DB
) -> list[DailySentiment]:
    with _conn(db_path) as c:
        rows = c.execute(
            """
            SELECT ticker, date, num_articles, avg_sentiment
            FROM news_daily
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
            """,
            (ticker.upper(), start, end),
        ).fetchall()
    return [
        DailySentiment(
            ticker=r["ticker"],
            date=r["date"],
            num_articles=r["num_articles"],
            avg_sentiment=r["avg_sentiment"],
        )
        for r in rows
    ]


def _resolve_api_key(api_key: str | None) -> str:
    if not api_key:
        api_key = Config.load().alphavantage_api_key
    if not api_key:
        raise RuntimeError(
            "AlphaVantage API key required. Set ALPHAVANTAGE_API_KEY in your .env "
            "(get a free key at https://www.alphavantage.co/support/#api-key)."
        )
    return api_key


def _fetch_and_persist(
    ticker: str, start: str, end: str, api_key: str, db_path: Path
) -> tuple[list[DailySentiment], list[Article]]:
    """One AV call → persist BOTH the daily aggregate and the raw articles.

    A single NEWS_SENTIMENT response feeds both caches, so we never pay the
    quota twice for the same window.
    """
    time_from = start.replace("-", "") + "T0000"
    time_to = end.replace("-", "") + "T2359"
    feed = _fetch_av_news(ticker, time_from, time_to, api_key)
    daily = _aggregate_by_day(ticker, feed)
    articles = _extract_articles(ticker, feed)
    _save_cached(daily, db_path=db_path)
    _save_articles(articles, db_path=db_path)
    return daily, articles


def get_daily_sentiment(
    ticker: str,
    start: str,
    end: str,
    *,
    api_key: str | None = None,
    use_cache: bool = True,
    db_path: Path = CACHE_DB,
) -> list[DailySentiment]:
    """Return the daily-aggregated ticker sentiment series for [start, end].

    Tries the cache first. If the cache has any rows in the window, returns
    those. If not, fetches from AlphaVantage, aggregates by day, and persists
    both the daily series and the raw headlines (see `get_headlines`).

    start / end: YYYY-MM-DD.
    """
    if use_cache:
        cached = _load_cached(ticker, start, end, db_path=db_path)
        if cached:
            return cached

    api_key = _resolve_api_key(api_key)
    daily, _ = _fetch_and_persist(ticker, start, end, api_key, db_path)
    return daily


def get_headlines(
    ticker: str,
    start: str,
    end: str,
    *,
    min_relevance: float = 0.0,
    limit: int | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    db_path: Path = CACHE_DB,
) -> list[Article]:
    """Return the raw per-ticker news headlines for [start, end].

    Same cache-first contract as `get_daily_sentiment`: a cache hit returns
    persisted rows; a miss does one AV call that fills both caches. Sorted by
    `time_published` ascending.

    min_relevance: drop articles below this AV per-ticker relevance score.
    limit:         cap the number of rows returned (after relevance filtering).
    start / end:   YYYY-MM-DD.
    """
    if use_cache:
        cached = _load_articles(
            ticker, start, end, min_relevance=min_relevance, limit=limit, db_path=db_path
        )
        if cached:
            return cached

    api_key = _resolve_api_key(api_key)
    _, articles = _fetch_and_persist(ticker, start, end, api_key, db_path)
    articles = [a for a in articles if a.relevance >= min_relevance]
    articles.sort(key=lambda a: a.time_published)
    if limit is not None:
        articles = articles[:limit]
    return articles
