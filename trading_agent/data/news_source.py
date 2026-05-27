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
            "sort": "EARLIEST",
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
    those. If not, fetches from AlphaVantage, aggregates by day, persists.

    start / end: YYYY-MM-DD.
    """
    if use_cache:
        cached = _load_cached(ticker, start, end, db_path=db_path)
        if cached:
            return cached

    if not api_key:
        cfg = Config.load()
        api_key = cfg.alphavantage_api_key
    if not api_key:
        raise RuntimeError(
            "AlphaVantage API key required. Set ALPHAVANTAGE_API_KEY in your .env "
            "(get a free key at https://www.alphavantage.co/support/#api-key)."
        )

    time_from = start.replace("-", "") + "T0000"
    time_to = end.replace("-", "") + "T2359"
    articles = _fetch_av_news(ticker, time_from, time_to, api_key)
    rows = _aggregate_by_day(ticker, articles)
    _save_cached(rows, db_path=db_path)
    return rows
