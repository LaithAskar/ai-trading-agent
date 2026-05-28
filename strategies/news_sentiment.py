from __future__ import annotations

from collections import deque
from datetime import datetime

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


class NewsSentiment(Strategy):
    """Long when the rolling N-day average of news sentiment is above a threshold,
    flat when it drops below the lower threshold. Uses AlphaVantage's
    NEWS_SENTIMENT endpoint (ticker-specific sentiment scores per article,
    aggregated to a daily mean in the data source).

    Hysteresis band prevents whipsaw on small noise: enter long only when the
    rolling mean exceeds `enter_threshold`; exit long only when it drops below
    `exit_threshold`. By default 0.15 / 0.05 — comfortably above the noise
    floor of AlphaVantage's [-1, 1] score range.
    """

    name = "news_sentiment"

    def __init__(
        self,
        window: int = 5,
        enter_threshold: float = 0.15,
        exit_threshold: float = 0.05,
        min_articles_per_day: int = 1,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if exit_threshold > enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        if not -1 <= exit_threshold <= 1 or not -1 <= enter_threshold <= 1:
            raise ValueError("thresholds must be in [-1, 1]")
        self.window = window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_articles_per_day = min_articles_per_day
        # Sorted ascending by date; (date, avg_sentiment, num_articles)
        self._series: list[tuple[datetime, float, int]] = []
        self._scored_dates: dict[datetime.date, float] = {}

    def on_start(self, symbols: list[str]) -> None:
        from datetime import date, timedelta

        from trading_agent.data.news_source import get_daily_sentiment

        symbol = symbols[0]
        # AV's free tier caps at 1000 articles per call. For most tickers
        # that covers roughly the last 18-24 months. Fetching a wider window
        # would hit the cap and return only the OLDEST 1000 (useless for
        # backtests of recent windows). So we pin to "recent past" relative
        # to wall-clock today.
        today = date.today()
        fetch_start = (today - timedelta(days=600)).strftime("%Y-%m-%d")
        fetch_end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            rows = get_daily_sentiment(symbol, fetch_start, fetch_end)
        except Exception:
            # Offline / no API key / rate limited: stay flat for the run.
            return

        for r in rows:
            if r.num_articles < self.min_articles_per_day:
                continue
            d = datetime.strptime(r.date, "%Y-%m-%d")
            self._series.append((d, r.avg_sentiment, r.num_articles))
            self._scored_dates[d.date()] = r.avg_sentiment

        self._series.sort(key=lambda x: x[0])

    def _rolling_mean_as_of(self, ts: datetime) -> float | None:
        """Mean of the last `window` sentiment scores with date <= ts.

        Returns None if we don't have at least `window` data points yet.
        """
        recent = [s for d, s, _ in self._series if d.date() <= ts.date()]
        if len(recent) < self.window:
            return None
        return sum(recent[-self.window:]) / self.window

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        rolling = self._rolling_mean_as_of(bar.timestamp)
        if rolling is None:
            return []

        held = portfolio.position(bar.symbol)

        if rolling > self.enter_threshold and held == 0:
            qty = int(portfolio.cash // bar.close)
            if qty > 0:
                return [Order(symbol=bar.symbol, side=Side.BUY, quantity=qty)]
        elif rolling < self.exit_threshold and held > 0:
            return [Order(symbol=bar.symbol, side=Side.SELL, quantity=held)]
        return []
