from __future__ import annotations

from datetime import date, datetime, timedelta

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


class NewsEventBurst(Strategy):
    """Trade on *bursts* of high-relevance news rather than a daily sentiment mean.

    `news_sentiment` averages every article's score into one daily number, which
    washes out a single high-conviction story among low-relevance noise. This
    strategy instead reads the raw headlines (via news_source.get_headlines),
    keeps only articles whose AlphaVantage per-ticker `relevance` clears
    `relevance_floor`, and counts how many *positive* (score >= score_threshold)
    such articles landed in the last `burst_window` days.

    - ENTER long when that count >= `burst_count` and we're flat — a cluster of
      relevant, favorable coverage.
    - EXIT when either a symmetric NEGATIVE burst appears (>= burst_count
      relevant articles with score <= -score_threshold) or we've held for
      `hold_days` bars (time stop), whichever comes first.

    Lookahead safety: a decision at bar t only sees articles dated <= t, and the
    resulting order fills at t+1's open (engine convention). AV timestamps are
    intraday; we map each article to its calendar day, matching `news_sentiment`.
    """

    name = "news_event_burst"

    def __init__(
        self,
        relevance_floor: float = 0.3,
        score_threshold: float = 0.15,
        burst_count: int = 3,
        burst_window: int = 3,
        hold_days: int = 5,
        lookback_days: int = 600,
        cash_buffer_pct: float = 0.01,
    ):
        if not 0.0 <= relevance_floor <= 1.0:
            raise ValueError("relevance_floor must be in [0, 1]")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if burst_count < 1:
            raise ValueError("burst_count must be >= 1")
        if burst_window < 1:
            raise ValueError("burst_window must be >= 1")
        if hold_days < 1:
            raise ValueError("hold_days must be >= 1")
        if not 0.0 <= cash_buffer_pct < 1.0:
            raise ValueError("cash_buffer_pct must be in [0, 1)")
        self.relevance_floor = relevance_floor
        self.score_threshold = score_threshold
        self.burst_count = burst_count
        self.burst_window = burst_window
        self.hold_days = hold_days
        self.lookback_days = lookback_days
        self.cash_buffer_pct = cash_buffer_pct
        # (article_date, ticker_sentiment) for articles clearing relevance_floor
        self._events: list[tuple[date, float]] = []
        self._bars_in_trade = 0

    def on_start(self, symbols: list[str]) -> None:
        from trading_agent.data.news_source import get_headlines

        symbol = symbols[0]
        # Same wall-clock-anchored window rationale as news_sentiment: AV's free
        # tier caps at 1000 articles and ~18-24 months of history, so pin to the
        # recent past rather than the backtest's start/end (which may be older).
        today = date.today()
        fetch_start = (today - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        fetch_end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            rows = get_headlines(
                symbol, fetch_start, fetch_end, min_relevance=self.relevance_floor
            )
        except Exception:
            # Offline / no API key / rate limited: stay flat for the run.
            return

        for r in rows:
            try:
                d = datetime.strptime(r.date, "%Y-%m-%d").date()
            except ValueError:
                continue
            self._events.append((d, r.ticker_sentiment))
        self._events.sort(key=lambda e: e[0])

    def _burst_counts_as_of(self, as_of: date) -> tuple[int, int]:
        """(positive, negative) relevant-article counts in the trailing window.

        Only articles dated within (as_of - burst_window, as_of] are counted, and
        only those dated <= as_of — never future articles (lookahead safety).
        """
        window_start = as_of - timedelta(days=self.burst_window - 1)
        pos = neg = 0
        for d, score in self._events:
            if d < window_start or d > as_of:
                continue
            if score >= self.score_threshold:
                pos += 1
            elif score <= -self.score_threshold:
                neg += 1
        return pos, neg

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        held = portfolio.position(bar.symbol)
        pos, neg = self._burst_counts_as_of(bar.timestamp.date())

        if held > 0:
            self._bars_in_trade += 1
            negative_burst = neg >= self.burst_count
            time_stop = self._bars_in_trade >= self.hold_days
            if negative_burst or time_stop:
                self._bars_in_trade = 0
                return [Order(symbol=bar.symbol, side=Side.SELL, quantity=held)]
            return []

        # Flat: enter on a positive burst. max_affordable reserves a small cash
        # buffer so the next-bar gap-up + slippage fill isn't silently dropped.
        if pos >= self.burst_count:
            qty = portfolio.max_affordable(bar.close, self.cash_buffer_pct)
            if qty > 0:
                self._bars_in_trade = 0
                return [Order(symbol=bar.symbol, side=Side.BUY, quantity=qty)]
        return []
