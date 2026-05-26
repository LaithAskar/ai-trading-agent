from __future__ import annotations

from datetime import datetime

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


class FilingsSentimentLLM(Strategy):
    """Same sentiment-momentum logic as `filings_sentiment`, but the score
    per filing comes from Claude analyzing the filing text instead of
    Loughran-McDonald word counting.

    Requires ANTHROPIC_API_KEY. Scores are cached in data/llm_cache.sqlite3,
    so a rerun of a backtest costs nothing after the first pass.

    Pass cache_only=True to refuse to call the API — for free, deterministic
    reruns after you've populated the cache once.
    """

    name = "filings_sentiment_llm"

    def __init__(
        self,
        threshold: float = 0.05,
        form: str = "10-Q",
        max_filings: int = 20,
        cache_only: bool = False,
        model: str = "claude-sonnet-4-6",
    ):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [-1, 1]")
        if form not in ("10-Q", "10-K", "8-K"):
            raise ValueError("form must be one of 10-Q, 10-K, 8-K")
        self.threshold = threshold
        self.form = form
        self.max_filings = max_filings
        self.cache_only = cache_only
        self.model = model
        self._scores: list[tuple[datetime, float, float]] = []  # (filed, score, confidence)

    def on_start(self, symbols: list[str]) -> None:
        from trading_agent.data.edgar_source import filing_excerpt, list_filings
        from trading_agent.llm.filing_analyzer import analyze_filing
        from trading_agent.config import Config

        cfg = Config.load()
        symbol = symbols[0]

        try:
            refs = list_filings(symbol, form=self.form, limit=self.max_filings)
        except Exception:
            return

        for ref in refs:
            try:
                ex = filing_excerpt(symbol, ref.accession_no, max_chars=12_000)
                if not ex["text_excerpt"]:
                    continue
                analysis = analyze_filing(
                    ticker=symbol,
                    accession_no=ref.accession_no,
                    text=ex["text_excerpt"],
                    api_key=cfg.anthropic_api_key,
                    model=self.model,
                    cache_only=self.cache_only,
                )
                filed = datetime.strptime(ref.filing_date, "%Y-%m-%d")
                self._scores.append((filed, analysis.score, analysis.confidence))
            except Exception:
                continue

        self._scores.sort(key=lambda x: x[0])

    def _signal_as_of(self, ts: datetime) -> int | None:
        prior = [(d, s, c) for d, s, c in self._scores if d <= ts]
        if len(prior) < 2:
            return None
        last_score = prior[-1][1]
        prev_score = prior[-2][1]
        delta = last_score - prev_score
        if delta > self.threshold:
            return 1
        if delta < -self.threshold:
            return -1
        return 0

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        signal = self._signal_as_of(bar.timestamp)
        if signal is None or signal == 0:
            return []
        held = portfolio.position(bar.symbol)
        if signal > 0 and held == 0:
            qty = int(portfolio.cash // bar.close)
            if qty > 0:
                return [Order(symbol=bar.symbol, side=Side.BUY, quantity=qty)]
        elif signal < 0 and held > 0:
            return [Order(symbol=bar.symbol, side=Side.SELL, quantity=held)]
        return []
