from __future__ import annotations

from datetime import datetime

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


# A tiny Loughran-McDonald-style finance sentiment lexicon. The full LM dictionary
# is hundreds of words long and freely available; this small subset is enough for a
# defensible MVP signal. Swap in the full lexicon for production.
POS_WORDS = (
    "increase", "growth", "strong", "improved", "exceeded", "favorable",
    "gained", "achieved", "successful", "robust", "expand", "rose", "higher",
    "outperform", "record", "surpassed", "momentum", "upbeat",
)

NEG_WORDS = (
    "decline", "decrease", "weak", "challenge", "uncertainty", "adverse",
    "loss", "failed", "deteriorat", "headwind", "lower", "underperform",
    "shortfall", "disappoint", "missed", "fell", "downturn", "warning",
)


def _score_text(text: str) -> float:
    """Net polarity in [-1, 1]: (pos - neg) / (pos + neg)."""
    if not text:
        return 0.0
    lower = text.lower()
    pos = sum(lower.count(w) for w in POS_WORDS)
    neg = sum(lower.count(w) for w in NEG_WORDS)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


class FilingsSentiment(Strategy):
    """Sentiment-momentum strategy over SEC filings.

    Long when the most recent filing's sentiment score IMPROVED relative to the
    prior filing. Flat when it deteriorated. The absolute level of LM-style
    sentiment is uninformative on its own — SEC filings are mandated to disclose
    risks, so most filings score negative — but the *direction of change*
    quarter over quarter carries signal.

    Single-symbol, quarterly cadence (signal changes only when a new filing posts).
    """

    name = "filings_sentiment"

    def __init__(self, threshold: float = 0.0, form: str = "10-Q", max_filings: int = 30):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [-1, 1]")
        if form not in ("10-Q", "10-K", "8-K"):
            raise ValueError("form must be one of 10-Q, 10-K, 8-K")
        self.threshold = threshold
        self.form = form
        self.max_filings = max_filings
        # Sorted ascending by filing_date.
        self._scores: list[tuple[datetime, float]] = []

    def on_start(self, symbols: list[str]) -> None:
        from trading_agent.data.edgar_source import filing_excerpt, list_filings

        symbol = symbols[0]
        try:
            refs = list_filings(symbol, form=self.form, limit=self.max_filings)
        except Exception:
            # Offline / EDGAR unavailable: strategy stays flat for the whole run.
            return

        for ref in refs:
            try:
                ex = filing_excerpt(symbol, ref.accession_no, max_chars=40_000)
                score = _score_text(ex["text_excerpt"])
                filed = datetime.strptime(ref.filing_date, "%Y-%m-%d")
                self._scores.append((filed, score))
            except Exception:
                continue

        self._scores.sort(key=lambda x: x[0])

    def _signal_as_of(self, ts: datetime) -> int | None:
        """+1 if sentiment improved by more than +threshold (go long).
        -1 if sentiment dropped by more than -threshold (exit).
        0 if the change is within the band (hold current position).
        None if we haven't yet seen two filings on or before ts.
        """
        prior = [(d, s) for d, s in self._scores if d <= ts]
        if len(prior) < 2:
            return None
        delta = prior[-1][1] - prior[-2][1]
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
