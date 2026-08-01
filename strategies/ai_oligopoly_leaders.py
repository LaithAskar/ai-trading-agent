from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable

from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


class AiOligopolyLeaders(Strategy):
    """Belcourt-inspired, long-only trend strategy for a curated leader universe.

    This is an independent rules-based adaptation of Brooker Belcourt's public
    investment themes, not a reproduction of his private Autopilot portfolio.
    It only consumes completed bars. Orders emitted for bar t are filled by the
    backtest engine at bar t+1 open.
    """

    name = "ai_oligopoly_leaders"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        momentum_lookback: int = 20,
        min_momentum_pct: float = 5.0,
        leaders: str | Iterable[str] = "NVDA,MSFT,GOOGL,AMZN,META,AVGO,TSM,ASML",
        target_notional: float = 20.0,
    ) -> None:
        if (
            not isinstance(fast, int)
            or isinstance(fast, bool)
            or not isinstance(slow, int)
            or isinstance(slow, bool)
        ):
            raise ValueError("fast and slow must be integers")
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("fast and slow must be positive and fast must be < slow")
        if (
            not isinstance(momentum_lookback, int)
            or isinstance(momentum_lookback, bool)
            or momentum_lookback <= 0
        ):
            raise ValueError("momentum_lookback must be positive")
        if not math.isfinite(min_momentum_pct):
            raise ValueError("min_momentum_pct must be finite")
        if not math.isfinite(target_notional) or target_notional <= 0:
            raise ValueError("target_notional must be finite and positive")

        raw_leaders = leaders.split(",") if isinstance(leaders, str) else leaders
        normalized = {str(symbol).strip().upper() for symbol in raw_leaders if str(symbol).strip()}
        if not normalized:
            raise ValueError("leaders must contain at least one symbol")

        self.fast = fast
        self.slow = slow
        self.momentum_lookback = momentum_lookback
        self.min_momentum_pct = float(min_momentum_pct)
        self.leaders = frozenset(normalized)
        self.target_notional = float(target_notional)
        history = max(slow, momentum_lookback + 1)
        self._closes: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history)
        )

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        symbol = bar.symbol.upper()
        if symbol not in self.leaders:
            return []
        if not math.isfinite(bar.close) or bar.close <= 0:
            return []

        closes = self._closes[symbol]
        closes.append(float(bar.close))
        required = max(self.slow, self.momentum_lookback + 1)
        if len(closes) < required:
            return []

        values = list(closes)
        fast_sma = self._mean(values[-self.fast :])
        slow_sma = self._mean(values[-self.slow :])
        momentum_base = values[-(self.momentum_lookback + 1)]
        if momentum_base <= 0:
            return []
        momentum_pct = (bar.close / momentum_base - 1.0) * 100.0
        held = portfolio.position(symbol)

        trend_confirmed = (
            bar.close > slow_sma
            and fast_sma > slow_sma
            and momentum_pct >= self.min_momentum_pct
        )
        trend_broken = bar.close < slow_sma or fast_sma < slow_sma

        if held == 0 and trend_confirmed:
            notional = min(self.target_notional, portfolio.cash)
            quantity = notional / bar.close
            if math.isfinite(quantity) and quantity > 0:
                return [Order(symbol=symbol, side=Side.BUY, quantity=quantity)]
        elif held > 0 and trend_broken:
            return [Order(symbol=symbol, side=Side.SELL, quantity=held)]
        return []
