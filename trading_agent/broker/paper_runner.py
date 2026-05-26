from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..backtest.runner import load_strategy
from ..core.events import Bar
from ..core.orders import Order
from ..core.portfolio import Portfolio
from ..data.yfinance_source import iter_bars, load_bars
from .alpaca import AlpacaOrder, AlpacaPaperBroker


@dataclass
class PaperTickResult:
    symbol: str
    strategy: str
    bars_seen: int
    proposed_orders: list[Order]
    submitted: list[AlpacaOrder]
    dry_run: bool
    skipped_reason: str | None


def paper_tick(
    *,
    strategy_name: str,
    symbol: str,
    params: dict | None = None,
    lookback_days: int = 365,
    broker: AlpacaPaperBroker | None = None,
    dry_run: bool = False,
) -> PaperTickResult:
    """Replay the strategy across the last `lookback_days` of daily bars to
    rebuild its internal state, then ask it for orders on TODAY's closed bar
    and optionally submit those orders to Alpaca paper.

    The replay-then-submit pattern is what makes a single tick work for a
    stateful strategy. Without the replay, the strategy has no memory of
    where its SMAs / RSI / sentiment scores were.

    dry_run=True returns proposed orders without hitting the broker. Use this
    from the agent or scripts before flipping the switch to send live.
    """
    params = params or {}
    strategy = load_strategy(strategy_name, params)

    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=lookback_days)
    df = load_bars(symbol, str(start_dt), str(end_dt))
    bars: list[Bar] = list(iter_bars(symbol, df))

    if not bars:
        return PaperTickResult(
            symbol=symbol,
            strategy=strategy_name,
            bars_seen=0,
            proposed_orders=[],
            submitted=[],
            dry_run=dry_run,
            skipped_reason="no bars in lookback window",
        )

    # Replay onto a synthetic portfolio so the strategy's internal state
    # advances. We never use the synthetic portfolio's cash/positions for
    # sizing — broker.account() is the truth for cash, and broker.positions()
    # is the truth for holdings.
    synthetic = Portfolio(starting_cash=100_000.0)
    if broker is not None and not dry_run:
        acct = broker.account()
        synthetic.cash = acct.cash
        for pos in broker.positions():
            if pos.symbol == symbol.upper():
                synthetic.positions[pos.symbol] = pos.quantity

    strategy.on_start([symbol])
    proposed: list[Order] = []
    for i, bar in enumerate(bars):
        synthetic.mark(bar.symbol, bar.close)
        emitted = strategy.on_bar(bar, synthetic)
        if i == len(bars) - 1:
            proposed.extend(emitted)

    submitted: list[AlpacaOrder] = []
    if dry_run or broker is None:
        return PaperTickResult(
            symbol=symbol,
            strategy=strategy_name,
            bars_seen=len(bars),
            proposed_orders=proposed,
            submitted=[],
            dry_run=True,
            skipped_reason=None if proposed else "strategy emitted no orders",
        )

    for order in proposed:
        try:
            resp = broker.submit_market_order(order)
            submitted.append(resp)
        except Exception as e:
            return PaperTickResult(
                symbol=symbol,
                strategy=strategy_name,
                bars_seen=len(bars),
                proposed_orders=proposed,
                submitted=submitted,
                dry_run=False,
                skipped_reason=f"broker rejected after {len(submitted)} submits: {e}",
            )

    return PaperTickResult(
        symbol=symbol,
        strategy=strategy_name,
        bars_seen=len(bars),
        proposed_orders=proposed,
        submitted=submitted,
        dry_run=False,
        skipped_reason=None,
    )
