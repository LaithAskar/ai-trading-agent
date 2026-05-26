from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading_agent.backtest.engine import BacktestEngine
from trading_agent.core.events import Bar
from trading_agent.core.orders import Order, Side
from trading_agent.core.portfolio import Portfolio
from trading_agent.core.strategy import Strategy


def make_bars(prices: list[tuple[float, float]], symbol: str = "TEST") -> list[Bar]:
    """Build a Bar series from [(open, close), ...]. Highs/lows match for simplicity."""
    start = datetime(2024, 1, 1)
    return [
        Bar(
            symbol=symbol,
            timestamp=start + timedelta(days=i),
            open=o,
            high=max(o, c),
            low=min(o, c),
            close=c,
            volume=1_000_000,
        )
        for i, (o, c) in enumerate(prices)
    ]


class BuyOnce(Strategy):
    name = "buy_once"

    def __init__(self, buy_on_bar_index: int, quantity: int):
        self.buy_on_bar_index = buy_on_bar_index
        self.quantity = quantity
        self._i = -1
        self.seen_bars: list[Bar] = []

    def on_bar(self, bar, portfolio):
        self._i += 1
        self.seen_bars.append(bar)
        if self._i == self.buy_on_bar_index:
            return [Order(symbol=bar.symbol, side=Side.BUY, quantity=self.quantity)]
        return []


def test_order_fills_at_next_bar_open_not_current_close():
    """Lookahead-safety contract: order emitted at bar t fills at bar t+1's OPEN."""
    bars = make_bars([(10.0, 11.0), (12.0, 13.0), (14.0, 15.0)])
    strat = BuyOnce(buy_on_bar_index=0, quantity=10)

    engine = BacktestEngine(starting_cash=1000.0)
    result = engine.run(strat, "TEST", iter(bars))

    fills = result.portfolio.fills
    assert len(fills) == 1
    fill = fills[0]
    assert fill.price == 12.0, "should fill at bar 1's open, not bar 0's close"
    assert fill.timestamp == bars[1].timestamp


def test_no_orders_filled_after_final_bar():
    """An order emitted on the LAST bar has no next bar — must not fill."""
    bars = make_bars([(10.0, 11.0), (12.0, 13.0)])
    strat = BuyOnce(buy_on_bar_index=1, quantity=5)

    engine = BacktestEngine(starting_cash=1000.0)
    result = engine.run(strat, "TEST", iter(bars))

    assert result.portfolio.fills == []


def test_equity_marked_to_close_not_open():
    """Mark-to-market should use the bar's close, not its open."""
    bars = make_bars([(10.0, 11.0), (12.0, 20.0)])

    class BuyFirst(Strategy):
        name = "buy_first"
        _i = -1

        def on_bar(self, bar, portfolio):
            self._i += 1
            if self._i == 0:
                return [Order(symbol=bar.symbol, side=Side.BUY, quantity=10)]
            return []

    engine = BacktestEngine(starting_cash=1000.0)
    result = engine.run(BuyFirst(), "TEST", iter(bars))

    final_equity = result.portfolio.equity_curve[-1][1]
    # After filling 10 shares @ $12 open on bar 1: cash = 1000 - 120 = 880.
    # Marked at close 20: equity = 880 + 10*20 = 1080.
    assert final_equity == pytest.approx(1080.0)


def test_insufficient_cash_order_rejected_silently():
    bars = make_bars([(10.0, 11.0), (12.0, 13.0)])
    strat = BuyOnce(buy_on_bar_index=0, quantity=10_000)

    engine = BacktestEngine(starting_cash=100.0)
    result = engine.run(strat, "TEST", iter(bars))

    assert result.portfolio.fills == []
    assert result.portfolio.cash == 100.0


def test_strategy_never_sees_future_bars():
    """Strategy.on_bar receives bars in order; the bar it sees is always 'now', never future."""
    bars = make_bars([(10, 11), (12, 13), (14, 15), (16, 17)])
    strat = BuyOnce(buy_on_bar_index=0, quantity=1)

    engine = BacktestEngine(starting_cash=1000.0)
    engine.run(strat, "TEST", iter(bars))

    for i, seen in enumerate(strat.seen_bars):
        assert seen.timestamp == bars[i].timestamp


def test_slippage_buys_worse_sells_worse():
    """Slippage moves buys up and sells down from the raw bar open."""
    bars = make_bars([(10.0, 11.0), (100.0, 100.0), (200.0, 200.0)])
    strat = BuyOnce(buy_on_bar_index=0, quantity=1)

    engine = BacktestEngine(starting_cash=1000.0, slippage_bps=50.0)  # 0.5%
    result = engine.run(strat, "TEST", iter(bars))

    fill = result.portfolio.fills[0]
    assert fill.price == pytest.approx(100.0 * 1.005)


def test_commission_deducted_from_cash():
    bars = make_bars([(10.0, 11.0), (100.0, 100.0)])
    strat = BuyOnce(buy_on_bar_index=0, quantity=1)

    engine = BacktestEngine(starting_cash=1000.0, commission_per_trade=2.50)
    result = engine.run(strat, "TEST", iter(bars))

    # Bought 1 share @ 100, paid $100 + $2.50 commission
    assert result.portfolio.cash == pytest.approx(1000.0 - 100.0 - 2.50)


def test_engine_rejects_negative_friction():
    with pytest.raises(ValueError):
        BacktestEngine(slippage_bps=-1)
    with pytest.raises(ValueError):
        BacktestEngine(commission_per_trade=-1)


def test_portfolio_round_trip_pnl():
    pf = Portfolio(starting_cash=1000.0)
    pf.fill_at(Order("X", Side.BUY, 10), price=10.0, timestamp=datetime(2024, 1, 1))
    pf.fill_at(Order("X", Side.SELL, 10), price=12.0, timestamp=datetime(2024, 1, 2))
    assert pf.cash == pytest.approx(1020.0)
    assert pf.position("X") == 0.0
