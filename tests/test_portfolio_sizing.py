"""Tests for Portfolio.max_affordable — the shared cash-buffered sizing helper."""
import pytest

from trading_agent.core.portfolio import Portfolio


def test_reserves_buffer_below_all_in():
    pf = Portfolio(starting_cash=100_000.0)
    # All-in would be 1000 @ $100; the default 1% buffer must leave headroom.
    assert pf.max_affordable(100.0) == 990
    assert pf.max_affordable(100.0) < int(pf.cash // 100.0)


def test_zero_buffer_is_all_in():
    pf = Portfolio(starting_cash=100_000.0)
    assert pf.max_affordable(100.0, buffer_pct=0.0) == 1000


def test_custom_buffer():
    pf = Portfolio(starting_cash=100_000.0)
    assert pf.max_affordable(100.0, buffer_pct=0.05) == 950


def test_non_positive_price_returns_zero():
    pf = Portfolio(starting_cash=100_000.0)
    assert pf.max_affordable(0.0) == 0
    assert pf.max_affordable(-5.0) == 0


def test_rejects_invalid_buffer():
    pf = Portfolio(starting_cash=100_000.0)
    with pytest.raises(ValueError):
        pf.max_affordable(100.0, buffer_pct=1.0)
    with pytest.raises(ValueError):
        pf.max_affordable(100.0, buffer_pct=-0.1)


def test_buffered_order_clears_a_gap_up_fill():
    """The buffered quantity must actually fill when the next-bar open gaps up."""
    from datetime import datetime
    from trading_agent.core.orders import Order, Side

    pf = Portfolio(starting_cash=100_000.0)
    close = 223.57
    qty = pf.max_affordable(close)            # sized against the close
    gap_up_open = close * 1.0011 * 1.0005     # +0.11% gap + 5bps slippage (the AAPL case)
    # All-in sizing would have raised here; the buffered qty must not.
    pf.fill_at(Order(symbol="AAPL", side=Side.BUY, quantity=qty),
               price=gap_up_open, timestamp=datetime(2024, 11, 18))
    assert pf.position("AAPL") == qty
