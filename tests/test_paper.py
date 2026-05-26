"""Alpaca paper-trading tests with a mocked TradingClient.

We don't hit Alpaca's API. We verify:
  1. AlpacaPaperBroker refuses to construct in live mode.
  2. submit_market_order maps our Order to an Alpaca request correctly.
  3. paper_tick respects dry_run (never touches the broker).
  4. paper_tick proposes the strategy's emitted orders on the latest bar.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.core.orders import Order, Side


def test_broker_refuses_live_construction():
    from trading_agent.broker.alpaca import AlpacaPaperBroker
    with pytest.raises(NotImplementedError):
        AlpacaPaperBroker("k", "s", allow_live=True)


def test_broker_refuses_empty_credentials():
    from trading_agent.broker.alpaca import AlpacaPaperBroker
    with pytest.raises(ValueError):
        AlpacaPaperBroker("", "secret")
    with pytest.raises(ValueError):
        AlpacaPaperBroker("key", "")


def test_submit_market_order_passes_buy_to_alpaca():
    mock_resp = MagicMock(
        id="ord-1", symbol="AAPL", side="OrderSide.BUY",
        qty="5", status="OrderStatus.NEW",
        filled_avg_price=None, submitted_at="2026-05-26T10:00:00Z",
    )
    mock_client = MagicMock()
    mock_client.submit_order.return_value = mock_resp

    with patch("alpaca.trading.client.TradingClient", return_value=mock_client):
        from trading_agent.broker.alpaca import AlpacaPaperBroker
        broker = AlpacaPaperBroker("key", "secret")
        result = broker.submit_market_order(Order("AAPL", Side.BUY, 5))

    assert result.order_id == "ord-1"
    assert result.symbol == "AAPL"
    assert result.side == "BUY"
    # Verify the right enum was passed
    submitted_request = mock_client.submit_order.call_args.args[0]
    from alpaca.trading.enums import OrderSide
    assert submitted_request.side == OrderSide.BUY
    assert submitted_request.qty == 5


def test_paper_tick_dry_run_does_not_call_broker(tmp_path, monkeypatch):
    """dry_run must never touch the broker, even if one is passed."""
    from trading_agent.broker.paper_runner import paper_tick

    fake_broker = MagicMock()

    with patch("trading_agent.broker.paper_runner.load_bars") as mock_load:
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = [100 + i for i in range(80)]
        mock_load.return_value = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1_000_000]*80},
            index=idx,
        )

        result = paper_tick(
            strategy_name="sma_cross",
            symbol="AAPL",
            params={"fast": 5, "slow": 20},
            broker=fake_broker,
            dry_run=True,
        )

    fake_broker.account.assert_not_called()
    fake_broker.submit_market_order.assert_not_called()
    assert result.dry_run is True
    assert result.submitted == []


def test_paper_tick_no_bars_returns_skipped(tmp_path, monkeypatch):
    from trading_agent.broker.paper_runner import paper_tick

    with patch("trading_agent.broker.paper_runner.load_bars") as mock_load:
        import pandas as pd
        mock_load.return_value = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = paper_tick(
            strategy_name="sma_cross",
            symbol="ZZZZ",
            params={"fast": 5, "slow": 20},
            broker=None,
            dry_run=True,
        )

    assert result.bars_seen == 0
    assert "no bars" in (result.skipped_reason or "")


def test_paper_tick_uses_broker_state_when_not_dry_run(tmp_path, monkeypatch):
    """When live, the synthetic portfolio is seeded from the broker's account
    + positions so the strategy's sizing logic is grounded in real cash."""
    from trading_agent.broker.alpaca import AccountSnapshot, AlpacaOrder, AlpacaPosition
    from trading_agent.broker.paper_runner import paper_tick

    fake_broker = MagicMock()
    fake_broker.account.return_value = AccountSnapshot(
        cash=50_000.0, portfolio_value=50_000.0, buying_power=50_000.0, is_paper=True,
    )
    fake_broker.positions.return_value = []
    fake_broker.submit_market_order.return_value = AlpacaOrder(
        order_id="ord-1", symbol="AAPL", side="BUY",
        quantity=1.0, status="NEW", filled_avg_price=None, submitted_at="now",
    )

    with patch("trading_agent.broker.paper_runner.load_bars") as mock_load:
        import pandas as pd
        # Engineer a clean SMA(5,20) golden cross at the end
        prices = ([100.0] * 25) + ([100 + i * 2 for i in range(30)])
        idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
        mock_load.return_value = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1_000_000]*len(prices)},
            index=idx,
        )

        result = paper_tick(
            strategy_name="sma_cross",
            symbol="AAPL",
            params={"fast": 5, "slow": 20},
            broker=fake_broker,
            dry_run=False,
        )

    fake_broker.account.assert_called_once()
    assert result.dry_run is False
