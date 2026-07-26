"""Alpaca paper-trading tests with a mocked TradingClient.

We don't hit Alpaca's API. We verify:
  1. AlpacaPaperBroker refuses to construct in live mode.
  2. submit_market_order maps our Order to an Alpaca request correctly.
  3. paper_tick respects dry_run (never touches the broker).
  4. paper_tick proposes the strategy's emitted orders on the latest bar.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Lock
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.core.orders import Order, Side


class _AlwaysBuyStrategy:
    def on_start(self, symbols):
        pass

    def on_bar(self, bar, portfolio):
        return [Order(bar.symbol, Side.BUY, 1)]


class _TwoBuysStrategy(_AlwaysBuyStrategy):
    def on_bar(self, bar, portfolio):
        return [Order(bar.symbol, Side.BUY, 1), Order(bar.symbol, Side.BUY, 2)]


def _paper_broker():
    from trading_agent.broker.alpaca import AccountSnapshot, AlpacaOrder

    broker = MagicMock()
    broker.account.return_value = AccountSnapshot(
        cash=50_000.0, portfolio_value=50_000.0, buying_power=50_000.0, is_paper=True,
    )
    broker.positions.return_value = []
    broker.submit_market_order.return_value = AlpacaOrder(
        order_id="ord-1", symbol="AAPL", side="BUY",
        quantity=1.0, status="NEW", filled_avg_price=None, submitted_at="now",
    )
    return broker


def _bar_frame(last_day="2024-01-02"):
    import pandas as pd

    idx = pd.date_range(end=last_day, periods=2, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_000],
        },
        index=idx,
    )


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
        result = broker.submit_market_order(
            Order("AAPL", Side.BUY, 5), client_order_id="paper-deterministic-key"
        )

    assert result.order_id == "ord-1"
    assert result.symbol == "AAPL"
    assert result.side == "BUY"
    # Verify the right enum was passed
    submitted_request = mock_client.submit_order.call_args.args[0]
    from alpaca.trading.enums import OrderSide
    assert submitted_request.side == OrderSide.BUY
    assert submitted_request.qty == 5
    assert submitted_request.client_order_id == "paper-deterministic-key"


def test_get_order_by_client_order_id_maps_alpaca_response():
    mock_client = MagicMock()
    mock_client.get_order_by_client_id.return_value = MagicMock(
        id="ord-1", symbol="AAPL", side="OrderSide.BUY", qty="5",
        status="OrderStatus.NEW", filled_avg_price=None, submitted_at="now",
    )
    with patch("alpaca.trading.client.TradingClient", return_value=mock_client):
        from trading_agent.broker.alpaca import AlpacaPaperBroker
        broker = AlpacaPaperBroker("key", "secret")
        result = broker.get_order_by_client_order_id("paper-key")

    assert result.order_id == "ord-1"
    mock_client.get_order_by_client_id.assert_called_once_with("paper-key")


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


def test_paper_tick_rejects_non_paper_broker_before_strategy_or_submission():
    """The execution boundary must fail closed for custom/wrapped live brokers."""
    from trading_agent.broker import paper_runner
    from trading_agent.broker.alpaca import AccountSnapshot

    broker = MagicMock()
    broker.account.return_value = AccountSnapshot(50_000, 50_000, 50_000, False)

    with (
        patch.object(paper_runner, "load_strategy") as load_strategy,
        patch.object(paper_runner, "load_bars") as load_bars,
        pytest.raises(RuntimeError, match="paper-only"),
    ):
        paper_runner.paper_tick(
            strategy_name="always_buy", symbol="AAPL", broker=broker, dry_run=False,
        )

    broker.account.assert_called_once_with()
    broker.positions.assert_not_called()
    broker.submit_market_order.assert_not_called()
    load_strategy.assert_not_called()
    load_bars.assert_not_called()


def test_paper_tick_repeat_run_submits_same_bar_order_once(tmp_path, monkeypatch):
    """A scheduler retry must not submit the same strategy/bar intent twice."""
    from trading_agent.broker import paper_runner

    monkeypatch.setattr(
        paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "paper-orders.sqlite3", raising=False
    )
    broker = _paper_broker()

    with (
        patch.object(paper_runner, "load_strategy", return_value=_AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        first = paper_runner.paper_tick(
            strategy_name="always_buy", symbol="aapl", params={"window": 2, "threshold": 1},
            broker=broker, dry_run=False,
        )
        second = paper_runner.paper_tick(
            strategy_name="always_buy", symbol="AAPL", params={"threshold": 1, "window": 2},
            broker=broker, dry_run=False,
        )

    assert len(first.submitted) == 1
    assert second.submitted == []
    assert "already submitted" in (second.skipped_reason or "")
    broker.submit_market_order.assert_called_once()


def test_paper_tick_dry_run_does_not_consume_idempotency_key(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner

    monkeypatch.setattr(
        paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "paper-orders.sqlite3", raising=False
    )
    broker = _paper_broker()

    with (
        patch.object(paper_runner, "load_strategy", return_value=_AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        preview = paper_runner.paper_tick(
            strategy_name="always_buy", symbol="AAPL", broker=broker, dry_run=True,
        )
        submitted = paper_runner.paper_tick(
            strategy_name="always_buy", symbol="AAPL", broker=broker, dry_run=False,
        )

    assert preview.submitted == []
    assert len(submitted.submitted) == 1
    broker.submit_market_order.assert_called_once()


def test_paper_tick_new_bar_gets_a_new_idempotency_key(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner

    monkeypatch.setattr(
        paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "paper-orders.sqlite3", raising=False
    )
    broker = _paper_broker()

    with (
        patch.object(paper_runner, "load_strategy", return_value=_AlwaysBuyStrategy()),
        patch.object(
            paper_runner,
            "load_bars",
            side_effect=[_bar_frame("2024-01-02"), _bar_frame("2024-01-03")],
        ),
    ):
        paper_runner.paper_tick(
            strategy_name="always_buy", symbol="AAPL", broker=broker, dry_run=False,
        )
        paper_runner.paper_tick(
            strategy_name="always_buy", symbol="AAPL", broker=broker, dry_run=False,
        )

    assert broker.submit_market_order.call_count == 2
    client_order_ids = [
        call.kwargs["client_order_id"] for call in broker.submit_market_order.call_args_list
    ]
    assert client_order_ids[0] != client_order_ids[1]
    assert all(len(client_order_id) == 48 for client_order_id in client_order_ids)


class _ReconcilingBroker:
    def __init__(self):
        from trading_agent.broker.alpaca import AccountSnapshot
        self._account = AccountSnapshot(50_000, 50_000, 50_000, True)
        self.orders = {}
        self.submit_count = 0
        self._lock = Lock()

    def account(self):
        return self._account

    def positions(self):
        return []

    def submit_market_order(self, order, *, client_order_id=None):
        from trading_agent.broker.alpaca import AlpacaOrder
        with self._lock:
            self.submit_count += 1
            accepted = AlpacaOrder(
                f"ord-{self.submit_count}", order.symbol, order.side.value,
                order.quantity, "NEW", None, "now",
            )
            self.orders[client_order_id] = accepted
            return accepted

    def get_order_by_client_order_id(self, client_order_id):
        return self.orders.get(client_order_id)


def _run_always_buy(paper_runner, broker):
    return paper_runner.paper_tick(
        strategy_name="always_buy", symbol="AAPL", broker=broker, dry_run=False,
    )


def test_concurrent_identical_ticks_have_one_submitter(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "orders.db")
    broker = _ReconcilingBroker()
    with (
        patch.object(paper_runner, "load_strategy", side_effect=lambda *_: _AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        results = list(pool.map(lambda _: _run_always_buy(paper_runner, broker), range(2)))

    assert broker.submit_count == 1
    assert sum(result.submitted_count for result in results) == 1
    assert sum(
        result.already_submitted_count + result.reconciled_count + result.pending_count
        for result in results
    ) == 1


def test_acceptance_then_checkpoint_failure_is_explicit_and_retry_repairs(
    tmp_path, monkeypatch,
):
    from trading_agent.broker import paper_runner
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "orders.db")
    broker = _ReconcilingBroker()
    original = paper_runner._PaperOrderCheckpoint.mark_submitted
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("disk full")
        return original(self, *args, **kwargs)

    with (
        patch.object(paper_runner, "load_strategy", side_effect=lambda *_: _AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
        patch.object(paper_runner._PaperOrderCheckpoint, "mark_submitted", fail_once),
    ):
        first = _run_always_buy(paper_runner, broker)
        second = _run_always_buy(paper_runner, broker)

    assert broker.submit_count == 1
    assert first.submitted_count == 1
    assert first.checkpoint_error_count == 1
    assert first.skipped_reason is None
    assert second.reconciled_count == 1
    assert second.submitted_count == 0
    assert second.skipped_reason is not None


def test_checkpoint_commit_failure_rolls_back_and_later_order_is_processed(
    tmp_path, monkeypatch,
):
    """A failed first transition must not poison the next order's transaction."""
    from trading_agent.broker import paper_runner

    db = tmp_path / "orders.db"
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", db)
    broker = _ReconcilingBroker()
    real_connect = sqlite3.connect

    class _FailThirdCommit:
        def __init__(self, connection):
            self._connection = connection
            self.commit_calls = 0

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def commit(self):
            self.commit_calls += 1
            # __init__, first claim, then first mark_submitted.
            if self.commit_calls == 3:
                raise sqlite3.OperationalError("simulated checkpoint commit failure")
            return self._connection.commit()

    monkeypatch.setattr(
        paper_runner.sqlite3, "connect",
        lambda *args, **kwargs: _FailThirdCommit(real_connect(*args, **kwargs)),
    )

    with (
        patch.object(paper_runner, "load_strategy", return_value=_TwoBuysStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        result = paper_runner.paper_tick(
            strategy_name="two_buys", symbol="AAPL", broker=broker, dry_run=False,
        )

    assert broker.submit_count == 2
    assert [outcome.status for outcome in result.outcomes] == [
        "submitted_checkpoint_error", "submitted",
    ]
    assert "simulated checkpoint commit failure" in (result.outcomes[0].detail or "")
    assert result.submitted_count == 2
    assert result.checkpoint_error_count == 1


def test_duplicate_submit_error_reconciles_instead_of_generic_failure(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "orders.db")
    broker = _ReconcilingBroker()

    def accepted_then_error(order, *, client_order_id=None):
        accepted = _ReconcilingBroker.submit_market_order(
            broker, order, client_order_id=client_order_id
        )
        assert accepted
        raise RuntimeError("client_order_id must be unique")

    broker.submit_market_order = accepted_then_error
    with (
        patch.object(paper_runner, "load_strategy", return_value=_AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        result = _run_always_buy(paper_runner, broker)

    assert broker.submit_count == 1
    assert result.reconciled_count == 1
    assert result.failed_count == 0
    assert result.skipped_reason is not None


def test_partial_dedup_has_explicit_counts_without_skipped_reason(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", tmp_path / "orders.db")
    broker = _ReconcilingBroker()

    class _TwoBuys(_AlwaysBuyStrategy):
        def on_bar(self, bar, portfolio):
            return [Order(bar.symbol, Side.BUY, 1), Order(bar.symbol, Side.BUY, 2)]

    with (
        patch.object(paper_runner, "load_strategy", side_effect=lambda *_: _TwoBuys()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        first = paper_runner.paper_tick(
            strategy_name="two_buys", symbol="AAPL", broker=broker, dry_run=False,
        )
        assert first.submitted_count == 2
        with sqlite3.connect(tmp_path / "orders.db") as connection:
            connection.execute(
                "DELETE FROM paper_order_claims WHERE broker_order_id = ?", ("ord-2",)
            )
        second = paper_runner.paper_tick(
            strategy_name="two_buys", symbol="AAPL", broker=broker, dry_run=False,
        )

    assert second.submitted_count == 1
    assert second.already_submitted_count == 1
    assert second.skipped_reason is None
    assert [outcome.status for outcome in second.outcomes] == ["already_submitted", "submitted"]


def test_stale_pending_is_resubmitted_only_after_broker_confirms_absence(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner
    db = tmp_path / "orders.db"
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", db)
    broker = _ReconcilingBroker()

    checkpoint = paper_runner._PaperOrderCheckpoint(db)
    bars = list(paper_runner.iter_bars("AAPL", _bar_frame()))
    key = paper_runner._paper_order_key(
        strategy_name="always_buy", params={}, requested_symbol="AAPL",
        bar=bars[-1], order=Order("AAPL", Side.BUY, 1), order_index=0,
    )
    checkpoint.claim(key, paper_runner._client_order_id(key))
    checkpoint.close()
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE paper_order_claims SET updated_at = ?",
            ((datetime.now().astimezone() - timedelta(hours=1)).isoformat(),),
        )

    with (
        patch.object(paper_runner, "load_strategy", return_value=_AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        result = _run_always_buy(paper_runner, broker)

    assert broker.submit_count == 1
    assert result.submitted_count == 1


def test_stale_pending_without_broker_lookup_stays_unknown(tmp_path, monkeypatch):
    from trading_agent.broker import paper_runner
    db = tmp_path / "orders.db"
    monkeypatch.setattr(paper_runner, "PAPER_ORDER_CHECKPOINT_DB", db)
    broker = _paper_broker()  # MagicMock deliberately has no declared lookup API.
    bars = list(paper_runner.iter_bars("AAPL", _bar_frame()))
    key = paper_runner._paper_order_key(
        strategy_name="always_buy", params={}, requested_symbol="AAPL",
        bar=bars[-1], order=Order("AAPL", Side.BUY, 1), order_index=0,
    )
    checkpoint = paper_runner._PaperOrderCheckpoint(db)
    checkpoint.claim(key, paper_runner._client_order_id(key))
    checkpoint.close()
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE paper_order_claims SET updated_at = ?",
            ((datetime.now().astimezone() - timedelta(hours=1)).isoformat(),),
        )

    with (
        patch.object(paper_runner, "load_strategy", return_value=_AlwaysBuyStrategy()),
        patch.object(paper_runner, "load_bars", return_value=_bar_frame()),
    ):
        result = _run_always_buy(paper_runner, broker)

    broker.submit_market_order.assert_not_called()
    assert result.pending_count == 1
    assert "cannot be reconciled" in (result.outcomes[0].detail or "")
