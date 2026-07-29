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


def test_broker_account_exposes_available_daily_loss_state():
    raw_account = MagicMock(
        cash="175.0", portfolio_value="190.0", buying_power="175.0", last_equity="200.0"
    )
    mock_client = MagicMock()
    mock_client.get_account.return_value = raw_account

    with patch("alpaca.trading.client.TradingClient", return_value=mock_client):
        from trading_agent.broker.alpaca import AlpacaPaperBroker
        account = AlpacaPaperBroker("key", "secret").account()

    assert account.cash == 175.0
    assert account.portfolio_value == 190.0
    assert account.daily_pnl == -10.0


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


@pytest.mark.parametrize("malformed", ["false", "true", 1, object(), None])
def test_market_session_fails_closed_on_non_boolean_open_state(malformed):
    raw_clock = MagicMock(timestamp=datetime.now().astimezone(), is_open=malformed)
    mock_client = MagicMock()
    mock_client.get_clock.return_value = raw_clock
    mock_client.get_calendar.return_value = []

    with patch("alpaca.trading.client.TradingClient", return_value=mock_client):
        from trading_agent.broker.alpaca import AlpacaPaperBroker

        session = AlpacaPaperBroker("key", "secret").market_session()

    assert session.is_open is False


def test_client_order_identity_includes_params_and_closed_bar_timestamp():
    from trading_agent.broker.paper_runner import _client_order_id
    from trading_agent.experiment import ExperimentContract

    contract = ExperimentContract(name="identity-test")
    order = Order("AAPL", Side.BUY, 1)
    baseline = _client_order_id(contract, "sma_cross", {"fast": 5}, order, "2026-07-28T20:00:00+00:00", 0)
    changed_params = _client_order_id(
        contract, "sma_cross", {"fast": 10}, order, "2026-07-28T20:00:00+00:00", 0
    )
    changed_bar = _client_order_id(
        contract, "sma_cross", {"fast": 5}, order, "2026-07-28T21:00:00+00:00", 0
    )

    assert len({baseline, changed_params, changed_bar}) == 3


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


def test_paper_tick_applies_experiment_contract_gate(tmp_path, monkeypatch):
    """Autonomous mode is not babysat, but proposed orders still pass through
    the deterministic experiment contract before broker submission."""
    from trading_agent.broker.alpaca import AccountSnapshot, MarketSessionSnapshot
    from trading_agent.broker.paper_runner import paper_tick
    from trading_agent.experiment import ExperimentContract
    from trading_agent.experiment_graph import ExperimentGraph

    fake_broker = MagicMock()
    fake_broker.account.return_value = AccountSnapshot(
        cash=50_000.0, portfolio_value=50_000.0, buying_power=50_000.0, is_paper=True, daily_pnl=0.0,
    )
    fake_broker.positions.return_value = []
    fake_broker.trades_today.return_value = 0
    fake_broker.asset_class.return_value = "US_EQUITY"
    now = datetime.now().astimezone() + timedelta(minutes=1)
    fake_broker.market_session.return_value = MarketSessionSnapshot(
        True, now, now - timedelta(hours=2), now + timedelta(hours=2)
    )

    with patch("trading_agent.broker.paper_runner.load_bars") as mock_load, patch(
        "trading_agent.broker.paper_runner.load_strategy"
    ) as mock_strategy:
        import pandas as pd

        class EmitsBuy:
            def on_start(self, symbols):
                pass

            def on_bar(self, bar, portfolio):
                return [Order("AAPL", Side.BUY, 1)]

        mock_strategy.return_value = EmitsBuy()
        prices = [100.0, 101.0]
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
            contract=ExperimentContract(name="paper-test", max_trade_usd=1.0),
            graph=ExperimentGraph(tmp_path / "graph.sqlite3"),
        )

    fake_broker.submit_market_order.assert_not_called()
    assert result.submitted == []
    assert result.rejected_orders
    assert "trade exceeds max_trade_usd" in result.rejected_orders[0]


def test_paper_tick_is_retry_idempotent_and_durably_tracks_broker_success(tmp_path):
    from trading_agent.broker.alpaca import AccountSnapshot, AlpacaOrder, MarketSessionSnapshot
    from trading_agent.broker.paper_runner import paper_tick
    from trading_agent.experiment import ExperimentContract
    from trading_agent.experiment_graph import ExperimentGraph

    fake_broker = MagicMock()
    fake_broker.account.return_value = AccountSnapshot(200, 200, 200, True, daily_pnl=0.0)
    fake_broker.positions.return_value = []
    fake_broker.trades_today.return_value = 0
    fake_broker.asset_class.return_value = "US_EQUITY"
    now = datetime.now().astimezone() + timedelta(minutes=1)
    fake_broker.market_session.return_value = MarketSessionSnapshot(
        True, now, now - timedelta(hours=2), now + timedelta(hours=2)
    )
    fake_broker.submit_market_order.return_value = AlpacaOrder(
        "broker-1", "AAPL", "BUY", 1, "NEW", None, "now"
    )
    graph = ExperimentGraph(tmp_path / "graph.sqlite3")

    class EmitsBuy:
        def on_start(self, symbols):
            pass

        def on_bar(self, bar, portfolio):
            return [Order("AAPL", Side.BUY, 1)]

    import pandas as pd
    idx = pd.date_range("2026-07-27", periods=2, freq="D")
    frame = pd.DataFrame(
        {"open": [10, 10], "high": [10, 10], "low": [10, 10], "close": [10, 10], "volume": [100, 100]},
        index=idx,
    )
    kwargs = dict(
        strategy_name="sma_cross",
        symbol="AAPL",
        broker=fake_broker,
        dry_run=False,
        contract=ExperimentContract(name="retry-test"),
        graph=graph,
    )
    with patch("trading_agent.broker.paper_runner.load_bars", return_value=frame), patch(
        "trading_agent.broker.paper_runner.load_strategy", return_value=EmitsBuy()
    ):
        first = paper_tick(**kwargs, run_id="run-1")
        second = paper_tick(**kwargs, run_id="run-2")

    fake_broker.submit_market_order.assert_called_once()
    first_client_id = first.execution_records[0].client_order_id
    assert first_client_id and first_client_id == second.execution_records[0].client_order_id
    assert second.execution_records[0].status == "idempotent_replay"
    assert second.execution_records[0].broker_order_id == "broker-1"
    assert graph.successful_execution(first_client_id)["broker_order_id"] == "broker-1"


def test_execution_claim_atomically_elects_one_submitter(tmp_path):
    from trading_agent.experiment_graph import ExperimentGraph

    db_path = tmp_path / "graph.sqlite3"
    ExperimentGraph(db_path)

    def claim_once(index: int):
        graph = ExperimentGraph(db_path)
        return graph.claim_execution(
            client_order_id="ta-concurrent-intent",
            experiment_name="concurrency-test",
            run_id=f"run-{index}",
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        claims = list(pool.map(claim_once, range(100)))

    assert sum(claim.acquired for claim in claims) == 1
    assert {claim.status for claim in claims} == {"pending"}


@pytest.mark.parametrize("paper_evidence", [False, "false", 1, object(), None])
def test_paper_tick_requires_exact_boolean_paper_attestation(tmp_path, paper_evidence):
    from trading_agent.broker.alpaca import AccountSnapshot
    from trading_agent.broker.paper_runner import paper_tick
    from trading_agent.experiment import ExperimentContract
    from trading_agent.experiment_graph import ExperimentGraph

    fake_broker = MagicMock()
    fake_broker.account.return_value = AccountSnapshot(200, 200, 200, paper_evidence, daily_pnl=0.0)
    fake_broker.positions.return_value = []

    class EmitsBuy:
        def on_start(self, symbols):
            pass

        def on_bar(self, bar, portfolio):
            return [Order("AAPL", Side.BUY, 1)]

    import pandas as pd
    frame = pd.DataFrame(
        {"open": [10], "high": [10], "low": [10], "close": [10], "volume": [100]},
        index=pd.date_range("2026-07-27", periods=1, freq="D"),
    )
    with patch("trading_agent.broker.paper_runner.load_bars", return_value=frame), patch(
        "trading_agent.broker.paper_runner.load_strategy", return_value=EmitsBuy()
    ):
        result = paper_tick(
            strategy_name="sma_cross",
            symbol="AAPL",
            broker=fake_broker,
            dry_run=False,
            contract=ExperimentContract(name="paper-only-test"),
            graph=ExperimentGraph(tmp_path / "graph.sqlite3"),
        )

    fake_broker.submit_market_order.assert_not_called()
    assert result.skipped_reason == "execution requires a paper brokerage account"
    assert result.execution_records[0].status == "rejected_non_paper_account"
    assert result.execution_records[0].decision.allowed is False


def test_paper_tick_records_ambiguous_submission_outcome(tmp_path):
    from trading_agent.broker.alpaca import AccountSnapshot, MarketSessionSnapshot
    from trading_agent.broker.paper_runner import paper_tick
    from trading_agent.experiment import ExperimentContract
    from trading_agent.experiment_graph import ExperimentGraph

    fake_broker = MagicMock()
    fake_broker.account.return_value = AccountSnapshot(200, 200, 200, True, daily_pnl=0.0)
    fake_broker.positions.return_value = []
    fake_broker.trades_today.return_value = 0
    fake_broker.asset_class.return_value = "US_EQUITY"
    now = datetime.now().astimezone() + timedelta(minutes=1)
    fake_broker.market_session.return_value = MarketSessionSnapshot(
        True, now, now - timedelta(hours=2), now + timedelta(hours=2)
    )
    fake_broker.submit_market_order.side_effect = TimeoutError("submit timed out")
    fake_broker.order_by_client_order_id.side_effect = LookupError("lookup unavailable")

    class EmitsBuy:
        def on_start(self, symbols):
            pass

        def on_bar(self, bar, portfolio):
            return [Order("AAPL", Side.BUY, 1)]

    import pandas as pd
    frame = pd.DataFrame(
        {"open": [10], "high": [10], "low": [10], "close": [10], "volume": [100]},
        index=pd.date_range("2026-07-27", periods=1, freq="D"),
    )
    with patch("trading_agent.broker.paper_runner.load_bars", return_value=frame), patch(
        "trading_agent.broker.paper_runner.load_strategy", return_value=EmitsBuy()
    ):
        result = paper_tick(
            strategy_name="sma_cross",
            symbol="AAPL",
            broker=fake_broker,
            dry_run=False,
            contract=ExperimentContract(name="ambiguous-test"),
            graph=ExperimentGraph(tmp_path / "graph.sqlite3"),
        )

    assert "ambiguous" in (result.skipped_reason or "")
    assert len(result.execution_records) == 1
    record = result.execution_records[0]
    assert record.status == "submission_ambiguous"
    assert record.client_order_id
    assert record.decision.allowed is True


def test_pre_submit_intent_survives_post_submit_checkpoint_failure(tmp_path):
    from trading_agent.broker.alpaca import AccountSnapshot, AlpacaOrder, MarketSessionSnapshot
    from trading_agent.broker.paper_runner import paper_tick
    from trading_agent.experiment import ExperimentContract
    from trading_agent.experiment_graph import ExperimentGraph

    db_path = tmp_path / "graph.sqlite3"

    class BrokenSuccessGraph(ExperimentGraph):
        def record_execution_success(self, **kwargs):
            raise OSError("simulated disk failure")

    graph = BrokenSuccessGraph(db_path)
    broker = MagicMock()
    broker.account.return_value = AccountSnapshot(200, 200, 200, True, daily_pnl=0.0)
    broker.positions.return_value = []
    broker.trades_today.return_value = 0
    broker.asset_class.return_value = "US_EQUITY"
    now = datetime.now().astimezone() + timedelta(minutes=1)
    broker.market_session.return_value = MarketSessionSnapshot(
        True, now, now - timedelta(hours=2), now + timedelta(hours=2)
    )
    broker.submit_market_order.return_value = AlpacaOrder(
        "broker-accepted", "AAPL", "BUY", 1, "NEW", None, "now"
    )

    class EmitsBuy:
        def on_start(self, symbols):
            pass

        def on_bar(self, bar, portfolio):
            return [Order("AAPL", Side.BUY, 1)]

    import pandas as pd

    frame = pd.DataFrame(
        {"open": [10], "high": [10], "low": [10], "close": [10], "volume": [100]},
        index=pd.date_range("2026-07-27", periods=1, freq="D"),
    )
    with patch("trading_agent.broker.paper_runner.load_bars", return_value=frame), patch(
        "trading_agent.broker.paper_runner.load_strategy", return_value=EmitsBuy()
    ):
        result = paper_tick(
            strategy_name="sma_cross",
            symbol="AAPL",
            broker=broker,
            dry_run=False,
            contract=ExperimentContract(name="checkpoint-failure-test"),
            graph=graph,
        )

    broker.submit_market_order.assert_called_once()
    assert result.execution_records[0].status == "post_submit_checkpoint_error"
    assert "manual reconciliation required" in (result.skipped_reason or "")
    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT status, order_id FROM trade_intents").fetchone()
    assert row == ("approved_pending_submission", None)


@pytest.mark.parametrize("field", ["cash", "portfolio_value", "buying_power", "daily_pnl"])
def test_paper_tick_rejects_non_finite_account_snapshot_before_clamping(tmp_path, field):
    from trading_agent.broker.alpaca import AccountSnapshot
    from trading_agent.broker.paper_runner import paper_tick
    from trading_agent.experiment import ExperimentContract
    from trading_agent.experiment_graph import ExperimentGraph

    values = {"cash": 200.0, "portfolio_value": 200.0, "buying_power": 200.0, "daily_pnl": 0.0}
    values[field] = float("inf")
    fake_broker = MagicMock()
    fake_broker.account.return_value = AccountSnapshot(is_paper=True, **values)
    fake_broker.positions.return_value = []

    class EmitsBuy:
        def on_start(self, symbols):
            pass

        def on_bar(self, bar, portfolio):
            return [Order("AAPL", Side.BUY, 1)]

    import pandas as pd
    frame = pd.DataFrame(
        {"open": [10], "high": [10], "low": [10], "close": [10], "volume": [100]},
        index=pd.date_range("2026-07-27", periods=1, freq="D"),
    )
    with patch("trading_agent.broker.paper_runner.load_bars", return_value=frame), patch(
        "trading_agent.broker.paper_runner.load_strategy", return_value=EmitsBuy()
    ):
        result = paper_tick(
            strategy_name="sma_cross",
            symbol="AAPL",
            broker=fake_broker,
            dry_run=False,
            contract=ExperimentContract(name="invalid-account-test"),
            graph=ExperimentGraph(tmp_path / "graph.sqlite3"),
        )

    fake_broker.submit_market_order.assert_not_called()
    assert "non-finite" in (result.skipped_reason or "")
    assert result.execution_records[0].status == "rejected_invalid_snapshot"
