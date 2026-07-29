from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trading_agent.autonomous import _score, run_autonomous_daily
from trading_agent.broker.paper_runner import TradeExecutionRecord
from trading_agent.core.orders import Order, Side
from trading_agent.experiment import ExperimentContract, GateDecision
from trading_agent.experiment_graph import ExperimentGraph


class FakeMetrics:
    sharpe = 1.2
    cagr_pct = 12.0
    total_return_pct = 25.0
    max_drawdown_pct = -8.0
    num_fills = 4


class FakeBenchmark:
    sharpe = 0.7
    cagr_pct = 8.0


def fake_backtest(**kwargs):
    return SimpleNamespace(
        run_id=f"bt_{kwargs['strategy_name']}_{kwargs['symbol']}",
        strategy=kwargs["strategy_name"],
        symbol=kwargs["symbol"],
        start=kwargs["start"],
        end=kwargs["end"],
        params=kwargs.get("params") or {},
        metrics=FakeMetrics(),
        benchmark=FakeBenchmark(),
        sharpe_p_value=0.04,
        artifact_dir=Path(__file__).resolve().parents[1] / "data" / "results" / "fake",
    )


def test_score_rejects_non_finite_evidence():
    run = fake_backtest(strategy_name="sma_cross", symbol="AAPL", start="a", end="b")
    run.metrics.sharpe = float("nan")

    score, verdict = _score(run)  # type: ignore[arg-type]

    assert verdict == "reject_invalid_metrics"
    assert score == -1_000_000_000.0


def test_autonomous_daily_records_candidate_and_trade_intent(tmp_path):
    graph = ExperimentGraph(tmp_path / "graph.sqlite3")
    contract = ExperimentContract(name="graph-test")
    order = Order("AAPL", Side.BUY, 1)
    decision = GateDecision(True, "allowed", 10.0, 10.0, 10.0)
    fake_tick = SimpleNamespace(
        proposed_orders=[order],
        dry_run=True,
        skipped_reason=None,
        rejected_orders=[],
        bars_seen=10,
        execution_records=[
            TradeExecutionRecord(order, 10.0, decision, "submitted", "client-1", "broker-1")
        ],
    )

    with patch("trading_agent.autonomous.run_backtest", side_effect=fake_backtest), patch(
        "trading_agent.autonomous.paper_tick", return_value=fake_tick
    ):
        result = run_autonomous_daily(
            contract=contract,
            symbols=["AAPL"],
            strategies=["sma_cross"],
            lookback_days=30,
            graph=graph,
        )

    assert result.status == "trade_intents_proposed"
    assert result.report_path is not None
    assert Path(result.report_path).exists()
    assert result.selected_strategy == "sma_cross"
    latest = graph.latest_runs("graph-test", limit=1)
    assert latest[0]["run_id"] == result.run_id
    assert latest[0]["selected_symbol"] == "AAPL"
    intents = graph.trade_intents(result.run_id)
    assert intents[0]["reference_price"] == 10.0
    assert intents[0]["gate_allowed"] == 1
    assert intents[0]["gate_reason"] == "allowed"
    assert intents[0]["client_order_id"] == "client-1"
    assert intents[0]["order_id"] == "broker-1"


def test_candidate_is_not_selectable_when_audit_record_fails(tmp_path):
    graph = ExperimentGraph(tmp_path / "graph.sqlite3")
    contract = ExperimentContract(name="audit-failure-test")

    with patch("trading_agent.autonomous.run_backtest", side_effect=fake_backtest), patch.object(
        graph, "record_candidate", side_effect=RuntimeError("database unavailable")
    ), patch("trading_agent.autonomous.paper_tick") as mock_tick:
        result = run_autonomous_daily(
            contract=contract,
            symbols=["AAPL"],
            strategies=["sma_cross"],
            lookback_days=30,
            graph=graph,
        )

    mock_tick.assert_not_called()
    assert result.status == "no_candidates"
    assert result.candidates == []
    assert result.summary["failure_count"] == 1
