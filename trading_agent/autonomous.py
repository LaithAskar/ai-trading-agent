from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .backtest.runner import BacktestRun, run_backtest
from .broker.paper_runner import PaperTickResult, paper_tick
from .config import DATA_DIR, PROJECT_ROOT
from .experiment import ExperimentContract
from .experiment_graph import BacktestCandidateRecord, ExperimentGraph

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
DEFAULT_STRATEGIES = ["sma_cross", "rsi_mean_rev"]
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "sma_cross": {"fast": 20, "slow": 50},
    "rsi_mean_rev": {"period": 14, "oversold": 30.0, "overbought": 70.0},
}


@dataclass
class AutonomousRunResult:
    run_id: str
    experiment_name: str
    selected_strategy: str | None
    selected_symbol: str | None
    candidates: list[BacktestCandidateRecord]
    paper_tick: PaperTickResult | None
    status: str
    summary: dict[str, Any]
    report_path: str | None = None


def write_daily_report(result: AutonomousRunResult) -> str:
    report_dir = DATA_DIR / "experiment_reports" / result.experiment_name
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{result.run_id}.md"
    lines = [
        f"# Autonomous Trading Daily Run — {result.experiment_name}",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Status: `{result.status}`",
        f"- Selected: `{result.selected_strategy or 'none'}` / `{result.selected_symbol or 'none'}`",
        "",
        "## Candidate backtests",
        "",
        "| Strategy | Symbol | Score | Sharpe | CAGR | Max DD | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in sorted(result.candidates, key=lambda row: row.score, reverse=True):
        lines.append(
            f"| {c.strategy} | {c.symbol} | {c.score:.3f} | {c.sharpe:.2f} | "
            f"{c.cagr_pct:.2f}% | {c.max_drawdown_pct:.2f}% | {c.verdict} |"
        )
    lines.extend(["", "## Trade intents", ""])
    if result.paper_tick is None or not result.paper_tick.proposed_orders:
        reason = None if result.paper_tick is None else result.paper_tick.skipped_reason
        lines.append(f"No trade intents. Reason: {reason or 'none'}")
    else:
        lines.extend(["| Symbol | Side | Quantity |", "|---|---:|---:|"])
        for order in result.paper_tick.proposed_orders:
            lines.append(f"| {order.symbol} | {order.side.value} | {order.quantity:g} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "Autonomous v0: no trade-by-trade approval. Execution remains bounded by the experiment contract.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _default_dates(lookback_days: int) -> tuple[str, str]:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _score(run: BacktestRun) -> tuple[float, str]:
    """Simple deterministic v0 score.

    Prefer positive Sharpe/CAGR, penalize drawdown, and require benchmark context.
    This is deliberately conservative and transparent; later versions can swap
    in a richer graph-aware ranker.
    """
    sharpe = run.metrics.sharpe
    cagr = run.metrics.cagr_pct
    max_dd = abs(run.metrics.max_drawdown_pct)
    benchmark = run.benchmark
    benchmark_sharpe = benchmark.sharpe if benchmark else 0.0
    benchmark_cagr = benchmark.cagr_pct if benchmark else 0.0
    evidence = (
        sharpe,
        cagr,
        max_dd,
        benchmark_sharpe,
        benchmark_cagr,
        run.sharpe_p_value,
    )
    if not all(math.isfinite(value) for value in evidence):
        return -1_000_000_000.0, "reject_invalid_metrics"
    score = (sharpe - benchmark_sharpe) + (cagr - benchmark_cagr) / 25.0 - max_dd / 100.0

    if run.metrics.num_fills == 0:
        verdict = "reject_no_trades"
        score -= 10.0
    elif sharpe <= 0:
        verdict = "reject_negative_sharpe"
    elif benchmark and sharpe < benchmark.sharpe and cagr < benchmark.cagr_pct:
        verdict = "reject_loses_to_benchmark"
    elif run.sharpe_p_value >= 0.10:
        verdict = "candidate_statistically_weak"
    else:
        verdict = "candidate"
    return score, verdict


def _select_candidate(candidates: list[BacktestCandidateRecord]) -> BacktestCandidateRecord | None:
    """Select only a fully promotable candidate.

    Statistically weak candidates remain visible in the evidence ledger but are
    not eligible to drive an autonomous trade intent. This prevents a high raw
    score from outranking a lower-scoring candidate that actually passed every
    promotion gate.
    """
    promotable = [candidate for candidate in candidates if candidate.verdict == "candidate"]
    return max(promotable, key=lambda candidate: candidate.score, default=None)


def run_autonomous_daily(
    *,
    contract: ExperimentContract,
    symbols: list[str] | None = None,
    strategies: list[str] | None = None,
    lookback_days: int = 365 * 2,
    execute: bool = False,
    broker=None,
    graph: ExperimentGraph | None = None,
) -> AutonomousRunResult:
    """Run one deterministic autonomous daily cycle.

    v0 does not ask for trade-by-trade approval. It evaluates candidates,
    picks the best-scoring candidate, records the evidence graph, and runs one
    paper tick for the selected strategy/symbol. `execute=False` keeps the
    broker path dry-run; `execute=True` may submit to Alpaca paper, still gated
    by the experiment contract.
    """
    graph = graph or ExperimentGraph()
    run_id = graph.new_run_id()
    symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
    strategies = strategies or DEFAULT_STRATEGIES
    start, end = _default_dates(lookback_days)

    candidates: list[BacktestCandidateRecord] = []
    failures: list[dict[str, str]] = []
    for strategy in strategies:
        params = DEFAULT_PARAMS.get(strategy, {})
        for symbol in symbols:
            try:
                bt = run_backtest(
                    strategy_name=strategy,
                    symbol=symbol,
                    start=start,
                    end=end,
                    params=params,
                    starting_cash=contract.capital_cap_usd,
                    write_artifacts=True,
                )
                score, verdict = _score(bt)
                benchmark = bt.benchmark
                record = BacktestCandidateRecord(
                    strategy=bt.strategy,
                    symbol=bt.symbol,
                    start_date=bt.start,
                    end_date=bt.end,
                    params=bt.params,
                    run_ref=bt.run_id,
                    sharpe=bt.metrics.sharpe,
                    cagr_pct=bt.metrics.cagr_pct,
                    total_return_pct=bt.metrics.total_return_pct,
                    max_drawdown_pct=bt.metrics.max_drawdown_pct,
                    benchmark_sharpe=None if benchmark is None else benchmark.sharpe,
                    benchmark_cagr_pct=None if benchmark is None else benchmark.cagr_pct,
                    score=score,
                    verdict=verdict,
                    artifact_dir=str(bt.artifact_dir.relative_to(PROJECT_ROOT)),
                )
                graph.record_candidate(run_id=run_id, experiment_name=contract.name, record=record)
                candidates.append(record)
            except Exception as exc:
                failures.append({"strategy": strategy, "symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})

    selected = _select_candidate(candidates)

    tick_result: PaperTickResult | None = None
    status = "no_candidates"
    if selected is not None:
        status = "selected"
        tick_result = paper_tick(
            strategy_name=selected.strategy,
            symbol=selected.symbol,
            params=selected.params,
            lookback_days=lookback_days,
            broker=broker if execute else None,
            dry_run=not execute,
            contract=contract,
            graph=graph,
            run_id=run_id,
        )
        for execution in tick_result.execution_records:
            graph.record_trade_intent(
                run_id=run_id,
                experiment_name=contract.name,
                strategy=selected.strategy,
                order=execution.order,
                reference_price=execution.reference_price,
                decision=execution.decision,
                status=execution.status,
                order_id=execution.broker_order_id,
                client_order_id=execution.client_order_id,
            )
        if tick_result.proposed_orders:
            execution_statuses = {record.status for record in tick_result.execution_records}
            if tick_result.dry_run:
                status = "trade_intents_proposed"
            elif execution_statuses & {"submitted", "idempotent_replay"}:
                status = "orders_submitted"
            else:
                status = "trade_intents_rejected"
        elif tick_result.skipped_reason:
            status = "no_trade_intents"

    summary = {
        "candidate_count": len(candidates),
        "failure_count": len(failures),
        "failures": failures[:10],
        "selected": None
        if selected is None
        else {
            "strategy": selected.strategy,
            "symbol": selected.symbol,
            "score": selected.score,
            "verdict": selected.verdict,
            "sharpe": selected.sharpe,
            "cagr_pct": selected.cagr_pct,
            "benchmark_sharpe": selected.benchmark_sharpe,
            "benchmark_cagr_pct": selected.benchmark_cagr_pct,
        },
        "paper_tick": None
        if tick_result is None
        else {
            "bars_seen": tick_result.bars_seen,
            "proposed_orders": [
                {"symbol": o.symbol, "side": o.side.value, "quantity": o.quantity}
                for o in tick_result.proposed_orders
            ],
            "dry_run": tick_result.dry_run,
            "skipped_reason": tick_result.skipped_reason,
            "rejected_orders": tick_result.rejected_orders,
            "execution_records": [
                {
                    "symbol": record.order.symbol,
                    "side": record.order.side.value,
                    "quantity": record.order.quantity,
                    "reference_price": record.reference_price,
                    "gate_allowed": record.decision.allowed,
                    "gate_reason": record.decision.reason,
                    "status": record.status,
                    "client_order_id": record.client_order_id,
                    "broker_order_id": record.broker_order_id,
                }
                for record in tick_result.execution_records
            ],
        },
    }
    graph.record_experiment_run(
        run_id=run_id,
        contract=contract,
        symbols=symbols,
        strategies=strategies,
        status=status,
        selected_strategy=None if selected is None else selected.strategy,
        selected_symbol=None if selected is None else selected.symbol,
        summary=summary,
    )
    result = AutonomousRunResult(
        run_id=run_id,
        experiment_name=contract.name,
        selected_strategy=None if selected is None else selected.strategy,
        selected_symbol=None if selected is None else selected.symbol,
        candidates=candidates,
        paper_tick=tick_result,
        status=status,
        summary=summary,
    )
    result.report_path = write_daily_report(result)
    return result
