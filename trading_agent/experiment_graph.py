from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DATA_DIR
from .core.orders import Order
from .experiment import ExperimentContract, GateDecision

EXPERIMENT_GRAPH_DB = DATA_DIR / "experiment_graph.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id          TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    mode            TEXT NOT NULL,
    status          TEXT NOT NULL,
    symbols_json    TEXT NOT NULL,
    strategies_json TEXT NOT NULL,
    selected_strategy TEXT,
    selected_symbol TEXT,
    summary_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiment_runs_name_created
ON experiment_runs(experiment_name, created_at);

CREATE TABLE IF NOT EXISTS backtest_candidates (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    run_ref         TEXT,
    sharpe          REAL NOT NULL,
    cagr_pct        REAL NOT NULL,
    total_return_pct REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    benchmark_sharpe REAL,
    benchmark_cagr_pct REAL,
    score           REAL NOT NULL,
    verdict         TEXT NOT NULL,
    artifact_dir    TEXT
);

CREATE INDEX IF NOT EXISTS idx_backtest_candidates_run
ON backtest_candidates(run_id);

CREATE TABLE IF NOT EXISTS trade_intents (
    intent_id       TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        REAL NOT NULL,
    reference_price REAL,
    estimated_notional REAL,
    gate_allowed    INTEGER,
    gate_reason     TEXT,
    status          TEXT NOT NULL,
    order_id        TEXT,
    client_order_id TEXT,
    raw_json        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_intents_run
ON trade_intents(run_id);

CREATE TABLE IF NOT EXISTS experiment_state (
    experiment_name TEXT PRIMARY KEY,
    baseline_portfolio_value REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_executions (
    client_order_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    broker_order_id TEXT,
    response_json TEXT,
    updated_at TEXT
);
"""


@contextmanager
def _conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


@dataclass(frozen=True)
class BacktestCandidateRecord:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    params: dict[str, Any]
    run_ref: str | None
    sharpe: float
    cagr_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    benchmark_sharpe: float | None
    benchmark_cagr_pct: float | None
    score: float
    verdict: str
    artifact_dir: str | None


@dataclass(frozen=True)
class ExecutionClaim:
    acquired: bool
    status: str
    broker_order_id: str | None = None


class ExperimentGraph:
    """SQLite evidence graph for the autonomous trading experiment.

    This is intentionally relational first: it preserves provenance cheaply and
    can later be projected into a graph database if the relationship queries
    outgrow SQLite.
    """

    def __init__(self, db_path: Path = EXPERIMENT_GRAPH_DB):
        self.db_path = db_path
        with _conn(self.db_path) as c:
            c.executescript(SCHEMA)
            columns = {row[1] for row in c.execute("PRAGMA table_info(trade_intents)")}
            if "client_order_id" not in columns:
                c.execute("ALTER TABLE trade_intents ADD COLUMN client_order_id TEXT")
            execution_columns = {row[1] for row in c.execute("PRAGMA table_info(order_executions)")}
            if "updated_at" not in execution_columns:
                c.execute("ALTER TABLE order_executions ADD COLUMN updated_at TEXT")
                c.execute("UPDATE order_executions SET updated_at = created_at WHERE updated_at IS NULL")

    def new_run_id(self) -> str:
        return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def record_experiment_run(
        self,
        *,
        run_id: str,
        contract: ExperimentContract,
        symbols: list[str],
        strategies: list[str],
        status: str,
        selected_strategy: str | None,
        selected_symbol: str | None,
        summary: dict[str, Any],
    ) -> None:
        with _conn(self.db_path) as c:
            c.execute(
                """
                INSERT OR REPLACE INTO experiment_runs
                (run_id, experiment_name, created_at, mode, status, symbols_json,
                 strategies_json, selected_strategy, selected_symbol, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    contract.name,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    contract.mode,
                    status,
                    json.dumps(symbols),
                    json.dumps(strategies),
                    selected_strategy,
                    selected_symbol,
                    json.dumps(summary, sort_keys=True, default=str),
                ),
            )

    def record_candidate(
        self,
        *,
        run_id: str,
        experiment_name: str,
        record: BacktestCandidateRecord,
    ) -> str:
        row_id = f"cand_{uuid.uuid4().hex[:12]}"
        with _conn(self.db_path) as c:
            c.execute(
                """
                INSERT INTO backtest_candidates
                (id, run_id, experiment_name, created_at, strategy, symbol,
                 start_date, end_date, params_json, run_ref, sharpe, cagr_pct,
                 total_return_pct, max_drawdown_pct, benchmark_sharpe,
                 benchmark_cagr_pct, score, verdict, artifact_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    run_id,
                    experiment_name,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    record.strategy,
                    record.symbol.upper(),
                    record.start_date,
                    record.end_date,
                    json.dumps(record.params, sort_keys=True),
                    record.run_ref,
                    record.sharpe,
                    record.cagr_pct,
                    record.total_return_pct,
                    record.max_drawdown_pct,
                    record.benchmark_sharpe,
                    record.benchmark_cagr_pct,
                    record.score,
                    record.verdict,
                    record.artifact_dir,
                ),
            )
        return row_id

    def record_trade_intent(
        self,
        *,
        run_id: str,
        experiment_name: str,
        strategy: str,
        order: Order,
        reference_price: float | None,
        decision: GateDecision | None,
        status: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> str:
        intent_id = (
            f"intent_{uuid.uuid5(uuid.NAMESPACE_URL, 'trade-intent:' + client_order_id).hex[:24]}"
            if client_order_id
            else f"intent_{uuid.uuid4().hex[:12]}"
        )
        estimated_notional = order.quantity * reference_price if reference_price is not None else None
        raw = {
            "order": {"symbol": order.symbol, "side": order.side.value, "quantity": order.quantity},
            "decision": asdict(decision) if decision is not None else None,
            "order_id": order_id,
            "client_order_id": client_order_id,
        }
        with _conn(self.db_path) as c:
            c.execute(
                """
                INSERT INTO trade_intents
                (intent_id, run_id, experiment_name, created_at, strategy, symbol,
                 side, quantity, reference_price, estimated_notional, gate_allowed,
                 gate_reason, status, order_id, client_order_id, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(intent_id) DO UPDATE SET
                    gate_allowed = excluded.gate_allowed,
                    gate_reason = excluded.gate_reason,
                    status = excluded.status,
                    order_id = excluded.order_id,
                    raw_json = excluded.raw_json
                """,
                (
                    intent_id,
                    run_id,
                    experiment_name,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    strategy,
                    order.symbol.upper(),
                    order.side.value,
                    order.quantity,
                    reference_price,
                    estimated_notional,
                    None if decision is None else int(decision.allowed),
                    None if decision is None else decision.reason,
                    status,
                    order_id,
                    client_order_id,
                    json.dumps(raw, sort_keys=True, default=str),
                ),
            )
        return intent_id

    def baseline_portfolio_value(self, experiment_name: str, current_value: float) -> float:
        """Persist the first observed paper-account value as the loss baseline."""
        with _conn(self.db_path) as c:
            c.execute(
                "INSERT OR IGNORE INTO experiment_state VALUES (?, ?, ?)",
                (experiment_name, current_value, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            row = c.execute(
                "SELECT baseline_portfolio_value FROM experiment_state WHERE experiment_name = ?",
                (experiment_name,),
            ).fetchone()
        return float(row[0])

    def successful_execution(self, client_order_id: str) -> dict[str, Any] | None:
        with _conn(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM order_executions WHERE client_order_id = ? AND status = 'submitted'",
                (client_order_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def claim_execution(
        self,
        *,
        client_order_id: str,
        experiment_name: str,
        run_id: str,
    ) -> ExecutionClaim:
        """Atomically elect one submitter for a broker order intent.

        The pending row is committed before broker I/O. Concurrent workers see
        the existing row and must not submit. Only a confirmed broker response
        or lookup may transition the row to ``submitted``.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            c.execute("BEGIN IMMEDIATE")
            inserted = c.execute(
                """
                INSERT OR IGNORE INTO order_executions
                (client_order_id, experiment_name, run_id, created_at, status,
                 broker_order_id, response_json, updated_at)
                VALUES (?, ?, ?, ?, 'pending', NULL, NULL, ?)
                """,
                (client_order_id, experiment_name, run_id, now, now),
            ).rowcount
            row = c.execute(
                "SELECT status, broker_order_id FROM order_executions WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()
        if row is None:
            raise RuntimeError("execution claim disappeared after insert")
        return ExecutionClaim(bool(inserted), str(row["status"]), row["broker_order_id"])

    def record_execution_success(
        self,
        *,
        client_order_id: str,
        experiment_name: str,
        run_id: str,
        broker_order_id: str,
        response: dict[str, Any],
    ) -> None:
        with _conn(self.db_path) as c:
            c.execute(
                """
                INSERT INTO order_executions
                (client_order_id, experiment_name, run_id, created_at, status,
                 broker_order_id, response_json, updated_at)
                VALUES (?, ?, ?, ?, 'submitted', ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    status = 'submitted',
                    broker_order_id = excluded.broker_order_id,
                    response_json = excluded.response_json,
                    updated_at = excluded.updated_at
                """,
                (
                    client_order_id,
                    experiment_name,
                    run_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    broker_order_id,
                    json.dumps(response, sort_keys=True, default=str),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def trade_intents(self, run_id: str) -> list[dict[str, Any]]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM trade_intents WHERE run_id = ? ORDER BY created_at, intent_id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_runs(self, experiment_name: str, limit: int = 10) -> list[dict[str, Any]]:
        with _conn(self.db_path) as c:
            rows = c.execute(
                """
                SELECT * FROM experiment_runs
                WHERE experiment_name = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (experiment_name, limit),
            ).fetchall()
        return [dict(r) for r in rows]
