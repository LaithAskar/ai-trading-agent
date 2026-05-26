from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    starting_cash   REAL NOT NULL,
    ending_equity   REAL NOT NULL,
    total_return_pct REAL NOT NULL,
    cagr_pct        REAL NOT NULL,
    sharpe          REAL NOT NULL,
    max_drawdown_pct REAL NOT NULL,
    num_fills       INTEGER NOT NULL,
    num_round_trips INTEGER NOT NULL,
    win_rate_pct    REAL NOT NULL,
    artifact_dir    TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy);
CREATE INDEX IF NOT EXISTS idx_runs_symbol   ON runs(symbol);
CREATE INDEX IF NOT EXISTS idx_runs_sharpe   ON runs(sharpe);

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    goal         TEXT NOT NULL,
    mode         TEXT NOT NULL,
    model        TEXT NOT NULL,
    iterations   INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    finished     INTEGER NOT NULL,
    final_summary TEXT
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


class Memory:
    """Structured run memory backed by SQLite.

    Why structured instead of vector: backtest runs have well-defined fields
    (strategy, symbol, dates, metrics). Filtering by sharpe > X, symbol = Y,
    or strategy = Z is far more useful than semantic similarity. Vector search
    becomes valuable later if/when we store free-form analysis notes.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        with _conn(self.db_path) as c:
            c.executescript(SCHEMA)

    def record_run(
        self,
        *,
        run_id: str,
        strategy: str,
        symbol: str,
        start_date: str,
        end_date: str,
        params: dict,
        starting_cash: float,
        metrics: dict,
        artifact_dir: str | None,
    ) -> None:
        row = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": strategy,
            "symbol": symbol.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "params_json": json.dumps(params, sort_keys=True),
            "starting_cash": starting_cash,
            "ending_equity": metrics["ending_equity"],
            "total_return_pct": metrics["total_return_pct"],
            "cagr_pct": metrics["cagr_pct"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "num_fills": metrics["num_fills"],
            "num_round_trips": metrics["num_round_trips"],
            "win_rate_pct": metrics["win_rate_pct"],
            "artifact_dir": artifact_dir,
        }
        with _conn(self.db_path) as c:
            c.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, created_at, strategy, symbol, start_date, end_date,
                 params_json, starting_cash, ending_equity, total_return_pct,
                 cagr_pct, sharpe, max_drawdown_pct, num_fills, num_round_trips,
                 win_rate_pct, artifact_dir)
                VALUES (:run_id, :created_at, :strategy, :symbol, :start_date,
                        :end_date, :params_json, :starting_cash, :ending_equity,
                        :total_return_pct, :cagr_pct, :sharpe, :max_drawdown_pct,
                        :num_fills, :num_round_trips, :win_rate_pct, :artifact_dir)
                """,
                row,
            )

    def search_runs(
        self,
        *,
        strategy: str | None = None,
        symbol: str | None = None,
        min_sharpe: float | None = None,
        order_by: str = "sharpe",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        allowed_order = {"sharpe", "cagr_pct", "total_return_pct", "max_drawdown_pct", "created_at"}
        if order_by not in allowed_order:
            order_by = "sharpe"

        sql = "SELECT * FROM runs WHERE 1=1"
        params: dict = {}
        if strategy:
            sql += " AND strategy = :strategy"
            params["strategy"] = strategy
        if symbol:
            sql += " AND symbol = :symbol"
            params["symbol"] = symbol.upper()
        if min_sharpe is not None:
            sql += " AND sharpe >= :min_sharpe"
            params["min_sharpe"] = min_sharpe
        sql += f" ORDER BY {order_by} DESC LIMIT :limit"
        params["limit"] = limit

        with _conn(self.db_path) as c:
            rows = c.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with _conn(self.db_path) as c:
            row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def record_session(
        self,
        *,
        session_id: str,
        goal: str,
        mode: str,
        model: str,
        iterations: int,
        input_tokens: int,
        output_tokens: int,
        finished: bool,
        final_summary: str | None,
    ) -> None:
        with _conn(self.db_path) as c:
            c.execute(
                """
                INSERT OR REPLACE INTO agent_sessions
                (session_id, created_at, goal, mode, model, iterations,
                 input_tokens, output_tokens, finished, final_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    goal,
                    mode,
                    model,
                    iterations,
                    input_tokens,
                    output_tokens,
                    1 if finished else 0,
                    final_summary,
                ),
            )
