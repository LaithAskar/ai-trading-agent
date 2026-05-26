from __future__ import annotations

from trading_agent.agent.memory import Memory


def test_record_and_search_runs(tmp_path):
    db = tmp_path / "mem.sqlite3"
    mem = Memory(db)

    base_metrics = {
        "ending_equity": 110_000.0,
        "total_return_pct": 10.0,
        "cagr_pct": 5.0,
        "sharpe": 0.0,
        "max_drawdown_pct": -8.0,
        "num_fills": 10,
        "num_round_trips": 5,
        "win_rate_pct": 60.0,
    }

    for i, sh in enumerate([0.5, 1.2, 2.1, 0.9]):
        m = {**base_metrics, "sharpe": sh}
        mem.record_run(
            run_id=f"run{i}",
            strategy="sma_cross" if i < 3 else "other",
            symbol="AAPL" if i % 2 == 0 else "MSFT",
            start_date="2022-01-01",
            end_date="2024-12-31",
            params={"fast": 10 + i},
            starting_cash=100_000.0,
            metrics=m,
            artifact_dir=None,
        )

    all_runs = mem.search_runs()
    assert len(all_runs) == 4
    assert all_runs[0]["sharpe"] >= all_runs[-1]["sharpe"]

    sma_only = mem.search_runs(strategy="sma_cross")
    assert len(sma_only) == 3
    assert all(r["strategy"] == "sma_cross" for r in sma_only)

    aapl_only = mem.search_runs(symbol="aapl")
    assert all(r["symbol"] == "AAPL" for r in aapl_only)

    decent = mem.search_runs(min_sharpe=1.0)
    assert len(decent) == 2
    assert all(r["sharpe"] >= 1.0 for r in decent)


def test_record_session(tmp_path):
    db = tmp_path / "mem.sqlite3"
    mem = Memory(db)
    mem.record_session(
        session_id="s1",
        goal="test",
        mode="auto",
        model="claude-sonnet-4-6",
        iterations=3,
        input_tokens=1000,
        output_tokens=200,
        finished=True,
        final_summary="done",
    )
    import sqlite3

    rows = sqlite3.connect(db).execute("SELECT * FROM agent_sessions").fetchall()
    assert len(rows) == 1


def test_get_run_returns_none_for_missing(tmp_path):
    mem = Memory(tmp_path / "mem.sqlite3")
    assert mem.get_run("nope") is None
