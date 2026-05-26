"""V3 tests: cost tracking, session caps, replay, stats."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from trading_agent.agent.pricing import estimate_cost
from trading_agent.agent.stats import aggregate


# Reuse the loop test scaffolding
from tests.test_agent_loop import (
    FakeAnthropic,
    FakeResponse,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
)


# ---------- Pricing ----------

def test_pricing_sonnet_4_6_known():
    est = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert est.input_dollars == pytest.approx(3.0)
    assert est.output_dollars == pytest.approx(15.0)
    assert est.total_dollars == pytest.approx(18.0)


def test_pricing_unknown_model_falls_back_to_sonnet():
    est = estimate_cost("totally-made-up-model", 1_000_000, 1_000_000)
    assert est.total_dollars == pytest.approx(18.0)


def test_pricing_zero_tokens_costs_zero():
    est = estimate_cost("claude-opus-4-7", 0, 0)
    assert est.total_dollars == 0.0


# ---------- Session caps ----------

def _stub_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")


def test_dollar_cap_kills_loop(tmp_path, monkeypatch):
    _stub_paths(tmp_path, monkeypatch)

    # Each call burns a million output tokens on Opus 4-7 — $75 per iteration.
    expensive = FakeResponse(
        content=[
            FakeTextBlock("Spinning forever."),
            FakeToolUseBlock("t1", "list_strategies", {}),
        ],
        stop_reason="tool_use",
        usage=FakeUsage(inp=1_000_000, out=1_000_000),
    )
    fake = FakeAnthropic([expensive] * 50)

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="burn money",
            mode="auto",
            model="claude-opus-4-7",
            max_iters=50,
            max_session_dollars=1.00,
            max_session_tokens=10**9,
            api_key="fake-key",
        )

    assert not session.finished
    assert session.stopped_by is not None
    assert "dollar cap" in session.stopped_by
    # Should bail well before max_iters
    assert len(session.transcript) <= 2


def test_token_cap_kills_loop(tmp_path, monkeypatch):
    _stub_paths(tmp_path, monkeypatch)

    looper = FakeResponse(
        content=[
            FakeTextBlock("Looking again..."),
            FakeToolUseBlock("t1", "list_strategies", {}),
        ],
        stop_reason="tool_use",
        usage=FakeUsage(inp=5_000, out=5_000),
    )
    fake = FakeAnthropic([looper] * 20)

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="burn tokens",
            mode="auto",
            model="claude-sonnet-4-6",
            max_iters=20,
            max_session_dollars=100.0,
            max_session_tokens=30_000,
            api_key="fake-key",
        )

    assert not session.finished
    assert session.stopped_by is not None
    assert "token cap" in session.stopped_by
    total = session.input_tokens + session.output_tokens
    assert total > 30_000  # tripped because cumulative exceeded
    assert total < 50_000  # didn't run away


def test_cost_dollars_tracked_in_session(tmp_path, monkeypatch):
    _stub_paths(tmp_path, monkeypatch)

    one_shot = FakeResponse(
        content=[FakeTextBlock("Done immediately.")],
        stop_reason="end_turn",
        usage=FakeUsage(inp=1000, out=500),
    )
    fake = FakeAnthropic([one_shot])

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="test cost tracking",
            mode="auto",
            model="claude-sonnet-4-6",
            max_iters=5,
            api_key="fake-key",
        )

    expected = estimate_cost("claude-sonnet-4-6", 1000, 500).total_dollars
    assert session.cost_dollars == pytest.approx(expected)


def test_tool_duration_recorded(tmp_path, monkeypatch):
    _stub_paths(tmp_path, monkeypatch)

    responses = [
        FakeResponse(
            content=[
                FakeTextBlock("Listing."),
                FakeToolUseBlock("t1", "list_strategies", {}),
            ],
            stop_reason="tool_use",
            usage=FakeUsage(inp=100, out=50),
        ),
        FakeResponse(
            content=[FakeTextBlock("Done.")],
            stop_reason="end_turn",
            usage=FakeUsage(inp=100, out=50),
        ),
    ]
    fake = FakeAnthropic(responses)

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="test duration",
            mode="auto",
            model="claude-sonnet-4-6",
            max_iters=5,
            api_key="fake-key",
        )

    tool_entries = [e for e in session.transcript if e.tool_name]
    assert tool_entries, "expected at least one tool call"
    assert tool_entries[0].tool_duration_ms is not None
    assert tool_entries[0].tool_duration_ms >= 0


# ---------- Stats aggregation ----------

def test_aggregate_empty_dir(tmp_path):
    agg = aggregate(tmp_path)
    assert agg.total_sessions == 0
    assert agg.total_cost == 0.0
    assert agg.tools == {}


def test_aggregate_reads_session_jsons(tmp_path):
    s1 = {
        "session_id": "s1",
        "model": "claude-sonnet-4-6",
        "input_tokens": 1000,
        "output_tokens": 500,
        "cost_dollars": 0.0105,
        "finished": True,
        "stopped_by": None,
        "transcript": [
            {"iteration": 1, "tool_name": "list_strategies", "tool_duration_ms": 5,  "is_error": False},
            {"iteration": 1, "tool_name": "run_backtest",    "tool_duration_ms": 250, "is_error": False},
            {"iteration": 2, "tool_name": "list_strategies", "tool_duration_ms": 4,  "is_error": False},
            {"iteration": 3, "tool_name": "run_backtest",    "tool_duration_ms": 240, "is_error": True},
            {"iteration": 4, "tool_name": None,              "is_final": True},
        ],
    }
    (tmp_path / "s1.json").write_text(json.dumps(s1))

    agg = aggregate(tmp_path)
    assert agg.total_sessions == 1
    assert agg.total_cost == pytest.approx(0.0105)
    assert agg.tools["run_backtest"].calls == 2
    assert agg.tools["run_backtest"].errors == 1
    assert agg.tools["run_backtest"].error_rate_pct == pytest.approx(50.0)
    assert agg.tools["list_strategies"].calls == 2
    assert agg.tools["list_strategies"].errors == 0
    assert agg.tools["list_strategies"].avg_duration_ms == pytest.approx(4.5)


# ---------- Replay ----------

def test_replay_runs_against_current_code(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.replay.MEMORY_DB", tmp_path / "mem.sqlite3")

    fake_session = {
        "session_id": "abc",
        "transcript": [
            {
                "iteration": 1,
                "tool_name": "list_strategies",
                "tool_input": {},
                "tool_result": json.dumps(["sma_cross"]),
                "is_final": False,
            },
            {
                "iteration": 2,
                "tool_name": None,
                "is_final": True,
            },
        ],
    }
    p = tmp_path / "abc.json"
    p.write_text(json.dumps(fake_session))

    from trading_agent.agent.replay import replay_session

    steps = replay_session(p)
    assert len(steps) == 1
    assert steps[0].tool_name == "list_strategies"
    # We added rsi_mean_rev after that session was logged — current code returns
    # both strategies, so we expect drift.
    assert steps[0].drifted
    assert "rsi_mean_rev" in steps[0].new_result


def test_replay_no_drift_for_matching_output(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.replay.MEMORY_DB", tmp_path / "mem.sqlite3")

    fake_session = {
        "session_id": "abc",
        "transcript": [
            {
                "iteration": 1,
                "tool_name": "list_strategies",
                "tool_input": {},
                "tool_result": json.dumps(["filings_sentiment", "rsi_mean_rev", "sma_cross"]),
                "is_final": False,
            }
        ],
    }
    p = tmp_path / "abc.json"
    p.write_text(json.dumps(fake_session))

    from trading_agent.agent.replay import replay_session

    steps = replay_session(p)
    assert len(steps) == 1
    assert not steps[0].drifted
