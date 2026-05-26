"""Loop driver tests using a fake Anthropic client.

We don't hit the real API in unit tests. We just verify the loop drives
correctly given a scripted sequence of responses.
"""
from __future__ import annotations

import types
from unittest.mock import patch

import pytest


class FakeUsage:
    def __init__(self, inp=100, out=20):
        self.input_tokens = inp
        self.output_tokens = out


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, tool_id, name, args):
        self.id = tool_id
        self.name = name
        self.input = args


class FakeResponse:
    def __init__(self, content, stop_reason, usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or FakeUsage()


class FakeMessages:
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeAnthropic:
    def __init__(self, scripted_responses):
        self.messages = FakeMessages(scripted_responses)


def test_loop_terminates_on_end_turn_with_no_tool_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3"
    )
    monkeypatch.setattr(
        "trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs"
    )

    responses = [
        FakeResponse(
            content=[FakeTextBlock("All done. Here's your answer.")],
            stop_reason="end_turn",
        )
    ]
    fake = FakeAnthropic(responses)

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="test goal",
            mode="auto",
            model="fake-model",
            max_iters=5,
            api_key="fake-key",
        )

    assert session.finished
    assert "All done" in session.final_summary
    assert len(session.transcript) == 1
    assert session.transcript[0].is_final


def test_loop_executes_tool_then_terminates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3"
    )
    monkeypatch.setattr(
        "trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs"
    )

    responses = [
        FakeResponse(
            content=[
                FakeTextBlock("I'll list strategies first."),
                FakeToolUseBlock("t1", "list_strategies", {}),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[FakeTextBlock("Found sma_cross. Done.")],
            stop_reason="end_turn",
        ),
    ]
    fake = FakeAnthropic(responses)

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="what strategies exist?",
            mode="auto",
            model="fake-model",
            max_iters=5,
            api_key="fake-key",
        )

    assert session.finished
    assert len(session.transcript) == 2
    assert session.transcript[0].tool_name == "list_strategies"
    assert "sma_cross" in session.transcript[0].tool_result


def test_loop_stops_at_max_iters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3"
    )
    monkeypatch.setattr(
        "trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs"
    )

    looping_response = FakeResponse(
        content=[
            FakeTextBlock("Looking again..."),
            FakeToolUseBlock("t_loop", "list_strategies", {}),
        ],
        stop_reason="tool_use",
    )
    fake = FakeAnthropic([looping_response] * 10)

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="never terminate",
            mode="auto",
            model="fake-model",
            max_iters=3,
            api_key="fake-key",
        )

    assert not session.finished
    assert len(session.transcript) == 3
    assert "max iterations" in session.final_summary
