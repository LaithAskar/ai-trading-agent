"""Ch2 context-engineering regression tests.

Verifies the agent loop applies prompt caching the way the book's chapter 2
prescribes (third-party/ai-agent-book, book-en/chapter2.md):
  - static cache breakpoint on the system prompt (with explicit 5m TTL),
  - a single ROLLING breakpoint on the newest message (not one per message,
    which would blow through the 4-breakpoint limit),
  - tool definition order fixed at session start (dynamic sorting kills the
    prefix cache from the first moved tool onward),
  - cache-aware cost accounting (cache reads ~0.1x, writes ~1.25x).

No credentials anywhere: the Anthropic client is patched out; the client
constructor receives None (resolved from the ambient environment, which is
empty under test).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_agent.agent.pricing import estimate_cost


class FakeUsage:
    def __init__(self, inp=100, out=20, cache_read=0, cache_write=0):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


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


def _run(fake, goal, max_iters=5):
    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        return run_agent(
            goal=goal,
            mode="auto",
            model="fake-model",
            max_iters=max_iters,
            api_key=None,
        )


def _cc_blocks(messages):
    """Yield (message_index, block) for dict blocks carrying cache_control."""
    for i, m in enumerate(messages):
        content = m["content"]
        blocks = content if isinstance(content, list) else []
        for b in blocks:
            if isinstance(b, dict) and "cache_control" in b:
                yield i, b


def test_system_breakpoint_with_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    fake = FakeAnthropic(
        [FakeResponse(content=[FakeTextBlock("Done.")], stop_reason="end_turn")]
    )
    _run(fake, "one shot")

    system = fake.messages.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_rolling_breakpoint_on_newest_message_only(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    fake = FakeAnthropic(
        [
            FakeResponse(
                content=[
                    FakeTextBlock("Checking strategies."),
                    FakeToolUseBlock("t1", "list_strategies", {}),
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(content=[FakeTextBlock("Done.")], stop_reason="end_turn"),
        ]
    )
    _run(fake, "two iteration goal")

    call0, call1 = fake.messages.calls[0], fake.messages.calls[1]

    # First call: the goal (string content) is converted to a single text
    # block carrying the rolling breakpoint.
    msg0 = call0["messages"]
    assert msg0[-1]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    # Second call: the breakpoint has MOVED to the newest message (the tool
    # results), and the older goal message carries no breakpoint anymore.
    msg1 = call1["messages"]
    marked = list(_cc_blocks(msg1))
    assert len(marked) == 1, f"expected exactly 1 rolling breakpoint, got {len(marked)}"
    marked_idx, marked_block = marked[0]
    assert marked_idx == len(msg1) - 1
    assert marked_block["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    # The goal message is pristine: same text, no cache marker left behind.
    # (cache_control is SDK metadata — it never enters the token stream, so
    # the earlier prefix stays byte-identical and remains cache-eligible.)
    goal_block = msg1[0]["content"][0]
    assert goal_block["text"] == "two iteration goal"
    assert "cache_control" not in goal_block

    # Total breakpoints per request stays within Anthropic's limit of 4:
    # 1 on system + 1 rolling on the newest message.
    total_cc = 1 + len(list(_cc_blocks(msg1)))
    assert total_cc <= 4


def test_tool_order_sorted_and_stable(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    looping = FakeResponse(
        content=[FakeTextBlock("Again."), FakeToolUseBlock("t", "list_strategies", {})],
        stop_reason="tool_use",
    )
    fake = FakeAnthropic([looping] * 3)
    _run(fake, "loop for tool order", max_iters=3)

    tools0 = fake.messages.calls[0]["tools"]
    names0 = [t["name"] for t in tools0]
    assert names0 == sorted(names0), "tool schemas must be sent in fixed sorted order"
    for call in fake.messages.calls[1:]:
        assert [t["name"] for t in call["tools"]] == names0, "tool order must not vary across requests"


def test_cache_aware_cost_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    usage = FakeUsage(inp=100, out=20, cache_read=500, cache_write=200)
    fake = FakeAnthropic(
        [FakeResponse(content=[FakeTextBlock("Done.")], stop_reason="end_turn", usage=usage)]
    )
    session = _run(fake, "cost check")

    assert session.cache_read_tokens == 500
    assert session.cache_write_tokens == 200

    # "fake-model" falls back to Sonnet pricing: 3.0 in / 15 out / 0.3 read / 3.75 write.
    expected = (
        100 * 3.0 + 500 * 0.3 + 200 * 3.75 + 20 * 15.0
    ) / 1_000_000
    assert session.cost_dollars == pytest.approx(expected, rel=1e-9)

    # And the cached version must be cheaper than billing everything as
    # fresh input (the old cache-blind math).
    naive = (100 + 500 + 200) * 3.0 / 1_000_000 + 20 * 15.0 / 1_000_000
    assert session.cost_dollars < naive


def test_estimate_cost_cache_multipliers():
    est = estimate_cost(
        "claude-sonnet-4-6",
        1_000_000,
        0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert est.total_dollars == pytest.approx(3.0 + 0.3 + 3.75, rel=1e-9)

    plain = estimate_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert plain.total_dollars == pytest.approx(3.0, rel=1e-9)
