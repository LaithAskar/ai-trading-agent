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


def test_openrouter_cache_dialect(tmp_path, monkeypatch):
    """OpenRouter routing: plain ephemeral markers (no Anthropic ttl field)."""
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    fake = FakeAnthropic(
        [FakeResponse(content=[FakeTextBlock("Done.")], stop_reason="end_turn")]
    )
    # client_kwargs is stubbed because the real one (correctly) refuses an
    # empty OpenRouter key; the client itself is faked, so no key is used.
    with (
        patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake),
        patch("trading_agent.agent.loop.client_kwargs", return_value={"api_key": "test"}),
    ):
        from trading_agent.agent.loop import run_agent

        run_agent(
            goal="or dialect",
            mode="auto",
            model="openai/gpt-5.6-sol",
            max_iters=1,
            provider="openrouter",
            api_key=None,
        )

    call = fake.messages.calls[0]
    assert call["model"] == "openai/gpt-5.6-sol"  # vendor slug passed through
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_resolve_model_vendor_qualified_passthrough():
    from trading_agent.llm.client import resolve_model

    assert resolve_model("openai/gpt-5.6-sol", "openrouter") == "openai/gpt-5.6-sol"
    assert resolve_model("qwen/qwen3.8-flash", "openrouter") == "qwen/qwen3.8-flash"
    # Bare ids keep the legacy mapping/namespace behavior.
    assert resolve_model("claude-sonnet-4-6", "openrouter") == "anthropic/claude-sonnet-4.6"
    assert resolve_model("claude-future-9", "openrouter") == "anthropic/claude-future-9"


def test_opencode_client_kwargs_and_slugs(monkeypatch):
    from trading_agent.llm import client as orc

    monkeypatch.setattr(orc, "OPENCODE_BASE_URL", "https://opencode.example/v1")
    kw = orc.client_kwargs("oc-test", "opencode")
    assert kw["base_url"] == "https://opencode.example/v1"
    assert kw["default_headers"]["Authorization"] == "Bearer oc-test"
    with pytest.raises(ValueError):
        orc.client_kwargs(None, "opencode")
    # Bare slugs resolve verbatim on the opencode gateway.
    assert orc.resolve_model("kimi-k3", "opencode") == "kimi-k3"
    assert orc.resolve_model("deepseek-v4-flash", "opencode") == "deepseek-v4-flash"


def test_opencode_cache_dialect(tmp_path, monkeypatch):
    """opencode routing: plain ephemeral markers, bare slug, correct model sent."""
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    fake = FakeAnthropic(
        [FakeResponse(content=[FakeTextBlock("Done.")], stop_reason="end_turn")]
    )
    with (
        patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake),
        patch("trading_agent.agent.loop.client_kwargs", return_value={"api_key": "test"}),
    ):
        from trading_agent.agent.loop import run_agent

        run_agent(
            goal="opencode dialect",
            mode="auto",
            model="kimi-k3",
            max_iters=1,
            provider="opencode",
            api_key=None,
        )

    call = fake.messages.calls[0]
    assert call["model"] == "kimi-k3"  # bare gateway slug passed through
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_opencode_pricing_rows():
    # Gateway slugs price from the OPENCODE table, not the Sonnet fallback.
    est = estimate_cost("deepseek-v4-flash", 1_000_000, 0)
    assert est.total_dollars == pytest.approx(0.08092, rel=1e-6)
    # Unknown opencode slugs still fall back (documented mispricing).
    assert estimate_cost("totally-unknown-gw-model", 1_000_000, 0).total_dollars == pytest.approx(3.0, rel=1e-9)


# ---------- OpenAI-format shim (opencode-openai / ollama) ----------


class _FakeHttpxResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_shim_converts_anthropic_request_to_openai_wire(monkeypatch):
    from trading_agent.llm.openai_shim import OpenAICompatClient

    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeHttpxResponse(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "I will check.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "t1", "arguments": '{"a": 2}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 123},
                },
            }
        )

    monkeypatch.setattr("trading_agent.llm.openai_shim.httpx.post", fake_post)
    client = OpenAICompatClient(base_url="http://127.0.0.1:11434", api_key="x")

    resp = client.messages_create(
        model="grok-4.5",
        max_tokens=100,
        system=[{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}],
        tools=[{"name": "t1", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking"},
                    {"type": "tool_use", "id": "t1", "name": "t1", "input": {"a": 1}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result-json"}]},
        ],
    )

    wire = captured["payload"]
    assert captured["url"].endswith("/v1/chat/completions")
    assert wire["model"] == "grok-4.5"
    assert wire["messages"][0]["role"] == "system"  # system block list -> system message
    assert wire["tools"][0]["type"] == "function"  # anthropic schema -> function schema
    assert wire["tools"][0]["function"]["name"] == "t1"
    roles = [m["role"] for m in wire["messages"][1:]]
    assert roles == ["user", "assistant", "tool"]  # tool_result became role:"tool"
    assert wire["messages"][3]["tool_call_id"] == "t1"
    assert wire["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

    # response conversion: OpenAI shapes -> Anthropic-shaped objects
    assert resp.stop_reason == "tool_use"
    assert resp.content[0].text == "I will check."
    assert resp.content[1].type == "tool_use"
    assert resp.content[1].name == "t1"
    assert resp.content[1].input == {"a": 2}
    assert resp.usage.input_tokens == 500
    assert resp.usage.output_tokens == 20
    assert resp.usage.cache_read_input_tokens == 123


def test_shim_roundtrips_own_response_blocks(monkeypatch):
    """The loop appends response.content (shim attribute objects) verbatim;
    the converter must accept those, not just plain dicts — this is exactly
    how iteration 2+ requests are built."""
    from trading_agent.llm.openai_shim import OpenAICompatClient, _ToolUseBlock, _TextBlock

    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _FakeHttpxResponse(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    monkeypatch.setattr("trading_agent.llm.openai_shim.httpx.post", fake_post)
    client = OpenAICompatClient(base_url="http://127.0.0.1:11434", api_key="x")

    client.messages_create(
        model="minimax-m3",
        max_tokens=16,
        system="s",
        tools=[{"name": "t1", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        messages=[
            {"role": "user", "content": "go"},
            # attribute-style blocks exactly as the loop appends them
            {"role": "assistant", "content": [_TextBlock("checking"), _ToolUseBlock("call_9", "t1", '{"a": 1}')]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_9", "content": "[]"}]},
        ],
    )

    wire_msgs = captured["payload"]["messages"]
    assistant = next(m for m in wire_msgs if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_9"
    assert assistant["tool_calls"][0]["function"]["name"] == "t1"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    tool_msg = next(m for m in wire_msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_9"


def test_shim_end_turn_and_string_system(monkeypatch):
    from trading_agent.llm.openai_shim import OpenAICompatClient

    def fake_post(url, headers=None, json=None, timeout=None):
        # system passed as string must arrive as a plain system message
        assert json["messages"][0] == {"role": "system", "content": "PLAIN"}
        return _FakeHttpxResponse(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "All done."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    monkeypatch.setattr("trading_agent.llm.openai_shim.httpx.post", fake_post)
    client = OpenAICompatClient(base_url="http://127.0.0.1:11434/v1", api_key="x")
    resp = client.messages_create(
        model="gpt-5.6-luna",
        max_tokens=50,
        system="PLAIN",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.stop_reason == "end_turn"
    assert resp.content[0].text == "All done."
    assert resp.usage.cache_read_input_tokens == 0
    assert resp.usage.cache_creation_input_tokens == 0
