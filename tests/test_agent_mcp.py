"""MCP integration tests.

We don't stand up a real MCP server — we verify:
  1. The token storage round-trips correctly to disk.
  2. The RemoteMCPTool adapter produces a Tool that fits the agent's registry.
  3. Loop merging works: extra_tools land in the registry alongside built-ins.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_agent.agent.mcp_client import (
    FileTokenStorage,
    RemoteMCPTool,
    _safe_id,
)
from trading_agent.agent.tools import Tool


def test_safe_id_strips_dangerous_chars():
    assert _safe_id("https://mcp.example.com/v1") == "https_mcp_example_com_v1"
    assert _safe_id("file://../etc/passwd") == "file_etc_passwd"
    assert _safe_id("") == "mcp_server"


def test_file_token_storage_roundtrip(tmp_path):
    from mcp.shared.auth import OAuthToken, OAuthClientInformationFull

    storage = FileTokenStorage("https://mcp.example.com", base_dir=tmp_path)

    assert asyncio.run(storage.get_tokens()) is None
    assert asyncio.run(storage.get_client_info()) is None

    token = OAuthToken(
        access_token="at-123",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="rt-456",
    )
    asyncio.run(storage.set_tokens(token))

    fetched = asyncio.run(storage.get_tokens())
    assert fetched is not None
    assert fetched.access_token == "at-123"
    assert fetched.refresh_token == "rt-456"


def test_remote_mcp_tool_wraps_as_local_tool():
    rt = RemoteMCPTool(
        name="fetch_filing",
        description="Fetch the latest SEC filing for a ticker",
        input_schema={
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
        server_url="https://mcp.example.com",
    )

    local = rt.as_local_tool("https://mcp.example.com")
    assert isinstance(local, Tool)
    assert local.name == "mcp_fetch_filing"
    assert "[MCP:" in local.description
    schema = local.anthropic_schema()
    assert schema["input_schema"]["properties"] == {"ticker": {"type": "string"}}


def test_loop_merges_extra_tools_into_registry(tmp_path, monkeypatch):
    """An extra Tool passed in via extra_tools must appear in the registry the agent sees."""
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    custom_tool = Tool(
        name="mcp_fake_thing",
        description="A fake remote tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=lambda args: {"ok": True},
    )

    from tests.test_agent_loop import FakeAnthropic, FakeResponse, FakeTextBlock

    fake = FakeAnthropic([
        FakeResponse(content=[FakeTextBlock("Nothing to do.")], stop_reason="end_turn"),
    ])

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        run_agent(
            goal="check tools",
            mode="auto",
            model="fake",
            max_iters=2,
            api_key="fake-key",
            extra_tools=[custom_tool],
        )

    sent_schemas = fake.messages.calls[0]["tools"]
    sent_names = {s["name"] for s in sent_schemas}
    assert "mcp_fake_thing" in sent_names
    # Built-ins still there too
    assert "list_strategies" in sent_names
    assert "run_backtest" in sent_names


def test_loop_extra_tool_collision_warns_and_overrides(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    # Collide with an existing built-in name on purpose.
    collider = Tool(
        name="list_strategies",
        description="Remote override",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=lambda args: {"remote": True},
    )

    from tests.test_agent_loop import FakeAnthropic, FakeResponse, FakeTextBlock

    fake = FakeAnthropic([
        FakeResponse(content=[FakeTextBlock("Done.")], stop_reason="end_turn"),
    ])

    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        run_agent(
            goal="collision test",
            mode="auto",
            model="fake",
            max_iters=2,
            api_key="fake-key",
            extra_tools=[collider],
        )

    sent_schemas = fake.messages.calls[0]["tools"]
    list_strat_schema = next(s for s in sent_schemas if s["name"] == "list_strategies")
    assert "Remote override" in list_strat_schema["description"]
