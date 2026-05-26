from __future__ import annotations

import json

from trading_agent.agent.memory import Memory
from trading_agent.agent.tools import build_tool_registry


def test_list_strategies_includes_reference(tmp_path):
    mem = Memory(tmp_path / "mem.sqlite3")
    registry = build_tool_registry(mem)
    result = json.loads(registry["list_strategies"].call({}))
    assert "sma_cross" in result


def test_read_strategy_code_rejects_path_traversal(tmp_path):
    mem = Memory(tmp_path / "mem.sqlite3")
    registry = build_tool_registry(mem)
    for evil in ["../etc/passwd", "../../trading_agent/config", "_secret"]:
        result = json.loads(registry["read_strategy_code"].call({"name": evil}))
        assert "error" in result, f"path traversal not blocked: {evil!r}"


def test_read_strategy_code_returns_source(tmp_path):
    mem = Memory(tmp_path / "mem.sqlite3")
    registry = build_tool_registry(mem)
    result = json.loads(registry["read_strategy_code"].call({"name": "sma_cross"}))
    assert result["name"] == "sma_cross"
    assert "class SmaCross" in result["code"]


def test_search_memory_empty(tmp_path):
    mem = Memory(tmp_path / "mem.sqlite3")
    registry = build_tool_registry(mem)
    result = json.loads(registry["search_memory"].call({}))
    assert result == {"count": 0, "runs": []}


def test_tool_anthropic_schema_shape(tmp_path):
    mem = Memory(tmp_path / "mem.sqlite3")
    registry = build_tool_registry(mem)
    for tool in registry.values():
        schema = tool.anthropic_schema()
        assert set(schema.keys()) == {"name", "description", "input_schema"}
        assert schema["input_schema"]["type"] == "object"
        assert "properties" in schema["input_schema"]
