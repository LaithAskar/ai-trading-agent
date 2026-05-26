from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import MEMORY_DB
from .memory import Memory
from .tools import build_tool_registry


# Fields in tool results that change between runs (timestamps, random IDs) —
# we redact these before comparing so the diff highlights real semantic drift.
_REDACTED_KEYS = {"run_id", "created_at", "timestamp", "started_at", "artifact_dir"}


@dataclass
class ReplayStep:
    iteration: int
    tool_name: str
    tool_input: dict
    original_result: object
    new_result: object
    drifted: bool


def _redact(obj):
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if k in _REDACTED_KEYS else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _safe_json(s: str | None):
    if s is None:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def replay_session(session_path: Path) -> list[ReplayStep]:
    """Re-execute every tool call in a stored session against the current code.

    Does NOT call the LLM. The point is to test whether tool implementations
    drifted relative to past behavior — useful for catching regressions or
    confirming a refactor preserves semantics.
    """
    data = json.loads(session_path.read_text(encoding="utf-8"))
    memory = Memory(MEMORY_DB)
    registry = build_tool_registry(memory)

    steps: list[ReplayStep] = []
    for entry in data.get("transcript", []):
        tool_name = entry.get("tool_name")
        if not tool_name or entry.get("is_final"):
            continue
        tool_input = entry.get("tool_input") or {}
        original_result = _safe_json(entry.get("tool_result"))

        tool = registry.get(tool_name)
        if tool is None:
            new_result = {"error": f"tool {tool_name!r} no longer exists"}
        else:
            new_result = _safe_json(tool.call(tool_input))

        drifted = _redact(original_result) != _redact(new_result)
        steps.append(
            ReplayStep(
                iteration=entry.get("iteration", 0),
                tool_name=tool_name,
                tool_input=tool_input,
                original_result=original_result,
                new_result=new_result,
                drifted=drifted,
            )
        )
    return steps
