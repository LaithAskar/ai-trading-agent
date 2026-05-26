from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolStats:
    name: str
    calls: int = 0
    errors: int = 0
    total_duration_ms: int = 0
    durations: list[int] = field(default_factory=list)

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.calls if self.calls else 0.0

    @property
    def p95_duration_ms(self) -> int:
        if not self.durations:
            return 0
        s = sorted(self.durations)
        idx = max(0, int(0.95 * (len(s) - 1)))
        return s[idx]

    @property
    def error_rate_pct(self) -> float:
        return (self.errors / self.calls * 100) if self.calls else 0.0


@dataclass
class SessionStats:
    session_id: str
    iterations: int
    input_tokens: int
    output_tokens: int
    cost_dollars: float
    finished: bool
    stopped_by: str | None
    model: str


@dataclass
class AggregateStats:
    sessions: list[SessionStats]
    tools: dict[str, ToolStats]

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_dollars for s in self.sessions)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.sessions)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.sessions)

    @property
    def finished_count(self) -> int:
        return sum(1 for s in self.sessions if s.finished)


def aggregate(logs_dir: Path) -> AggregateStats:
    sessions: list[SessionStats] = []
    tools: dict[str, ToolStats] = defaultdict(lambda: ToolStats(name=""))

    for path in sorted(logs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        sessions.append(
            SessionStats(
                session_id=data.get("session_id", path.stem),
                iterations=len(data.get("transcript", [])),
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                cost_dollars=data.get("cost_dollars", 0.0),
                finished=data.get("finished", False),
                stopped_by=data.get("stopped_by"),
                model=data.get("model", "unknown"),
            )
        )

        for entry in data.get("transcript", []):
            tname = entry.get("tool_name")
            if not tname:
                continue
            ts = tools[tname]
            if not ts.name:
                ts.name = tname
            ts.calls += 1
            if entry.get("is_error"):
                ts.errors += 1
            d = entry.get("tool_duration_ms")
            if isinstance(d, int):
                ts.total_duration_ms += d
                ts.durations.append(d)

    return AggregateStats(sessions=sessions, tools=dict(tools))
