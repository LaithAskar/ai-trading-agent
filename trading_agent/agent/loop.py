from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from ..config import AGENT_LOGS_DIR, MEMORY_DB
from .memory import Memory
from .prompts import SYSTEM_PROMPT
from .tools import Tool, build_tool_registry

console = Console()


@dataclass
class TranscriptEntry:
    iteration: int
    thought: str
    tool_name: str | None
    tool_input: dict | None
    tool_result: str | None
    is_final: bool


@dataclass
class AgentSession:
    session_id: str
    goal: str
    mode: str
    model: str
    started_at: str
    transcript: list[TranscriptEntry] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    finished: bool = False
    final_summary: str | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "mode": self.mode,
            "model": self.model,
            "started_at": self.started_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finished": self.finished,
            "final_summary": self.final_summary,
            "transcript": [
                {
                    "iteration": e.iteration,
                    "thought": e.thought,
                    "tool_name": e.tool_name,
                    "tool_input": e.tool_input,
                    "tool_result": e.tool_result,
                    "is_final": e.is_final,
                }
                for e in self.transcript
            ],
        }


def _render_thought(text: str) -> None:
    if not text.strip():
        return
    console.print(Panel(text.strip(), title="[bold blue]Thought[/bold blue]", border_style="blue"))


def _render_action(tool_name: str, tool_input: dict) -> None:
    pretty = json.dumps(tool_input, indent=2, default=str)
    console.print(
        Panel(
            Syntax(pretty, "json", theme="ansi_dark", word_wrap=True),
            title=f"[bold cyan]Action[/bold cyan] -> {tool_name}",
            border_style="cyan",
        )
    )


def _render_observation(result_json: str, is_error: bool) -> None:
    color = "red" if is_error else "green"
    label = "Observation (error)" if is_error else "Observation"
    try:
        pretty = json.dumps(json.loads(result_json), indent=2, default=str)
    except json.JSONDecodeError:
        pretty = result_json
    if len(pretty) > 4000:
        pretty = pretty[:4000] + "\n... (truncated)"
    console.print(
        Panel(
            Syntax(pretty, "json", theme="ansi_dark", word_wrap=True),
            title=f"[bold {color}]{label}[/bold {color}]",
            border_style=color,
        )
    )


def _render_final(text: str) -> None:
    console.print(Panel(text.strip(), title="[bold green]Final Answer[/bold green]", border_style="green"))


def run_agent(
    *,
    goal: str,
    mode: str = "auto",
    model: str = "claude-sonnet-4-6",
    max_iters: int = 20,
    max_tokens_per_call: int = 4096,
    api_key: str | None = None,
) -> AgentSession:
    """Run the ReAct agent on a single goal.

    mode:
      - 'auto': fully autonomous; tools execute without confirmation.
      - 'interactive': prompts the user before each tool execution. Matches the
        course Module 3 demo of fully-automated vs semi-automated agent runs.
    """
    if mode not in ("auto", "interactive"):
        raise ValueError(f"mode must be 'auto' or 'interactive', got {mode!r}")

    session = AgentSession(
        session_id=uuid.uuid4().hex[:12],
        goal=goal,
        mode=mode,
        model=model,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    memory = Memory(MEMORY_DB)
    registry = build_tool_registry(memory)
    tool_schemas = [t.anthropic_schema() for t in registry.values()]

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    console.rule(f"[bold]Agent session {session.session_id}[/bold]  ({mode}, {model})")
    console.print(Panel(goal, title="[bold]Goal[/bold]", border_style="white"))

    messages: list[dict] = [{"role": "user", "content": goal}]

    for iteration in range(1, max_iters + 1):
        console.rule(f"[dim]iteration {iteration}/{max_iters}[/dim]")

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens_per_call,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=tool_schemas,
                messages=messages,
            )
        except anthropic.APIError as e:
            console.print(f"[red]API error: {e}[/red]")
            break

        session.input_tokens += response.usage.input_tokens
        session.output_tokens += response.usage.output_tokens

        thought_parts: list[str] = []
        tool_uses: list[Any] = []
        for block in response.content:
            if block.type == "text":
                thought_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        thought_text = "\n".join(thought_parts).strip()
        _render_thought(thought_text)

        if response.stop_reason == "end_turn" and not tool_uses:
            _render_final(thought_text)
            session.final_summary = thought_text
            session.finished = True
            session.transcript.append(
                TranscriptEntry(
                    iteration=iteration,
                    thought=thought_text,
                    tool_name=None,
                    tool_input=None,
                    tool_result=None,
                    is_final=True,
                )
            )
            break

        if not tool_uses:
            console.print("[yellow]No tool call and no end_turn — terminating to avoid loop[/yellow]")
            session.final_summary = thought_text or "(no output)"
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results_for_user: list[dict] = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            _render_action(tool_name, tool_input)

            if mode == "interactive":
                approved = Confirm.ask(
                    f"[yellow]Run {tool_name}?[/yellow]", default=True
                )
                if not approved:
                    result_str = json.dumps({"error": "User rejected this action"})
                    is_error = True
                    _render_observation(result_str, is_error=True)
                    tool_results_for_user.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result_str,
                            "is_error": True,
                        }
                    )
                    session.transcript.append(
                        TranscriptEntry(
                            iteration=iteration,
                            thought=thought_text,
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_result=result_str,
                            is_final=False,
                        )
                    )
                    continue

            tool = registry.get(tool_name)
            if tool is None:
                result_str = json.dumps({"error": f"unknown tool: {tool_name}"})
                is_error = True
            else:
                result_str = tool.call(tool_input)
                is_error = '"error"' in result_str[:50]

            _render_observation(result_str, is_error=is_error)

            tool_results_for_user.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str,
                    "is_error": is_error,
                }
            )
            session.transcript.append(
                TranscriptEntry(
                    iteration=iteration,
                    thought=thought_text,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_result=result_str,
                    is_final=False,
                )
            )

        messages.append({"role": "user", "content": tool_results_for_user})

    else:
        console.print(f"[yellow]Hit max_iters={max_iters} without final answer.[/yellow]")
        session.final_summary = "(stopped: max iterations reached)"

    _persist_session(session, memory)
    _print_summary(session)
    return session


def _persist_session(session: AgentSession, memory: Memory) -> None:
    log_path = AGENT_LOGS_DIR / f"{session.session_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(session.to_dict(), indent=2, default=str))
    memory.record_session(
        session_id=session.session_id,
        goal=session.goal,
        mode=session.mode,
        model=session.model,
        iterations=len(session.transcript),
        input_tokens=session.input_tokens,
        output_tokens=session.output_tokens,
        finished=session.finished,
        final_summary=session.final_summary,
    )


def _print_summary(session: AgentSession) -> None:
    console.rule("[bold]session summary[/bold]")
    console.print(
        f"  iterations:    {len(session.transcript)}\n"
        f"  input tokens:  {session.input_tokens:,}\n"
        f"  output tokens: {session.output_tokens:,}\n"
        f"  finished:      {session.finished}\n"
        f"  log:           data/logs/agent_runs/{session.session_id}.json"
    )
