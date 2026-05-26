from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from ..config import AGENT_LOGS_DIR, MEMORY_DB
from .memory import Memory
from .pricing import estimate_cost
from .prompts import SYSTEM_PROMPT
from .tools import build_tool_registry

console = Console()


@dataclass
class TranscriptEntry:
    iteration: int
    thought: str
    tool_name: str | None
    tool_input: dict | None
    tool_result: str | None
    is_final: bool
    tool_duration_ms: int | None = None
    is_error: bool = False


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
    cost_dollars: float = 0.0
    finished: bool = False
    final_summary: str | None = None
    stopped_by: str | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "mode": self.mode,
            "model": self.model,
            "started_at": self.started_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_dollars": round(self.cost_dollars, 6),
            "finished": self.finished,
            "stopped_by": self.stopped_by,
            "final_summary": self.final_summary,
            "transcript": [
                {
                    "iteration": e.iteration,
                    "thought": e.thought,
                    "tool_name": e.tool_name,
                    "tool_input": e.tool_input,
                    "tool_result": e.tool_result,
                    "tool_duration_ms": e.tool_duration_ms,
                    "is_error": e.is_error,
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


def _render_observation(result_json: str, is_error: bool, duration_ms: int | None = None) -> None:
    color = "red" if is_error else "green"
    base_label = "Observation (error)" if is_error else "Observation"
    label = f"{base_label}  ({duration_ms} ms)" if duration_ms is not None else base_label
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


def _render_cap_hit(reason: str) -> None:
    console.print(
        Panel(
            reason,
            title="[bold red]Session Cap Hit — stopping[/bold red]",
            border_style="red",
        )
    )


def run_agent(
    *,
    goal: str,
    mode: str = "auto",
    model: str = "claude-sonnet-4-6",
    max_iters: int = 20,
    max_tokens_per_call: int = 4096,
    max_session_tokens: int = 200_000,
    max_session_dollars: float = 1.00,
    api_key: str | None = None,
    extra_tools: list | None = None,
    on_iteration=None,
) -> AgentSession:
    """Run the ReAct agent on a single goal.

    Safety caps (hard kill the loop if exceeded):
      - max_iters:           bound on iterations
      - max_session_tokens:  bound on cumulative input + output tokens
      - max_session_dollars: bound on cumulative estimated $ cost
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
    for t in (extra_tools or []):
        if t.name in registry:
            console.print(f"[yellow]Tool name collision on {t.name!r}; remote version wins.[/yellow]")
        registry[t.name] = t
    tool_schemas = [t.anthropic_schema() for t in registry.values()]

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    console.rule(f"[bold]Agent session {session.session_id}[/bold]  ({mode}, {model})")
    console.print(Panel(goal, title="[bold]Goal[/bold]", border_style="white"))
    console.print(
        f"[dim]caps: {max_iters} iters  |  {max_session_tokens:,} tokens  |  "
        f"${max_session_dollars:.2f}[/dim]"
    )

    messages: list[dict] = [{"role": "user", "content": goal}]
    cap_hit_reason: str | None = None

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
            cap_hit_reason = f"API error: {e}"
            break

        session.input_tokens += response.usage.input_tokens
        session.output_tokens += response.usage.output_tokens
        cost = estimate_cost(model, session.input_tokens, session.output_tokens)
        session.cost_dollars = cost.total_dollars

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
            entry = TranscriptEntry(
                iteration=iteration,
                thought=thought_text,
                tool_name=None,
                tool_input=None,
                tool_result=None,
                is_final=True,
            )
            session.transcript.append(entry)
            if on_iteration:
                on_iteration(entry)
            break

        if not tool_uses:
            console.print("[yellow]No tool call and no end_turn — terminating to avoid loop[/yellow]")
            session.final_summary = thought_text or "(no output)"
            cap_hit_reason = "stalled (no tool call, no end_turn)"
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
                    _render_observation(result_str, is_error=True)
                    tool_results_for_user.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result_str,
                            "is_error": True,
                        }
                    )
                    entry = TranscriptEntry(
                        iteration=iteration,
                        thought=thought_text,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_result=result_str,
                        tool_duration_ms=0,
                        is_error=True,
                        is_final=False,
                    )
                    session.transcript.append(entry)
                    if on_iteration:
                        on_iteration(entry)
                    continue

            tool = registry.get(tool_name)
            t0 = time.perf_counter()
            if tool is None:
                result_str = json.dumps({"error": f"unknown tool: {tool_name}"})
                is_error = True
            else:
                result_str = tool.call(tool_input)
                is_error = '"error"' in result_str[:50]
            duration_ms = int((time.perf_counter() - t0) * 1000)

            _render_observation(result_str, is_error=is_error, duration_ms=duration_ms)

            tool_results_for_user.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str,
                    "is_error": is_error,
                }
            )
            entry = TranscriptEntry(
                iteration=iteration,
                thought=thought_text,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=result_str,
                tool_duration_ms=duration_ms,
                is_error=is_error,
                is_final=False,
            )
            session.transcript.append(entry)
            if on_iteration:
                on_iteration(entry)

        messages.append({"role": "user", "content": tool_results_for_user})

        # Hard-kill the loop if cumulative caps exceeded.
        total_tokens = session.input_tokens + session.output_tokens
        if total_tokens > max_session_tokens:
            cap_hit_reason = (
                f"token cap exceeded: {total_tokens:,} > {max_session_tokens:,}"
            )
            break
        if session.cost_dollars > max_session_dollars:
            cap_hit_reason = (
                f"dollar cap exceeded: ${session.cost_dollars:.4f} > ${max_session_dollars:.2f}"
            )
            break

    else:
        cap_hit_reason = f"max_iters={max_iters} reached"

    if cap_hit_reason and not session.finished:
        _render_cap_hit(cap_hit_reason)
        session.stopped_by = cap_hit_reason
        session.final_summary = f"(stopped: {cap_hit_reason})"

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
        f"  est. cost:     ${session.cost_dollars:.4f}\n"
        f"  finished:      {session.finished}\n"
        f"  stopped_by:    {session.stopped_by or '(completed normally)'}\n"
        f"  log:           data/logs/agent_runs/{session.session_id}.json"
    )
