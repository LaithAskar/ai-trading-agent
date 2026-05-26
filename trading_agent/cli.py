from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import typer
from rich.console import Console
from rich.table import Table

from .backtest.runner import run_backtest as _run_backtest
from .config import PROJECT_ROOT, Config, ensure_dirs

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _parse_params(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--param expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


@app.command()
def backtest(
    strategy: str = typer.Option(..., help="Strategy module name under strategies/"),
    symbol: str = typer.Option(..., help="Ticker (e.g. AAPL)"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    cash: float = typer.Option(100_000.0, help="Starting cash"),
    param: list[str] = typer.Option(  # noqa: B008
        [], "--param", help="Strategy param key=value (repeatable)"
    ),
) -> None:
    """Run a backtest and write results to data/results/."""
    Config.load()
    ensure_dirs()

    params = _parse_params(param)
    console.print(f"[bold]Loading[/bold] {symbol} {start} to {end}")
    console.print(f"[bold]Running[/bold] {strategy} {params}")
    run = _run_backtest(
        strategy_name=strategy,
        symbol=symbol,
        start=start,
        end=end,
        params=params,
        starting_cash=cash,
    )

    table = Table(title=f"{run.strategy} on {run.symbol}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in run.metrics.as_table():
        table.add_row(k, v)
    console.print(table)
    console.print(f"[green]Wrote[/green] {run.artifact_dir}")


@app.command(name="list-strategies")
def list_strategies() -> None:
    """List strategy modules available under strategies/."""
    strat_dir = PROJECT_ROOT / "strategies"
    if not strat_dir.exists():
        console.print("[yellow]No strategies/ directory found[/yellow]")
        raise typer.Exit(0)
    modules = sorted(
        p.stem
        for p in strat_dir.glob("*.py")
        if not p.name.startswith("_") and p.stem != "__init__"
    )
    if not modules:
        console.print("[yellow]No strategies found[/yellow]")
        return
    table = Table(title="Strategies")
    table.add_column("Module")
    for m in modules:
        table.add_row(m)
    console.print(table)


@app.command(name="agent-stats")
def agent_stats() -> None:
    """Aggregate stats across all stored agent sessions.

    Per-tool: call count, avg + p95 latency, error rate.
    Per-session: tokens, cost, finished/stopped status.
    Overall: total cost, total sessions, completion rate.
    """
    from .agent.stats import aggregate
    from .config import AGENT_LOGS_DIR

    Config.load()
    ensure_dirs()

    agg = aggregate(AGENT_LOGS_DIR)
    if agg.total_sessions == 0:
        console.print("[yellow]No sessions found in data/logs/agent_runs/[/yellow]")
        raise typer.Exit(0)

    overall = Table(title="Overall")
    overall.add_column("Metric")
    overall.add_column("Value", justify="right")
    overall.add_row("Sessions", str(agg.total_sessions))
    overall.add_row("Finished", f"{agg.finished_count} / {agg.total_sessions}")
    overall.add_row("Total input tokens",  f"{agg.total_input_tokens:,}")
    overall.add_row("Total output tokens", f"{agg.total_output_tokens:,}")
    overall.add_row("Estimated total cost", f"${agg.total_cost:.4f}")
    console.print(overall)

    tools_table = Table(title="Per-tool")
    tools_table.add_column("Tool")
    tools_table.add_column("Calls", justify="right")
    tools_table.add_column("Errors", justify="right")
    tools_table.add_column("Error %", justify="right")
    tools_table.add_column("Avg ms", justify="right")
    tools_table.add_column("p95 ms", justify="right")
    for tname in sorted(agg.tools, key=lambda n: -agg.tools[n].calls):
        ts = agg.tools[tname]
        tools_table.add_row(
            ts.name,
            str(ts.calls),
            str(ts.errors),
            f"{ts.error_rate_pct:.1f}",
            f"{ts.avg_duration_ms:.0f}",
            str(ts.p95_duration_ms),
        )
    console.print(tools_table)

    sessions_table = Table(title="Recent sessions")
    sessions_table.add_column("Session")
    sessions_table.add_column("Model")
    sessions_table.add_column("Iters", justify="right")
    sessions_table.add_column("Cost", justify="right")
    sessions_table.add_column("Status")
    for s in agg.sessions[-15:]:
        status = "[green]ok[/green]" if s.finished else f"[red]{s.stopped_by or 'incomplete'}[/red]"
        sessions_table.add_row(
            s.session_id,
            s.model,
            str(s.iterations),
            f"${s.cost_dollars:.4f}",
            status,
        )
    console.print(sessions_table)


@app.command()
def replay(
    session_id: str = typer.Argument(..., help="Session ID to replay (or 'latest')"),
) -> None:
    """Re-execute the tool calls from a past session against current code.

    Useful for catching regressions: did sma_cross start returning different
    metrics? Did a refactor change what list_strategies sees? Replay tells you.
    Does NOT call the LLM — only the tool layer.
    """
    from .agent.replay import replay_session
    from .config import AGENT_LOGS_DIR

    Config.load()
    ensure_dirs()

    if session_id == "latest":
        candidates = sorted(AGENT_LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            console.print("[yellow]No sessions found.[/yellow]")
            raise typer.Exit(0)
        path = candidates[-1]
    else:
        path = AGENT_LOGS_DIR / f"{session_id}.json"
        if not path.exists():
            console.print(f"[red]No such session: {path}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Replaying[/bold] {path.name}")
    steps = replay_session(path)

    drifted = [s for s in steps if s.drifted]
    table = Table(title=f"Replay of {path.stem}")
    table.add_column("Iter")
    table.add_column("Tool")
    table.add_column("Status", justify="center")
    for s in steps:
        status = "[red]DRIFTED[/red]" if s.drifted else "[green]OK[/green]"
        table.add_row(str(s.iteration), s.tool_name, status)
    console.print(table)

    if not drifted:
        console.print("[green]No drift detected. All tool outputs match (timestamps redacted).[/green]")
        return

    console.print(f"\n[red]{len(drifted)} step(s) drifted:[/red]\n")
    import json as _json
    for s in drifted:
        console.print(f"[bold]Iteration {s.iteration} — {s.tool_name}[/bold]")
        console.print("  input:", _json.dumps(s.tool_input, default=str))
        console.print(f"  [yellow]original:[/yellow] {_json.dumps(s.original_result, default=str)[:300]}")
        console.print(f"  [cyan]now:     [/cyan] {_json.dumps(s.new_result, default=str)[:300]}\n")


@app.command(name="render-transcript")
def render_transcript(
    session_id: str = typer.Argument(..., help="Session ID, or 'all' to render every transcript, or 'latest' for the most recent"),
) -> None:
    """Render an agent session JSON as a styled HTML report."""
    from .agent.report import render_session_file
    from .config import AGENT_LOGS_DIR

    Config.load()
    ensure_dirs()

    if session_id == "all":
        paths = sorted(AGENT_LOGS_DIR.glob("*.json"))
    elif session_id == "latest":
        paths = sorted(AGENT_LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)[-1:]
    else:
        path = AGENT_LOGS_DIR / f"{session_id}.json"
        if not path.exists():
            console.print(f"[red]No such session: {path}[/red]")
            raise typer.Exit(1)
        paths = [path]

    if not paths:
        console.print("[yellow]No transcripts found.[/yellow]")
        raise typer.Exit(0)

    for p in paths:
        out = render_session_file(p)
        console.print(f"[green]Wrote[/green] {out}")


@app.command()
def agent(
    goal: str = typer.Option(..., help="Natural-language goal for the agent"),
    mode: str = typer.Option("auto", help="auto | interactive"),
    max_iters: int = typer.Option(0, help="Override AGENT_MAX_ITERS (0 = use config)"),
    max_dollars: float = typer.Option(0.0, help="Override AGENT_MAX_SESSION_DOLLARS (0 = use config)"),
    max_tokens: int = typer.Option(0, help="Override AGENT_MAX_SESSION_TOKENS (0 = use config)"),
    model: str = typer.Option("", help="Override AGENT_MODEL"),
) -> None:
    """Run the AI agent on a natural-language goal."""
    from .agent.loop import run_agent

    cfg = Config.load()
    ensure_dirs()

    if not cfg.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY not set.[/red] "
            "Add it to .env or export it in your shell."
        )
        raise typer.Exit(1)
    if mode not in ("auto", "interactive"):
        raise typer.BadParameter("mode must be 'auto' or 'interactive'")

    run_agent(
        goal=goal,
        mode=mode,
        model=model or cfg.agent_model,
        max_iters=max_iters or cfg.agent_max_iters,
        max_tokens_per_call=cfg.agent_max_tokens_per_call,
        max_session_tokens=max_tokens or cfg.agent_max_session_tokens,
        max_session_dollars=max_dollars or cfg.agent_max_session_dollars,
        api_key=cfg.anthropic_api_key,
    )


if __name__ == "__main__":
    app()
