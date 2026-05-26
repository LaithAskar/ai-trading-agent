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
    table.add_column("Strategy", justify="right")
    table.add_column("Buy & Hold", justify="right")
    bench = run.benchmark
    rows = [
        ("Total return",        f"{run.metrics.total_return_pct:.2f}%",  f"{bench.total_return_pct:.2f}%"),
        ("CAGR",                f"{run.metrics.cagr_pct:.2f}%",          f"{bench.cagr_pct:.2f}%"),
        ("Sharpe (annualized)", f"{run.metrics.sharpe:.2f}",             f"{bench.sharpe:.2f}"),
        ("Max drawdown",        f"{run.metrics.max_drawdown_pct:.2f}%",  f"{bench.max_drawdown_pct:.2f}%"),
        ("Ending equity",       f"${run.metrics.ending_equity:,.2f}",    f"${bench.end_equity:,.2f}"),
        ("Fills",               str(run.metrics.num_fills),              "1"),
        ("Round trips",         str(run.metrics.num_round_trips),        "0"),
        ("Win rate",            f"{run.metrics.win_rate_pct:.2f}%",      "—"),
    ]
    for r in rows:
        table.add_row(*r)
    table.add_row("Sharpe t-stat", f"{run.sharpe_t_stat:.2f}", "")
    table.add_row("Sharpe p-value", f"{run.sharpe_p_value:.4f}", "")
    console.print(table)
    if run.sharpe_p_value >= 0.10:
        console.print(
            f"[yellow]Sharpe is not statistically distinguishable from zero "
            f"(p={run.sharpe_p_value:.3f}). Don't over-interpret.[/yellow]"
        )
    console.print(f"[green]Wrote[/green] {run.artifact_dir}")


@app.command(name="paper-status")
def paper_status() -> None:
    """Show current Alpaca paper account, positions, and open orders."""
    from .broker.alpaca import AlpacaPaperBroker

    cfg = Config.load()
    ensure_dirs()
    if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
        console.print("[red]ALPACA_API_KEY / ALPACA_API_SECRET not set in .env[/red]")
        raise typer.Exit(1)

    broker = AlpacaPaperBroker(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    acct = broker.account()

    table = Table(title="Alpaca paper account")
    table.add_column("Field"); table.add_column("Value", justify="right")
    table.add_row("Cash",          f"${acct.cash:,.2f}")
    table.add_row("Portfolio val", f"${acct.portfolio_value:,.2f}")
    table.add_row("Buying power",  f"${acct.buying_power:,.2f}")
    table.add_row("Paper?",        "yes" if acct.is_paper else "NO (refusing)")
    console.print(table)

    positions = broker.positions()
    if positions:
        ptable = Table(title="Positions")
        ptable.add_column("Symbol"); ptable.add_column("Qty", justify="right")
        ptable.add_column("Avg entry", justify="right"); ptable.add_column("Mkt value", justify="right")
        ptable.add_column("Unrealized P/L", justify="right")
        for p in positions:
            ptable.add_row(p.symbol, f"{p.quantity:g}", f"${p.avg_entry_price:.2f}",
                           f"${p.market_value:,.2f}", f"${p.unrealized_pl:,.2f}")
        console.print(ptable)
    else:
        console.print("[dim]No open positions[/dim]")

    open_orders = broker.open_orders()
    if open_orders:
        otable = Table(title="Open orders")
        otable.add_column("Order"); otable.add_column("Symbol"); otable.add_column("Side")
        otable.add_column("Qty", justify="right"); otable.add_column("Status")
        for o in open_orders:
            otable.add_row(o.order_id[:8] + "...", o.symbol, o.side, f"{o.quantity:g}", o.status)
        console.print(otable)


@app.command(name="paper-trade")
def paper_trade(
    strategy: str = typer.Option(..., help="Strategy module name"),
    symbol: str = typer.Option(..., help="Ticker"),
    lookback_days: int = typer.Option(365, help="History days needed to replay strategy state"),
    param: list[str] = typer.Option([], "--param"),  # noqa: B008
    dry_run: bool = typer.Option(True, help="If true, show proposed orders without submitting"),
) -> None:
    """Run one tick of the strategy against current market data and (optionally)
    submit any proposed orders to Alpaca paper.

    Default is dry-run. Pass --no-dry-run to actually submit.
    """
    from .broker.alpaca import AlpacaPaperBroker
    from .broker.paper_runner import paper_tick

    cfg = Config.load()
    ensure_dirs()

    params = _parse_params(param)

    broker = None
    if not dry_run:
        if not cfg.alpaca_api_key or not cfg.alpaca_api_secret:
            console.print("[red]ALPACA_API_KEY / ALPACA_API_SECRET required when --no-dry-run.[/red]")
            raise typer.Exit(1)
        broker = AlpacaPaperBroker(cfg.alpaca_api_key, cfg.alpaca_api_secret)

    result = paper_tick(
        strategy_name=strategy,
        symbol=symbol,
        params=params,
        lookback_days=lookback_days,
        broker=broker,
        dry_run=dry_run,
    )

    console.print(f"[bold]Strategy:[/bold] {result.strategy}  [bold]Symbol:[/bold] {result.symbol}  [bold]Bars replayed:[/bold] {result.bars_seen}")
    if result.proposed_orders:
        otable = Table(title="Proposed orders")
        otable.add_column("Symbol"); otable.add_column("Side"); otable.add_column("Qty", justify="right")
        for o in result.proposed_orders:
            otable.add_row(o.symbol, o.side.value, f"{o.quantity:g}")
        console.print(otable)
    else:
        console.print("[dim]No orders proposed this tick.[/dim]")

    if result.dry_run:
        console.print("[yellow]Dry run — nothing submitted. Re-run with --no-dry-run to send to Alpaca paper.[/yellow]")
    elif result.submitted:
        stable = Table(title="Submitted")
        stable.add_column("Order"); stable.add_column("Symbol"); stable.add_column("Status")
        for o in result.submitted:
            stable.add_row(o.order_id[:8] + "...", o.symbol, o.status)
        console.print(stable)
    if result.skipped_reason:
        console.print(f"[yellow]{result.skipped_reason}[/yellow]")


def _expand_grid(grid_args: list[str]) -> list[dict]:
    """Expand `--grid key=v1,v2,v3` flags into the cartesian product of param dicts.

    Each --grid is one parameter and a comma-separated value list. Values are
    coerced int -> float -> str in that order (same rules as --param).
    """
    if not grid_args:
        return [{}]
    axes: list[tuple[str, list]] = []
    for spec in grid_args:
        if "=" not in spec:
            raise typer.BadParameter(f"--grid expects key=v1,v2,..., got '{spec}'")
        key, raw_values = spec.split("=", 1)
        vals: list = []
        for v in raw_values.split(","):
            v = v.strip()
            try:
                vals.append(int(v))
            except ValueError:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(v)
        axes.append((key, vals))

    combos: list[dict] = [{}]
    for key, vals in axes:
        combos = [{**combo, key: v} for combo in combos for v in vals]
    return combos


@app.command(name="param-sweep")
def param_sweep(
    strategy: str = typer.Option(..., help="Strategy module name"),
    symbol: str = typer.Option(..., help="Ticker"),
    start: str = typer.Option(..., help="Backtest start YYYY-MM-DD"),
    end: str = typer.Option(..., help="Backtest end YYYY-MM-DD"),
    grid: list[str] = typer.Option(  # noqa: B008
        [], "--grid",
        help="Repeatable: 'param=val1,val2,val3'. Cartesian product is run.",
    ),
    walk_forward_flag: bool = typer.Option(
        False, "--walk-forward",
        help="Run each combo through walk-forward instead of one window. Slower, more honest.",
    ),
    train_years: float = typer.Option(2.0),
    test_years: float = typer.Option(1.0),
    stride_years: float = typer.Option(1.0),
) -> None:
    """Grid-search a strategy's parameters. Default: single-window. With
    --walk-forward, each combo runs through rolling train/test splits and
    we report OOS-CAGR mean+stdev so you can rank by *robustness* not just
    in-sample fit.
    """
    from .backtest.rigor import walk_forward_splits
    from .backtest.runner import run_backtest as _rb

    Config.load()
    ensure_dirs()

    combos = _expand_grid(grid)
    console.print(f"[dim]Running {len(combos)} param combo(s)[/dim]")

    if walk_forward_flag:
        splits = walk_forward_splits(start, end, train_years, test_years, stride_years)
        if not splits:
            console.print(f"[red]No splits between {start} and {end}[/red]")
            raise typer.Exit(1)

        table = Table(title=f"Walk-forward sweep: {strategy} on {symbol}")
        table.add_column("Params")
        table.add_column("OOS CAGR mean", justify="right")
        table.add_column("OOS CAGR stdev", justify="right")
        table.add_column("OOS Sharpe mean", justify="right")
        table.add_column("Splits", justify="right")

        results = []
        for combo in combos:
            cagrs, sharpes = [], []
            for s in splits:
                try:
                    r = _rb(
                        strategy_name=strategy, symbol=symbol,
                        start=s.test_start, end=s.test_end,
                        params=combo, write_artifacts=False,
                    )
                    cagrs.append(r.metrics.cagr_pct)
                    sharpes.append(r.metrics.sharpe)
                except Exception:
                    continue
            if not cagrs:
                continue
            import statistics
            cagr_mean = statistics.mean(cagrs)
            cagr_std = statistics.stdev(cagrs) if len(cagrs) > 1 else 0.0
            sharpe_mean = statistics.mean(sharpes)
            results.append((combo, cagr_mean, cagr_std, sharpe_mean, len(cagrs)))

        results.sort(key=lambda r: -r[3])  # by Sharpe mean
        for combo, c_m, c_s, s_m, n in results:
            table.add_row(
                ", ".join(f"{k}={v}" for k, v in combo.items()) or "(none)",
                f"{c_m:.2f}%", f"{c_s:.2f}%", f"{s_m:.2f}", str(n),
            )
        console.print(table)
        return

    # Single-window mode
    table = Table(title=f"Param sweep: {strategy} on {symbol}  ({start} → {end})")
    table.add_column("Params")
    table.add_column("CAGR", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("B&H CAGR", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Round trips", justify="right")

    results = []
    for combo in combos:
        try:
            r = _rb(
                strategy_name=strategy, symbol=symbol,
                start=start, end=end, params=combo, write_artifacts=False,
            )
            results.append((combo, r))
        except Exception as e:
            console.print(f"[red]combo {combo} failed: {e}[/red]")

    results.sort(key=lambda x: -x[1].metrics.sharpe)
    for combo, r in results:
        table.add_row(
            ", ".join(f"{k}={v}" for k, v in combo.items()) or "(none)",
            f"{r.metrics.cagr_pct:.2f}%",
            f"{r.metrics.sharpe:.2f}",
            f"{r.metrics.max_drawdown_pct:.2f}%",
            f"{r.benchmark.cagr_pct:.2f}%",
            f"{r.sharpe_p_value:.3f}",
            str(r.metrics.num_round_trips),
        )
    console.print(table)


@app.command(name="walk-forward")
def walk_forward(
    strategy: str = typer.Option(..., help="Strategy module name"),
    symbol: str = typer.Option(..., help="Ticker"),
    start: str = typer.Option(..., help="Overall start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Overall end date YYYY-MM-DD"),
    train_years: float = typer.Option(3.0, help="Training window years"),
    test_years: float = typer.Option(1.0, help="Test (out-of-sample) window years"),
    stride_years: float = typer.Option(1.0, help="Years to slide the window each split"),
    param: list[str] = typer.Option([], "--param"),  # noqa: B008
) -> None:
    """Walk-forward validation. Runs the strategy on rolling train/test windows.

    Reports the OUT-OF-SAMPLE performance per split. Most strategies that
    look great on a single backtest fall apart here — that's the point.
    """
    from .backtest.rigor import walk_forward_splits
    from .backtest.runner import run_backtest as _rb

    Config.load()
    ensure_dirs()

    params = _parse_params(param)
    splits = walk_forward_splits(start, end, train_years, test_years, stride_years)
    if not splits:
        console.print(
            f"[red]No valid splits between {start} and {end} with "
            f"train={train_years}y, test={test_years}y[/red]"
        )
        raise typer.Exit(1)

    table = Table(title=f"Walk-forward: {strategy} on {symbol}")
    table.add_column("Split")
    table.add_column("Train")
    table.add_column("Test (OOS)")
    table.add_column("Strat CAGR", justify="right")
    table.add_column("B&H CAGR", justify="right")
    table.add_column("Strat Sharpe", justify="right")
    table.add_column("p-value", justify="right")

    oos_returns = []
    for i, split in enumerate(splits, start=1):
        try:
            run = _rb(
                strategy_name=strategy,
                symbol=symbol,
                start=split.test_start,
                end=split.test_end,
                params=params,
                write_artifacts=False,
            )
        except Exception as e:
            console.print(f"[red]Split {i} failed: {e}[/red]")
            continue
        oos_returns.append(run.metrics.cagr_pct)
        table.add_row(
            f"#{i}",
            f"{split.train_start} → {split.train_end}",
            f"{split.test_start} → {split.test_end}",
            f"{run.metrics.cagr_pct:.2f}%",
            f"{run.benchmark.cagr_pct:.2f}%",
            f"{run.metrics.sharpe:.2f}",
            f"{run.sharpe_p_value:.3f}",
        )
    console.print(table)

    if oos_returns:
        import statistics
        mean_oos = statistics.mean(oos_returns)
        stdev_oos = statistics.stdev(oos_returns) if len(oos_returns) > 1 else 0.0
        console.print(
            f"[bold]Out-of-sample CAGR: mean={mean_oos:.2f}%, stdev={stdev_oos:.2f}%, "
            f"splits={len(oos_returns)}[/bold]"
        )


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


@app.command(name="mcp-connect")
def mcp_connect(
    url: str = typer.Argument(..., help="Remote MCP server URL (e.g. https://mcp.example.com)"),
) -> None:
    """Authenticate with a remote MCP server via OAuth and list its tools.

    Opens your browser to the auth URL. Once you approve, the tokens are
    stored under data/mcp/ and subsequent agent runs can use --mcp-server <url>
    to load that server's tools alongside the built-in ones.
    """
    from .agent.mcp_client import discover_tools, is_authenticated

    Config.load()
    ensure_dirs()

    already = is_authenticated(url)
    if already:
        console.print(f"[dim]Existing tokens found for {url}; refreshing tool list.[/dim]")

    console.print(f"[bold]Connecting to[/bold] {url}")
    try:
        tools = discover_tools(url)
    except Exception as e:
        console.print(f"[red]Connection failed: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Tools exposed by {url}")
    table.add_column("Name")
    table.add_column("Description")
    for t in tools:
        desc = (t.description or "").replace("\n", " ")[:80]
        table.add_row(t.name, desc)
    console.print(table)
    console.print(f"[green]Done.[/green] {len(tools)} tool(s) available. Run the agent with --mcp-server {url}")


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
    mcp_server: list[str] = typer.Option(  # noqa: B008
        [], "--mcp-server", help="URL of a remote MCP server whose tools the agent should also use (repeatable). Auth via `mcp-connect` first."
    ),
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

    extra_tools = []
    if mcp_server:
        from .agent.mcp_client import discover_tools, is_authenticated
        for url in mcp_server:
            if not is_authenticated(url):
                console.print(
                    f"[red]Not authenticated to {url}.[/red] "
                    f"Run: python -m trading_agent mcp-connect {url}"
                )
                raise typer.Exit(1)
            console.print(f"[dim]Loading tools from {url}...[/dim]")
            extra_tools.extend(discover_tools(url))
        console.print(f"[dim]+{len(extra_tools)} MCP tool(s) loaded[/dim]")

    run_agent(
        goal=goal,
        mode=mode,
        model=model or cfg.agent_model,
        max_iters=max_iters or cfg.agent_max_iters,
        max_tokens_per_call=cfg.agent_max_tokens_per_call,
        max_session_tokens=max_tokens or cfg.agent_max_session_tokens,
        max_session_dollars=max_dollars or cfg.agent_max_session_dollars,
        api_key=cfg.anthropic_api_key,
        extra_tools=extra_tools,
    )


if __name__ == "__main__":
    app()
