from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..backtest.runner import run_backtest
from ..config import MEMORY_DB, PROJECT_ROOT, Config
from .memory import Memory


@dataclass
class Tool:
    """A callable tool exposed to the agent.

    `schema` is the Anthropic tool-use JSON Schema (input_schema).
    `fn` takes a dict of arguments and returns a JSON-serializable result.
    """

    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], Any]

    def anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def call(self, args: dict) -> str:
        try:
            result = self.fn(args)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        return json.dumps(result, default=str)


# ----- individual tool implementations -----

def _list_strategies(_: dict) -> list[str]:
    strat_dir = PROJECT_ROOT / "strategies"
    if not strat_dir.exists():
        return []
    return sorted(
        p.stem
        for p in strat_dir.glob("*.py")
        if not p.name.startswith("_") and p.stem != "__init__"
    )


def _read_strategy_code(args: dict) -> dict:
    name = args["name"]
    if "/" in name or ".." in name or name.startswith("_"):
        raise ValueError(f"Invalid strategy name: {name!r}")
    path = PROJECT_ROOT / "strategies" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"strategies/{name}.py not found")
    code = path.read_text()
    return {"name": name, "path": str(path.relative_to(PROJECT_ROOT)), "code": code}


def _run_backtest_tool(args: dict) -> dict:
    cfg = Config.load()
    if cfg.live_trading:
        raise PermissionError(
            "live_trading is enabled in config but the agent only operates in "
            "backtest/paper mode — refusing to proceed"
        )

    memory: Memory = args["__memory"]
    rb_kwargs = dict(
        strategy_name=args["strategy"],
        symbol=args["symbol"],
        start=args["start"],
        end=args["end"],
        params=args.get("params") or {},
        starting_cash=float(args.get("starting_cash", 100_000.0)),
    )
    if "slippage_bps" in args:
        rb_kwargs["slippage_bps"] = float(args["slippage_bps"])
    if "commission_per_trade" in args:
        rb_kwargs["commission_per_trade"] = float(args["commission_per_trade"])
    run = run_backtest(**rb_kwargs)

    memory.record_run(
        run_id=run.run_id,
        strategy=run.strategy,
        symbol=run.symbol,
        start_date=run.start,
        end_date=run.end,
        params=run.params,
        starting_cash=run.starting_cash,
        metrics={
            "ending_equity": run.metrics.ending_equity,
            "total_return_pct": run.metrics.total_return_pct,
            "cagr_pct": run.metrics.cagr_pct,
            "sharpe": run.metrics.sharpe,
            "max_drawdown_pct": run.metrics.max_drawdown_pct,
            "num_fills": run.metrics.num_fills,
            "num_round_trips": run.metrics.num_round_trips,
            "win_rate_pct": run.metrics.win_rate_pct,
        },
        artifact_dir=str(run.artifact_dir.relative_to(PROJECT_ROOT)),
    )

    return {
        "run_id": run.run_id,
        "strategy": run.strategy,
        "symbol": run.symbol,
        "start": run.start,
        "end": run.end,
        "params": run.params,
        "slippage_bps": run.slippage_bps,
        "commission_per_trade": run.commission_per_trade,
        "metrics": dict(run.metrics.as_table()),
    }


def _search_memory(args: dict) -> dict:
    memory: Memory = args["__memory"]
    rows = memory.search_runs(
        strategy=args.get("strategy"),
        symbol=args.get("symbol"),
        min_sharpe=args.get("min_sharpe"),
        order_by=args.get("order_by", "sharpe"),
        limit=int(args.get("limit", 10)),
    )
    trimmed = [
        {
            "run_id": r["run_id"],
            "created_at": r["created_at"],
            "strategy": r["strategy"],
            "symbol": r["symbol"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "params": json.loads(r["params_json"]),
            "sharpe": r["sharpe"],
            "cagr_pct": r["cagr_pct"],
            "total_return_pct": r["total_return_pct"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "num_round_trips": r["num_round_trips"],
            "win_rate_pct": r["win_rate_pct"],
        }
        for r in rows
    ]
    return {"count": len(trimmed), "runs": trimmed}


def _get_run_details(args: dict) -> dict:
    memory: Memory = args["__memory"]
    row = memory.get_run(args["run_id"])
    if not row:
        raise KeyError(f"run_id {args['run_id']!r} not found")
    return {
        **{k: v for k, v in row.items() if k != "params_json"},
        "params": json.loads(row["params_json"]),
    }


def _list_filings_tool(args: dict) -> dict:
    from ..data.edgar_source import list_filings

    refs = list_filings(
        ticker=args["ticker"],
        form=args.get("form", "10-Q"),
        limit=int(args.get("limit", 4)),
    )
    return {
        "ticker": args["ticker"].upper(),
        "form": args.get("form", "10-Q"),
        "count": len(refs),
        "filings": [
            {
                "filing_date": r.filing_date,
                "accession_no": r.accession_no,
                "primary_doc_url": r.primary_doc_url,
            }
            for r in refs
        ],
    }


def _fetch_filing_tool(args: dict) -> dict:
    from ..data.edgar_source import filing_excerpt

    return filing_excerpt(
        ticker=args["ticker"],
        accession_no=args["accession_no"],
        max_chars=int(args.get("max_chars", 8000)),
    )


def _compare_runs(args: dict) -> dict:
    memory: Memory = args["__memory"]
    run_ids = args["run_ids"]
    if not isinstance(run_ids, list) or len(run_ids) < 2:
        raise ValueError("run_ids must be a list of at least 2 ids")
    runs = []
    for rid in run_ids:
        row = memory.get_run(rid)
        if not row:
            raise KeyError(f"run_id {rid!r} not found")
        runs.append(
            {
                "run_id": row["run_id"],
                "strategy": row["strategy"],
                "symbol": row["symbol"],
                "params": json.loads(row["params_json"]),
                "sharpe": row["sharpe"],
                "cagr_pct": row["cagr_pct"],
                "total_return_pct": row["total_return_pct"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "num_round_trips": row["num_round_trips"],
                "win_rate_pct": row["win_rate_pct"],
            }
        )
    best = max(runs, key=lambda r: r["sharpe"])
    return {"runs": runs, "best_by_sharpe": best["run_id"]}


# ----- schemas -----

LIST_STRATEGIES = Tool(
    name="list_strategies",
    description="List all strategy modules available under strategies/. Returns module names that can be passed to run_backtest.",
    input_schema={"type": "object", "properties": {}, "required": []},
    fn=_list_strategies,
)

READ_STRATEGY_CODE = Tool(
    name="read_strategy_code",
    description="Return the Python source of a strategy module so you can inspect its logic and parameters before backtesting it.",
    input_schema={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Strategy module name, e.g. 'sma_cross'"}},
        "required": ["name"],
    },
    fn=_read_strategy_code,
)

RUN_BACKTEST = Tool(
    name="run_backtest",
    description=(
        "Run a historical backtest of a strategy on one symbol. Returns metrics "
        "(Sharpe, CAGR, max drawdown, win rate, etc.) and a run_id that can be "
        "passed to get_run_details or compare_runs. Order-of-operations: orders "
        "submitted at bar t fill at bar t+1's OPEN (no lookahead bias). "
        "Defaults: 5 bps slippage on each fill, 0 commission. These are "
        "configurable and the values used are returned with the result."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "strategy": {"type": "string", "description": "Strategy module name from list_strategies"},
            "symbol": {"type": "string", "description": "Ticker, e.g. 'AAPL'"},
            "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end": {"type": "string", "description": "End date YYYY-MM-DD"},
            "params": {
                "type": "object",
                "description": "Strategy constructor params (e.g. {'fast': 20, 'slow': 50}). Optional.",
            },
            "starting_cash": {"type": "number", "description": "Starting cash (default 100000)"},
            "slippage_bps": {
                "type": "number",
                "description": "Override default slippage in basis points (default 5). Set to 0 to disable.",
            },
            "commission_per_trade": {
                "type": "number",
                "description": "Override default $-per-fill commission (default 0).",
            },
        },
        "required": ["strategy", "symbol", "start", "end"],
    },
    fn=_run_backtest_tool,
)

SEARCH_MEMORY = Tool(
    name="search_memory",
    description=(
        "Search past backtest runs persisted in memory. Filter by strategy, symbol, "
        "min Sharpe; order by sharpe / cagr_pct / total_return_pct / max_drawdown_pct / created_at. "
        "Use this before running new backtests to avoid duplicating work."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "strategy": {"type": "string"},
            "symbol": {"type": "string"},
            "min_sharpe": {"type": "number"},
            "order_by": {
                "type": "string",
                "enum": ["sharpe", "cagr_pct", "total_return_pct", "max_drawdown_pct", "created_at"],
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": [],
    },
    fn=_search_memory,
)

GET_RUN_DETAILS = Tool(
    name="get_run_details",
    description="Fetch full details of one past backtest by run_id.",
    input_schema={
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    },
    fn=_get_run_details,
)

COMPARE_RUNS = Tool(
    name="compare_runs",
    description="Compare two or more past backtests side by side and identify the best by Sharpe.",
    input_schema={
        "type": "object",
        "properties": {
            "run_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            }
        },
        "required": ["run_ids"],
    },
    fn=_compare_runs,
)


LIST_FILINGS = Tool(
    name="list_filings",
    description=(
        "List recent SEC EDGAR filings for a ticker. Use this to find filing "
        "accession numbers before calling fetch_filing. Default form is 10-Q "
        "(quarterly); use '10-K' for annual or '8-K' for material events."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker, e.g. 'AAPL'"},
            "form": {
                "type": "string",
                "description": "Filing form type: '10-Q', '10-K', '8-K', etc. Default '10-Q'.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Max filings to return (default 4).",
            },
        },
        "required": ["ticker"],
    },
    fn=_list_filings_tool,
)

FETCH_FILING = Tool(
    name="fetch_filing",
    description=(
        "Fetch text excerpt from a specific SEC filing by ticker + accession number. "
        "Returns truncated text (default 8000 chars) plus metadata. Use the LLM's own "
        "reasoning over the returned text to extract sentiment, risk-factor changes, "
        "or guidance. Use list_filings first to find the accession_no."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "accession_no": {
                "type": "string",
                "description": "Filing accession number from list_filings (e.g. '0000320193-26-000013')",
            },
            "max_chars": {
                "type": "integer",
                "minimum": 500,
                "maximum": 40000,
                "description": "Maximum text chars to return (default 8000).",
            },
        },
        "required": ["ticker", "accession_no"],
    },
    fn=_fetch_filing_tool,
)


ALL_TOOLS: list[Tool] = [
    LIST_STRATEGIES,
    READ_STRATEGY_CODE,
    RUN_BACKTEST,
    SEARCH_MEMORY,
    GET_RUN_DETAILS,
    COMPARE_RUNS,
    LIST_FILINGS,
    FETCH_FILING,
]


def build_tool_registry(memory: Memory) -> dict[str, Tool]:
    """Return a name → Tool map, with memory injected into tools that need it."""
    registry: dict[str, Tool] = {}
    for tool in ALL_TOOLS:
        original_fn = tool.fn

        def wrap(fn, mem=memory):
            def wrapped(args: dict):
                return fn({**args, "__memory": mem})
            return wrapped

        bound_tool = Tool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            fn=wrap(original_fn),
        )
        registry[tool.name] = bound_tool
    return registry
