# Trading Agent

A backtest + AI research agent for stock trading strategies. Built ground-up — no LangChain, no agent frameworks. Direct Anthropic SDK with a hand-written ReAct loop.

## What's in the box

- **Backtest engine** with a pinned lookahead-safety contract (orders at bar `t` fill at bar `t+1`'s open), configurable slippage (default 5 bps) and commission (default $0).
- **Pluggable Strategy interface** — drop a Python class in `strategies/` and the platform discovers it. Reference strategies: `sma_cross`, `rsi_mean_rev`.
- **AI agent** that runs the platform autonomously via a ReAct loop:
  - Thinks in plain English about what to do next
  - Calls tools (`list_strategies`, `run_backtest`, `search_memory`, ...)
  - Reads observations, decides next action
  - Stops when it has a final answer
- **SQLite-backed run memory** so the agent doesn't repeat work across sessions.
- **Two run modes**: `--auto` (fully autonomous) and `--interactive` (approve each tool call).
- **CLI**: backtest, list strategies, run agent.
- **25 tests** including lookahead-safety regression tests, slippage/commission correctness, and a mock-driven loop test.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# then edit .env and add ANTHROPIC_API_KEY=sk-ant-...
```

## Run a backtest (no LLM)

```powershell
python -m trading_agent backtest --strategy sma_cross --symbol AAPL --start 2020-01-01 --end 2024-12-31 --param fast=20 --param slow=50
```

Outputs `data/results/<run>/`:
- `summary.json` — metrics + config
- `equity_curve.csv` + `equity_curve.png`
- `trades.csv` — every fill

## Run the agent

```powershell
# fully autonomous
python -m trading_agent agent --goal "Backtest sma_cross on AAPL and MSFT from 2022 to 2024 with three different parameter combinations. Tell me which performed best by Sharpe and why I should be skeptical."

# step-by-step (you approve each tool call)
python -m trading_agent agent --goal "..." --mode interactive

# pick the model
python -m trading_agent agent --goal "..." --model claude-opus-4-7
```

The agent emits visible **Thought → Action → Observation** panels as it runs. Every session is logged to `data/logs/agent_runs/<session_id>.json` and indexed in `data/memory.sqlite3` so the next session can `search_memory` to find prior work.

A rendered sample transcript lives at [`docs/examples/sample-session.html`](docs/examples/sample-session.html) — that one shows the agent finding two prior backtests in memory and skipping the re-runs.

## Agent architecture

```
trading_agent/agent/
├── prompts.py    # System prompt: ReAct discipline + constraints
├── tools.py      # 6 tools w/ Anthropic JSON schemas + impls
├── memory.py     # SQLite store for runs + sessions
└── loop.py       # ReAct driver: thought → action → observation
```

**Tools the agent has:**
| Tool | Purpose |
|---|---|
| `list_strategies` | Discover strategy modules |
| `read_strategy_code` | Inspect a strategy before testing it |
| `run_backtest` | Execute a backtest, returns metrics + run_id |
| `search_memory` | Query past runs (by strategy, symbol, min sharpe) |
| `get_run_details` | Fetch one past run by id |
| `compare_runs` | Side-by-side comparison of 2+ runs |

**Tools the agent does NOT have**, by design:
- No `place_live_order`. Live trading is physically impossible in this build.
- No `install_package`, no `shell`, no `read_file` outside strategies/.
- No code generation. (Coming in V3 with AST sandboxing.)

## Adding a strategy

1. Create `strategies/my_thing.py`.
2. Subclass `trading_agent.core.strategy.Strategy`, set `name`, implement `on_bar(bar, portfolio) -> list[Order]`.
3. Run `python -m trading_agent list-strategies` to confirm pickup.
4. Backtest it or ask the agent to.

See `strategies/sma_cross.py` for a reference.

## The lookahead-safety contract

`tests/test_backtest.py` is the spec, not just a test file:

1. Orders submitted by `on_bar()` at bar `t` fill at bar `t+1`'s **open** price.
2. Mark-to-market uses bar `t`'s **close**.
3. `on_bar()` only ever receives bars up to and including the current one.
4. Orders pending after the final bar are dropped (no fill).

If you change `backtest/engine.py`, those tests are your tripwire. Don't let them go red.

## Scope and roadmap

**Why these decisions, in plain English:**

- **No LLM-in-the-strategy.** The agent uses an LLM to *orchestrate* (decide what to backtest, what to compare, how to summarize). The strategies themselves are deterministic Python. This is a deliberate choice: an LLM that emits buy/sell decisions on the fly is unauditable, expensive, and non-reproducible. The orchestrator pattern (LLM as planner over deterministic tools) is what production agents like Cursor and Claude Code actually use.
- **Structured SQLite memory, not vector search.** Run history has well-defined fields (symbol, strategy, sharpe, dates). Filtering by `sharpe > X` is more useful than semantic similarity. Vector retrieval gets relevant when we store free-form analysis notes — not yet.
- **Hand-written ReAct loop, no LangChain.** Frameworks abstract away exactly the parts of an agent's behavior you most need to inspect and control. Building from scratch keeps the loop visible to the developer and the user.
- **Strategy generation deferred to V3.** Letting the LLM write Python that then runs is the showy demo. It's also the safety risk: arbitrary code execution, AST whitelisting, sandboxing — non-trivial. Doing it after the platform is solid is the right order.
- **Single-symbol backtests for V1.** Multi-symbol portfolios change the `Strategy` interface (per-symbol state, position sizing across the basket, correlation). Stress-test the single-symbol interface with 3+ strategies first. V3.
- **Alpaca paper-trading deferred to V3.** Same `Strategy` interface, different runner. Needs API integration + a separate run loop. Comes after the strategy interface is proven across more strategies.

## Known limitations (V2)

- **Long-only.** No shorts, no margin, no options.
- **5 bps slippage, 0 commission** by default. Configurable per backtest via `--param` (CLI) or args (agent tool). Real illiquid-name fills can be worse.
- **Whole-share quantities.** Fractional shares are V3.
- **Single symbol per backtest.** Strategy interface assumes one symbol at a time.
- **yfinance** has survivorship bias — delisted tickers are not present. Long-history backtests on speculative names are unreliable.
- **Daily bars only.** Intraday is V3.
- **No timezone handling** beyond what yfinance provides — fine for US equities, would need work for cross-market.

## Tests

```powershell
pytest
```

25 tests covering: lookahead safety, slippage + commission application, portfolio bookkeeping, memory persistence, tool schemas + path-traversal blocking, and the loop driver (mocked).
