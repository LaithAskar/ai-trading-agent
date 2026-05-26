# Trading Agent

A backtest + AI research agent for stock trading strategies. Built ground-up — no LangChain, no agent frameworks. Direct Anthropic SDK with a hand-written ReAct loop.

**🟢 Try it live**:
- **Web app** (run the agent, backtest, paper trade — all in your browser): https://ai-trading-agent-laith.streamlit.app/
- **Project page** with interactive transcript demos: https://laithaskar.github.io/ai-trading-agent/

## What's in the box

- **Backtest engine** with a pinned lookahead-safety contract (orders at bar `t` fill at bar `t+1`'s open), configurable slippage (default 5 bps) and commission (default $0).
- **Pluggable Strategy interface** — drop a Python class in `strategies/` and the platform discovers it. Reference strategies: `sma_cross`, `rsi_mean_rev`, `filings_sentiment` (uses SEC EDGAR data).
- **AI agent** that runs the platform autonomously via a ReAct loop:
  - Thinks in plain English about what to do next
  - Calls tools (`list_strategies`, `run_backtest`, `search_memory`, ...)
  - Reads observations, decides next action
  - Stops when it has a final answer
- **SQLite-backed run memory** so the agent doesn't repeat work across sessions.
- **Two run modes**: `--auto` (fully autonomous) and `--interactive` (approve each tool call).
- **CLI**: backtest, list strategies, run agent, connect to remote MCP servers, replay past sessions, agent-stats, render HTML transcripts.
- **78 tests** including lookahead-safety regression tests, slippage/commission correctness, and a mock-driven loop test.

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

Rendered interactive transcripts — open in a browser, press **▶ Play** (or **Space**, or **→**) to step through the agent's reasoning iteration by iteration:
- [`docs/examples/v6-flagship-session.html`](docs/examples/v6-flagship-session.html) — **flagship demo**. Agent compares two AAPL strategies, finds prior backtests via memory (skips redundant work), pulls Apple's current Q2 FY2026 10-Q via EDGAR, and writes a regime analysis tying the filing's buyback / R&D / services data back to which strategy's regime is likely to persist. 7 iterations, $0.0845, all 8 tools exercised.
- [`docs/examples/sample-session.html`](docs/examples/sample-session.html) — earlier demo, same interactive viewer.

## Observability and safety (V3)

**Cost caps.** Every session has hard limits on cumulative tokens *and* dollars. If either is exceeded, the loop stops immediately, the cap-hit reason is logged in the session JSON's `stopped_by` field, and a red "Session Cap Hit" panel appears in the terminal. Configurable via `AGENT_MAX_SESSION_DOLLARS` and `AGENT_MAX_SESSION_TOKENS`, or `--max-dollars` / `--max-tokens` flags. This is the answer to "how do you prevent the agent from looping and burning $5k by accident."

**Per-tool observability.** Every tool execution records its duration and whether it errored. Aggregate across all sessions:

```powershell
python -m trading_agent agent-stats
```

Shows total $/tokens, per-tool call count + avg/p95 latency + error rate, and a recent-sessions table.

**Replay for regression detection.** Given any past session JSON, the `replay` command re-executes every tool call against current code and diffs the outputs:

```powershell
python -m trading_agent replay <session_id>   # or "latest"
```

Drift detection knows to redact volatile fields (timestamps, run IDs, artifact paths) so the diff highlights real semantic changes — e.g., adding 5 bps slippage caused `run_backtest` to return 20.67% instead of 25.92% on the same input, and replay catches that.

*Known limitation*: replaying `run_backtest` actually executes a new backtest (and adds a row to memory). Read-only replay mode is a future improvement.

## Paper trading via Alpaca (V5)

Same `Strategy` interface, different runner. Set `ALPACA_API_KEY` and `ALPACA_API_SECRET` in `.env` (paper credentials from app.alpaca.markets/paper/dashboard/overview).

```powershell
# Check account state
python -m trading_agent paper-status

# Dry-run: replay strategy history, show what orders WOULD be submitted
python -m trading_agent paper-trade --strategy sma_cross --symbol AAPL --param fast=20 --param slow=50

# Actually submit
python -m trading_agent paper-trade --strategy sma_cross --symbol AAPL --param fast=20 --param slow=50 --no-dry-run
```

The broker is hardcoded to paper. `AlpacaPaperBroker(..., allow_live=True)` raises `NotImplementedError` — live trading is *physically impossible* in this codebase, by design.

## SEC EDGAR filings + filings_sentiment strategy (V5)

```powershell
# Backtest the filings-sentiment strategy
python -m trading_agent backtest --strategy filings_sentiment --symbol AAPL --start 2023-01-01 --end 2025-12-31

# Agent can also fetch + reason over filings via the list_filings and fetch_filing tools
python -m trading_agent agent --goal "Look at the most recent 10-Q for NVDA and tell me whether sentiment has improved or worsened versus the prior quarter."
```

The reference strategy uses Loughran-McDonald-style word counting on filing text and trades on sentiment **momentum** (quarter-over-quarter change) rather than absolute level — SEC filings are mandated to disclose risks so absolute scores are always negative; what matters is the direction.

A second variant, `filings_sentiment_llm`, replaces the word counter with Claude reading the filing. Same momentum logic, much better signal extraction. Results are cached in `data/llm_cache.sqlite3` keyed on `(ticker, accession, model, prompt_hash)` so reruns are free + deterministic. Pass `cache_only=true` as a strategy param to refuse new API calls (e.g., for cost-controlled rebacktests).

```powershell
# First run: hits Claude on each filing, caches results
python -m trading_agent backtest --strategy filings_sentiment_llm --symbol AAPL --start 2023-01-01 --end 2025-12-31

# Reruns: free + deterministic from cache
python -m trading_agent backtest --strategy filings_sentiment_llm --symbol AAPL --start 2023-01-01 --end 2025-12-31 --param cache_only=true
```

## Quant rigor (V5)

Every backtest now also computes the buy-and-hold benchmark for the same window and a Sharpe significance test (t-stat + two-sided p-value vs. the null Sharpe=0). When p ≥ 0.10 the CLI surfaces a yellow warning.

**Parameter sweep:**

```powershell
# Cartesian grid: 9 backtests, ranked by Sharpe
python -m trading_agent param-sweep --strategy sma_cross --symbol AAPL --start 2020-01-01 --end 2024-12-31 --grid "fast=10,20,50" --grid "slow=50,100,200"

# Combine with walk-forward — each combo runs through rolling splits,
# ranked by OUT-OF-SAMPLE Sharpe mean. The robust version.
python -m trading_agent param-sweep --strategy sma_cross --symbol AAPL --start 2018-01-01 --end 2024-12-31 --grid "fast=10,20,50" --grid "slow=50,100,200" --walk-forward --train-years 2 --test-years 1
```

The sweep on `sma_cross` over AAPL 2020–2024 found that **no parameter combination beat buy-and-hold** (28.5% CAGR), and most combos had a Sharpe p-value > 0.1 — i.e., statistically indistinguishable from random. That's not a failure of the platform; that's the platform doing its job.

**Walk-forward validation:**

```powershell
python -m trading_agent walk-forward --strategy sma_cross --symbol AAPL --start 2018-01-01 --end 2024-12-31 --train-years 2 --test-years 1 --param fast=20 --param slow=50
```

Runs the strategy on rolling train→test splits and reports OOS CAGR per split plus the mean+stdev across splits. Most strategies that look great on a single backtest fall apart here — that's the point.

## Remote MCP servers (V4)

The agent can also use tools from any remote Model Context Protocol server — NexusTrade, GitHub, anything that speaks MCP over HTTP with OAuth. The flow:

```powershell
# One-time: authenticate (opens your browser for OAuth)
python -m trading_agent mcp-connect https://mcp.example.com

# Then point the agent at it. Built-in + remote tools coexist in one registry.
python -m trading_agent agent --mcp-server https://mcp.example.com --goal "..."
```

Tokens are persisted under `data/mcp/` so subsequent runs don't re-authenticate. Remote tools are exposed to the agent with an `mcp_` prefix and a `[MCP: <url>]` label in the description, so the LLM can tell them apart from built-ins. Name collisions log a warning and remote wins.

OAuth implementation details:
- Local HTTP server on a free port catches the redirect (standard desktop OAuth pattern).
- `mcp.client.auth.OAuthClientProvider` from the official SDK handles client registration, code exchange, and refresh.
- `FileTokenStorage` persists tokens + client info as JSON, one file pair per server URL.

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

78 tests covering: lookahead safety, slippage + commission application, portfolio bookkeeping, memory persistence, tool schemas + path-traversal blocking, and the loop driver (mocked).
