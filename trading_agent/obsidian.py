from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_OBSIDIAN_VAULT = Path("/Users/laithaskar/Documents/Hermes Brain")


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "unknown"


def _artifact_path(run: Any, filename: str) -> str:
    return str(Path(run.artifact_dir) / filename)


def render_backtest_report(run: Any) -> str:
    """Render a cautious Obsidian markdown report for a backtest run.

    The report is intentionally conservative: it records results, benchmark
    context, and explicit non-approval for paper/live trading. It is a research
    log, not a trading signal.
    """
    metrics = run.metrics
    benchmark = run.benchmark
    params_json = json.dumps(run.params, indent=2, sort_keys=True, default=str)
    statistically_significant = run.sharpe_p_value < 0.10
    compared = benchmark is not None
    outperformed_benchmark = False
    if benchmark is not None:
        outperformed_benchmark = metrics.cagr_pct > benchmark.cagr_pct

    benchmark_section = "Benchmark unavailable."
    if benchmark is not None:
        benchmark_section = "\n".join(
            [
                f"- Strategy CAGR: {_fmt_pct(metrics.cagr_pct)}",
                f"- Buy-and-hold CAGR: {_fmt_pct(benchmark.cagr_pct)}",
                f"- Strategy Sharpe: {metrics.sharpe:.2f}",
                f"- Buy-and-hold Sharpe: {benchmark.sharpe:.2f}",
                f"- Strategy max drawdown: {_fmt_pct(metrics.max_drawdown_pct)}",
                f"- Buy-and-hold max drawdown: {_fmt_pct(benchmark.max_drawdown_pct)}",
                f"- Outperformed benchmark by CAGR: {'Yes' if outperformed_benchmark else 'No'}",
            ]
        )

    return f"""# Backtest Report: {run.strategy} on {run.symbol}

## Status
Research only. Not trading advice. Not approved for paper trading or real trading.

## Strategy
- Name: {run.strategy}
- Symbol: {run.symbol}
- Date range: {run.start} → {run.end}
- Starting cash: {_fmt_money(run.starting_cash)}
- Slippage: {run.slippage_bps:.2f} bps
- Commission per trade: {_fmt_money(run.commission_per_trade)}

## Parameters
```json
{params_json}
```

## Metrics
- Ending equity: {_fmt_money(metrics.ending_equity)}
- Total return: {_fmt_pct(metrics.total_return_pct)}
- CAGR: {_fmt_pct(metrics.cagr_pct)}
- Sharpe: {metrics.sharpe:.2f}
- Sharpe t-stat: {run.sharpe_t_stat:.2f}
- Sharpe p-value: {run.sharpe_p_value:.4f}
- Max drawdown: {_fmt_pct(metrics.max_drawdown_pct)}
- Win rate: {_fmt_pct(metrics.win_rate_pct)}
- Fills: {metrics.num_fills}
- Round trips: {metrics.num_round_trips}

## Benchmark Comparison
{benchmark_section}

## Skepticism Checklist
- Compared against buy-and-hold: {'Yes' if compared else 'No'}
- Includes slippage: {'Yes' if run.slippage_bps > 0 else 'No'}
- Statistically significant: {'Yes' if statistically_significant else 'No'}
- Out-of-sample tested: No
- Walk-forward tested: No
- Paper-trade approved: No
- Real-trade approved: No

## Verdict
Needs human review. Do not promote this strategy from one backtest alone.

## Lessons / Notes
- What market regime did this run cover?
- Did it beat buy-and-hold after slippage?
- Is the Sharpe statistically meaningful?
- Could this be overfit to this symbol/date range?
- What follow-up test should run next?

## Artifacts
- summary.json: {_artifact_path(run, 'summary.json')}
- equity_curve.csv: {_artifact_path(run, 'equity_curve.csv')}
- trades.csv: {_artifact_path(run, 'trades.csv')}
"""


def save_backtest_report(run: Any, vault_path: str | Path = DEFAULT_OBSIDIAN_VAULT) -> Path:
    vault = Path(vault_path)
    backtests_dir = vault / "03_Trading" / "Backtests"
    backtests_dir.mkdir(parents=True, exist_ok=True)

    date_prefix = str(run.run_id)[:8]
    filename = f"{date_prefix}_{_safe_slug(run.strategy)}_{_safe_slug(run.symbol)}_{_safe_slug(run.run_id)}.md"
    report_path = backtests_dir / filename
    report_path.write_text(render_backtest_report(run), encoding="utf-8")
    return report_path
