from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import RESULTS_DIR
from ..core.strategy import Strategy
from ..data.yfinance_source import iter_bars, load_bars
from .engine import BacktestEngine
from .metrics import Metrics, compute_metrics, equity_curve_df
from .rigor import BenchmarkResult, default_baselines, sharpe_significance


@dataclass
class BacktestRun:
    run_id: str
    strategy: str
    symbol: str
    start: str
    end: str
    params: dict
    starting_cash: float
    slippage_bps: float
    commission_per_trade: float
    metrics: Metrics
    benchmark: BenchmarkResult | None
    baselines: dict[str, BenchmarkResult]
    sharpe_t_stat: float
    sharpe_p_value: float
    artifact_dir: Path


def load_strategy(name: str, params: dict | None = None) -> Strategy:
    """Load `strategies.<name>` and instantiate the first Strategy subclass found."""
    module = importlib.import_module(f"strategies.{name}")
    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and issubclass(attr, Strategy)
            and attr is not Strategy
            and attr.__module__ == module.__name__
        ):
            return attr(**(params or {}))
    raise ValueError(f"No Strategy subclass found in strategies.{name}")


def _metric_value(value: float) -> float | str:
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return value


def _benchmark_summary(benchmark: BenchmarkResult) -> dict:
    return {
        "name": benchmark.name,
        "total_return_pct": benchmark.total_return_pct,
        "cagr_pct": benchmark.cagr_pct,
        "sharpe": benchmark.sharpe,
        "volatility_pct": benchmark.volatility_pct,
        "max_drawdown_pct": benchmark.max_drawdown_pct,
        "exposure_pct": benchmark.exposure_pct,
        "ending_equity": benchmark.end_equity,
    }


def _write_markdown_report(run_dir: Path, summary: dict, baselines: dict[str, BenchmarkResult]) -> Path:
    metrics = summary["metrics"]
    lines = [
        f"# Backtest report: {summary['strategy']} on {summary['symbol']}",
        "",
        "Research/backtesting-only report. Results are historical simulations and are not financial advice or live-trading readiness claims.",
        "",
        "## Run configuration",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Window: {summary['start']} to {summary['end']}",
        f"- Starting cash: ${summary['starting_cash']:,.2f}",
        f"- Slippage: {summary['slippage_bps']:.2f} bps",
        f"- Commission per trade: ${summary['commission_per_trade']:,.2f}",
        f"- Params: `{summary['params']}`",
        "",
        "## Strategy metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Baseline comparison",
        "",
        "| Baseline | Total return | CAGR | Sharpe | Volatility | Max DD | Exposure | Ending equity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.append(
        "| Strategy | "
        f"{summary['strategy_metrics_raw']['total_return_pct']:.2f}% | "
        f"{summary['strategy_metrics_raw']['cagr_pct']:.2f}% | "
        f"{summary['strategy_metrics_raw']['sharpe']:.2f} | "
        f"{summary['strategy_metrics_raw']['volatility_pct']:.2f}% | "
        f"{summary['strategy_metrics_raw']['max_drawdown_pct']:.2f}% | "
        f"{summary['strategy_metrics_raw']['exposure_pct']:.2f}% | "
        f"${summary['strategy_metrics_raw']['ending_equity']:,.2f} |"
    )
    for name, b in baselines.items():
        lines.append(
            f"| {name} | {b.total_return_pct:.2f}% | {b.cagr_pct:.2f}% | {b.sharpe:.2f} | "
            f"{b.volatility_pct:.2f}% | {b.max_drawdown_pct:.2f}% | {b.exposure_pct:.2f}% | ${b.end_equity:,.2f} |"
        )

    lines.extend([
        "",
        "## Statistical caution",
        "",
        f"- Sharpe t-stat: {summary['sharpe_t_stat']:.2f}",
        f"- Approx. two-sided p-value for Sharpe != 0: {summary['sharpe_p_value']:.4f}",
        "- Baselines are deterministic, small-scope comparisons using the same OHLCV sample and no new data dependencies.",
        "",
        "## Artifacts",
        "",
        "- `summary.json`",
        "- `equity_curve.csv`",
        "- `trades.csv`",
        "- `equity_curve.png`",
    ])
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION_PER_TRADE = 0.0


def run_backtest(
    *,
    strategy_name: str,
    symbol: str,
    start: str,
    end: str,
    params: dict | None = None,
    starting_cash: float = 100_000.0,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_per_trade: float = DEFAULT_COMMISSION_PER_TRADE,
    write_artifacts: bool = True,
) -> BacktestRun:
    """Execute one backtest and (optionally) write CSV/PNG/JSON artifacts.

    Default friction: 5 bps slippage (a realistic-floor estimate for liquid
    large-caps), 0 commission (matches retail brokers like Alpaca/Schwab).
    Override via kwargs if you want to model IBKR-style per-share commissions.
    """
    params = params or {}
    strat = load_strategy(strategy_name, params)
    df = load_bars(symbol, start, end)
    bars = list(iter_bars(symbol, df))
    engine = BacktestEngine(
        starting_cash=starting_cash,
        slippage_bps=slippage_bps,
        commission_per_trade=commission_per_trade,
    )
    result = engine.run(strat, symbol, iter(bars))
    metrics = compute_metrics(result.portfolio)

    baselines = default_baselines(bars, starting_cash)
    benchmark = baselines["buy_and_hold"]

    eq_df = equity_curve_df(result.portfolio)
    daily_returns = eq_df["equity"].pct_change().dropna().tolist() if not eq_df.empty else []
    t_stat, p_value = sharpe_significance(daily_returns)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RESULTS_DIR / f"{strat.name}_{symbol}_{run_id}"

    if write_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)

        eq = eq_df
        eq.to_csv(run_dir / "equity_curve.csv")

        trades = pd.DataFrame(
            [
                {
                    "timestamp": f.timestamp,
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                }
                for f in result.portfolio.fills
            ]
        )
        trades.to_csv(run_dir / "trades.csv", index=False)

        summary = {
            "run_id": run_id,
            "strategy": strat.name,
            "params": params,
            "symbol": symbol,
            "start": start,
            "end": end,
            "starting_cash": starting_cash,
            "slippage_bps": slippage_bps,
            "commission_per_trade": commission_per_trade,
            "metrics": dict(metrics.as_table()),
            "strategy_metrics_raw": {
                "starting_equity": metrics.starting_equity,
                "ending_equity": metrics.ending_equity,
                "total_return_pct": metrics.total_return_pct,
                "cagr_pct": metrics.cagr_pct,
                "sharpe": metrics.sharpe,
                "volatility_pct": metrics.volatility_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "exposure_pct": metrics.exposure_pct,
                "num_fills": metrics.num_fills,
                "num_round_trips": metrics.num_round_trips,
                "win_rate_pct": metrics.win_rate_pct,
                "profit_factor": _metric_value(metrics.profit_factor),
            },
            "baselines": {name: _benchmark_summary(b) for name, b in baselines.items()},
            "benchmark_buy_hold": _benchmark_summary(benchmark),
            "sharpe_t_stat": t_stat,
            "sharpe_p_value": p_value,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        _write_markdown_report(run_dir, summary, baselines)

        if not eq.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(eq.index, eq["equity"])
            ax.set_title(f"{strat.name} on {symbol} — equity")
            ax.set_ylabel("Equity ($)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "equity_curve.png", dpi=120)
            plt.close(fig)

    return BacktestRun(
        run_id=run_id,
        strategy=strat.name,
        symbol=symbol.upper(),
        start=start,
        end=end,
        params=params,
        starting_cash=starting_cash,
        slippage_bps=slippage_bps,
        commission_per_trade=commission_per_trade,
        metrics=metrics,
        benchmark=benchmark,
        baselines=baselines,
        sharpe_t_stat=t_stat,
        sharpe_p_value=p_value,
        artifact_dir=run_dir,
    )
