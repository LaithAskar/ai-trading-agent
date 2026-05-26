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
    engine = BacktestEngine(
        starting_cash=starting_cash,
        slippage_bps=slippage_bps,
        commission_per_trade=commission_per_trade,
    )
    result = engine.run(strat, symbol, iter_bars(symbol, df))
    metrics = compute_metrics(result.portfolio)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RESULTS_DIR / f"{strat.name}_{symbol}_{run_id}"

    if write_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)

        eq = equity_curve_df(result.portfolio)
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
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

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
        artifact_dir=run_dir,
    )
