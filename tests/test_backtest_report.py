from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_agent.backtest import runner
from trading_agent.core.orders import Order, Side
from trading_agent.core.strategy import Strategy


class DeterministicBuyHold(Strategy):
    name = "deterministic_buy_hold"

    def __init__(self):
        self._i = -1

    def on_bar(self, bar, portfolio):
        self._i += 1
        if self._i == 0:
            return [Order(symbol=bar.symbol, side=Side.BUY, quantity=10)]
        return []


def test_run_backtest_writes_markdown_report_and_baseline_summary(monkeypatch, tmp_path: Path):
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    prices = [100.0 + i for i in range(80)]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=dates,
    )

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "load_bars", lambda symbol, start, end: df)
    monkeypatch.setattr(runner, "load_strategy", lambda name, params=None: DeterministicBuyHold())

    run = runner.run_backtest(
        strategy_name="ignored_by_monkeypatch",
        symbol="TEST",
        start="2024-01-01",
        end="2024-03-21",
        starting_cash=10_000.0,
        slippage_bps=0.0,
        write_artifacts=True,
    )

    summary_path = run.artifact_dir / "summary.json"
    report_path = run.artifact_dir / "report.md"
    assert summary_path.exists()
    assert report_path.exists()

    report = report_path.read_text()
    assert "Research/backtesting-only report" in report
    assert "Baseline comparison" in report
    assert "buy_and_hold" in report
    assert "sma_20_50" in report
    assert "momentum_20" in report
    assert set(run.baselines) == {"buy_and_hold", "sma_20_50", "momentum_20"}
