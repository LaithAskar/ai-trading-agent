from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading_agent.obsidian import render_backtest_report, save_backtest_report


@dataclass
class DummyMetrics:
    starting_equity: float = 100_000.0
    ending_equity: float = 110_000.0
    total_return_pct: float = 10.0
    cagr_pct: float = 9.5
    sharpe: float = 1.2
    max_drawdown_pct: float = -8.0
    num_fills: int = 4
    num_round_trips: int = 2
    win_rate_pct: float = 50.0


@dataclass
class DummyBenchmark:
    total_return_pct: float = 6.0
    cagr_pct: float = 5.7
    sharpe: float = 0.8
    max_drawdown_pct: float = -12.0
    end_equity: float = 106_000.0


@dataclass
class DummyRun:
    run_id: str = "20260602_120000_000000"
    strategy: str = "sma_cross"
    symbol: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    params: dict[str, Any] = field(default_factory=lambda: {"fast": 20, "slow": 50})
    starting_cash: float = 100_000.0
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    metrics: DummyMetrics = field(default_factory=DummyMetrics)
    benchmark: DummyBenchmark = field(default_factory=DummyBenchmark)
    sharpe_t_stat: float = 1.1
    sharpe_p_value: float = 0.27
    artifact_dir: Path = Path("data/results/sma_cross_AAPL_20260602_120000_000000")


def test_render_backtest_report_includes_safety_and_skepticism_sections():
    markdown = render_backtest_report(DummyRun())

    assert "# Backtest Report: sma_cross on AAPL" in markdown
    assert "Research only. Not trading advice. Not approved for paper trading or real trading." in markdown
    assert "## Benchmark Comparison" in markdown
    assert "Sharpe p-value: 0.2700" in markdown
    assert "Statistically significant: No" in markdown
    assert "Paper-trade approved: No" in markdown
    assert "Needs human review" in markdown
    assert "data/results/sma_cross_AAPL_20260602_120000_000000/summary.json" in markdown


def test_save_backtest_report_writes_under_backtests_folder(tmp_path):
    report_path = save_backtest_report(DummyRun(), vault_path=tmp_path)

    assert report_path.parent == tmp_path / "03_Trading" / "Backtests"
    assert report_path.name == "20260602_sma_cross_AAPL_20260602_120000_000000.md"
    assert report_path.read_text(encoding="utf-8").startswith("# Backtest Report: sma_cross on AAPL")
