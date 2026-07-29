import unittest

from trading_agent.autonomous import _select_candidate
from trading_agent.experiment_graph import BacktestCandidateRecord


def candidate(*, strategy: str, symbol: str, score: float, verdict: str) -> BacktestCandidateRecord:
    return BacktestCandidateRecord(
        strategy=strategy,
        symbol=symbol,
        start_date="2024-01-01",
        end_date="2025-01-01",
        params={},
        run_ref="test",
        sharpe=1.0,
        cagr_pct=10.0,
        total_return_pct=10.0,
        max_drawdown_pct=-5.0,
        benchmark_sharpe=0.5,
        benchmark_cagr_pct=5.0,
        score=score,
        verdict=verdict,
        artifact_dir=None,
    )


class AutonomousSelectionTests(unittest.TestCase):
    def test_strict_candidate_beats_higher_scoring_weak_candidate(self):
        weak = candidate(strategy="rsi_mean_rev", symbol="AAPL", score=0.624, verdict="candidate_statistically_weak")
        strict = candidate(strategy="rsi_mean_rev", symbol="NVDA", score=0.101, verdict="candidate")
        self.assertEqual(_select_candidate([weak, strict]), strict)

    def test_returns_none_when_no_candidate_passes_all_gates(self):
        weak = candidate(strategy="rsi_mean_rev", symbol="AAPL", score=0.624, verdict="candidate_statistically_weak")
        rejected = candidate(strategy="sma_cross", symbol="SPY", score=-10.0, verdict="reject_no_trades")
        self.assertIsNone(_select_candidate([weak, rejected]))


if __name__ == "__main__":
    unittest.main()
