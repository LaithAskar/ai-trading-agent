from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading_agent.backtest.rigor import (
    buy_and_hold,
    sharpe_significance,
    walk_forward_splits,
)
from trading_agent.core.events import Bar


def _bars(prices: list[tuple[float, float]], symbol: str = "TEST") -> list[Bar]:
    start = datetime(2024, 1, 1)
    return [
        Bar(
            symbol=symbol,
            timestamp=start + timedelta(days=i),
            open=o,
            high=max(o, c),
            low=min(o, c),
            close=c,
            volume=1_000_000,
        )
        for i, (o, c) in enumerate(prices)
    ]


def test_buy_and_hold_fills_at_second_bar_open():
    bars = _bars([(10, 11), (12, 14), (15, 18), (16, 20)])
    result = buy_and_hold(bars, starting_cash=100.0)
    # 100 // 12 = 8 shares, cash remaining = 100 - 96 = 4
    # Final equity = 4 + 8*20 = 164
    assert result.end_equity == pytest.approx(164.0)
    assert result.total_return_pct > 0


def test_buy_and_hold_empty_returns_starting_cash():
    result = buy_and_hold([], starting_cash=100.0)
    assert result.end_equity == 100.0


def test_sharpe_significance_zero_returns():
    t_stat, p = sharpe_significance([0.0, 0.0, 0.0, 0.0])
    assert t_stat == 0.0
    assert p == 1.0


def test_sharpe_significance_large_positive():
    """100 daily returns of +0.001 with low variance → very significant."""
    rets = [0.001] * 100
    # std would be 0 here — let's add tiny noise
    rets = [0.001 + (i % 3 - 1) * 1e-5 for i in range(100)]
    t_stat, p = sharpe_significance(rets)
    assert t_stat > 5
    assert p < 0.001


def test_sharpe_significance_small_sample_high_p():
    """3 days of returns — noisy, shouldn't be significant."""
    t_stat, p = sharpe_significance([0.01, -0.005, 0.008])
    assert p > 0.05  # not significant at standard threshold


def test_walk_forward_basic_split():
    splits = walk_forward_splits("2018-01-01", "2024-12-31", train_years=3.0, test_years=1.0)
    assert len(splits) >= 3
    # First split: 2018-2021 train, 2021-2022 test
    s0 = splits[0]
    assert s0.train_start == "2018-01-01"
    # No overlap between this train and its test
    assert s0.train_end == s0.test_start


def test_walk_forward_no_overlap_between_train_and_its_test():
    splits = walk_forward_splits("2020-01-01", "2025-12-31", train_years=2.0, test_years=1.0)
    for s in splits:
        train_end = datetime.strptime(s.train_end, "%Y-%m-%d")
        test_start = datetime.strptime(s.test_start, "%Y-%m-%d")
        assert test_start >= train_end, f"train/test overlap in {s}"


def test_walk_forward_returns_empty_when_window_too_small():
    splits = walk_forward_splits("2024-01-01", "2024-06-01", train_years=3.0, test_years=1.0)
    assert splits == []
