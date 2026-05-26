from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.events import Bar
from .metrics import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class BenchmarkResult:
    symbol: str
    start_equity: float
    end_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float


def buy_and_hold(bars: list[Bar], starting_cash: float) -> BenchmarkResult:
    """Compute equity, return, CAGR, Sharpe, max-DD for buying at the first bar's
    OPEN with all of starting_cash and holding through the last bar.

    Matches the engine's fill model (next-bar open) so the comparison is apples-
    to-apples: a strategy that emits a BUY on bar 0 would also fill at bar 1's open.
    """
    if len(bars) < 2:
        return BenchmarkResult(bars[0].symbol if bars else "", starting_cash, starting_cash, 0, 0, 0, 0)

    fill_price = bars[1].open
    qty = int(starting_cash // fill_price)
    cash = starting_cash - qty * fill_price

    equity_series = []
    timestamps = []
    for bar in bars[1:]:
        equity = cash + qty * bar.close
        equity_series.append(equity)
        timestamps.append(bar.timestamp)

    s = pd.Series(equity_series, index=pd.to_datetime(timestamps))

    start_eq = float(s.iloc[0])
    end_eq = float(s.iloc[-1])
    total_ret = end_eq / start_eq - 1.0
    days = (s.index[-1] - s.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (end_eq / start_eq) ** (1 / years) - 1.0 if start_eq > 0 else 0.0

    daily = s.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * math.sqrt(TRADING_DAYS_PER_YEAR)) if daily.std() > 0 else 0.0

    running_max = s.cummax()
    drawdown = (s / running_max) - 1.0
    max_dd = float(drawdown.min())

    return BenchmarkResult(
        symbol=bars[0].symbol,
        start_equity=starting_cash,
        end_equity=end_eq,
        total_return_pct=total_ret * 100,
        cagr_pct=cagr * 100,
        sharpe=sharpe,
        max_drawdown_pct=max_dd * 100,
    )


def sharpe_significance(daily_returns: Iterable[float]) -> tuple[float, float]:
    """t-statistic and approximate p-value for H0: Sharpe == 0.

    Under the standard normal-IID assumption (which is wrong but is the
    textbook test), the annualized Sharpe estimator has standard error
    sqrt(N / TRADING_DAYS_PER_YEAR), where N is the number of daily-return
    observations. The t-stat is Sharpe / SE.

    p-value is two-sided, computed from a normal approximation (cheap, good
    enough for N >> 30).
    """
    r = np.asarray(list(daily_returns), dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0, 1.0

    sharpe_annualized = r.mean() / r.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    se = math.sqrt(TRADING_DAYS_PER_YEAR / len(r))
    t_stat = sharpe_annualized / se if se > 0 else 0.0
    p_value = 2 * (1 - _phi(abs(t_stat)))
    return float(t_stat), float(p_value)


def _phi(x: float) -> float:
    """Standard normal CDF via erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def walk_forward_splits(
    start: str,
    end: str,
    train_years: float = 3.0,
    test_years: float = 1.0,
    stride_years: float = 1.0,
) -> list[WalkForwardSplit]:
    """Generate rolling train/test windows for walk-forward validation.

    Each split has a train window of `train_years` followed immediately by a
    test window of `test_years`. The next split's train window starts
    `stride_years` after the previous one's. No overlap between train and
    its own test (the bedrock invariant of walk-forward).
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    splits: list[WalkForwardSplit] = []
    cursor = s
    train_days = int(train_years * 365.25)
    test_days = int(test_years * 365.25)
    stride_days = int(stride_years * 365.25)

    while cursor + pd.Timedelta(days=train_days + test_days) <= e:
        from datetime import timedelta
        train_start = cursor
        train_end = cursor + timedelta(days=train_days)
        test_start = train_end
        test_end = train_end + timedelta(days=test_days)
        splits.append(
            WalkForwardSplit(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            )
        )
        cursor = cursor + timedelta(days=stride_days)

    return splits
