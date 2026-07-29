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
    name: str
    start_equity: float
    end_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    volatility_pct: float
    max_drawdown_pct: float
    exposure_pct: float


def _metrics_from_equity(
    *,
    symbol: str,
    name: str,
    equity_series: list[float],
    timestamps: list,
    starting_cash: float,
    exposure_count: int,
) -> BenchmarkResult:
    if len(equity_series) < 2:
        return BenchmarkResult(symbol, name, starting_cash, starting_cash, 0, 0, 0, 0, 0, 0)

    s = pd.Series(equity_series, index=pd.to_datetime(timestamps))
    start_eq = float(s.iloc[0])
    end_eq = float(s.iloc[-1])
    total_ret = end_eq / start_eq - 1.0 if start_eq > 0 else 0.0
    days = (s.index[-1] - s.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (end_eq / start_eq) ** (1 / years) - 1.0 if start_eq > 0 else 0.0

    daily = s.pct_change().dropna()
    if daily.std() > 0:
        sharpe = float(daily.mean() / daily.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
        volatility = float(daily.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0
        volatility = 0.0

    running_max = s.cummax()
    drawdown = (s / running_max) - 1.0
    max_dd = float(drawdown.min())
    exposure_pct = exposure_count / len(equity_series) * 100.0 if equity_series else 0.0

    return BenchmarkResult(
        symbol=symbol,
        name=name,
        start_equity=starting_cash,
        end_equity=end_eq,
        total_return_pct=total_ret * 100,
        cagr_pct=cagr * 100,
        sharpe=sharpe,
        volatility_pct=volatility * 100,
        max_drawdown_pct=max_dd * 100,
        exposure_pct=exposure_pct,
    )


def buy_and_hold(bars: list[Bar], starting_cash: float) -> BenchmarkResult:
    """Buy at the second bar's open and hold through the last close.

    Using bar 1 open matches the engine's next-bar fill contract for a strategy
    that emits a buy after observing bar 0.
    """
    symbol = bars[0].symbol if bars else ""
    if len(bars) < 2:
        return BenchmarkResult(symbol, "buy_and_hold", starting_cash, starting_cash, 0, 0, 0, 0, 0, 0)

    fill_price = bars[1].open
    qty = int(starting_cash // fill_price)
    cash = starting_cash - qty * fill_price
    equity_series = [cash + qty * bar.close for bar in bars[1:]]
    timestamps = [bar.timestamp for bar in bars[1:]]
    return _metrics_from_equity(
        symbol=symbol,
        name="buy_and_hold",
        equity_series=equity_series,
        timestamps=timestamps,
        starting_cash=starting_cash,
        exposure_count=len(equity_series) if qty > 0 else 0,
    )


def _all_in_out_baseline(
    bars: list[Bar],
    starting_cash: float,
    name: str,
    desired_long: list[bool],
) -> BenchmarkResult:
    """Simulate a deterministic long/cash baseline with next-open fills."""
    symbol = bars[0].symbol if bars else ""
    if len(bars) < 2:
        return BenchmarkResult(symbol, name, starting_cash, starting_cash, 0, 0, 0, 0, 0, 0)

    cash = starting_cash
    qty = 0
    equity_series: list[float] = []
    timestamps: list = []
    exposure_count = 0
    target_long = False

    for i, bar in enumerate(bars):
        if i > 0:
            if target_long and qty == 0:
                qty = int(cash // bar.open)
                cash -= qty * bar.open
            elif not target_long and qty > 0:
                cash += qty * bar.open
                qty = 0

        equity = cash + qty * bar.close
        equity_series.append(equity)
        timestamps.append(bar.timestamp)
        if qty > 0:
            exposure_count += 1

        target_long = desired_long[i] if i < len(desired_long) else target_long

    return _metrics_from_equity(
        symbol=symbol,
        name=name,
        equity_series=equity_series,
        timestamps=timestamps,
        starting_cash=starting_cash,
        exposure_count=exposure_count,
    )


def sma_crossover_baseline(
    bars: list[Bar],
    starting_cash: float,
    fast: int = 20,
    slow: int = 50,
) -> BenchmarkResult:
    """Simple long/cash SMA crossover baseline using only prior closes."""
    if fast >= slow:
        raise ValueError("fast must be < slow")
    closes = [bar.close for bar in bars]
    desired: list[bool] = []
    for i in range(len(closes)):
        if i + 1 < slow:
            desired.append(False)
            continue
        fast_sma = sum(closes[i + 1 - fast : i + 1]) / fast
        slow_sma = sum(closes[i + 1 - slow : i + 1]) / slow
        desired.append(fast_sma > slow_sma)
    return _all_in_out_baseline(bars, starting_cash, f"sma_{fast}_{slow}", desired)


def momentum_baseline(
    bars: list[Bar],
    starting_cash: float,
    lookback: int = 20,
) -> BenchmarkResult:
    """Simple long/cash momentum baseline: long when close > close N bars ago."""
    closes = [bar.close for bar in bars]
    desired = [False if i < lookback else closes[i] > closes[i - lookback] for i in range(len(closes))]
    return _all_in_out_baseline(bars, starting_cash, f"momentum_{lookback}", desired)


def default_baselines(bars: list[Bar], starting_cash: float) -> dict[str, BenchmarkResult]:
    """Small set of deterministic research baselines available without new deps."""
    return {
        "buy_and_hold": buy_and_hold(bars, starting_cash),
        "sma_20_50": sma_crossover_baseline(bars, starting_cash, fast=20, slow=50),
        "momentum_20": momentum_baseline(bars, starting_cash, lookback=20),
    }



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
