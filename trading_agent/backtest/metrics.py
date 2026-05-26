from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.portfolio import Portfolio


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Metrics:
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    num_fills: int
    num_round_trips: int
    win_rate_pct: float

    def as_table(self) -> list[tuple[str, str]]:
        return [
            ("Starting equity", f"${self.starting_equity:,.2f}"),
            ("Ending equity", f"${self.ending_equity:,.2f}"),
            ("Total return", f"{self.total_return_pct:.2f}%"),
            ("CAGR", f"{self.cagr_pct:.2f}%"),
            ("Sharpe (annualized)", f"{self.sharpe:.2f}"),
            ("Max drawdown", f"{self.max_drawdown_pct:.2f}%"),
            ("Fills", str(self.num_fills)),
            ("Round trips", str(self.num_round_trips)),
            ("Win rate", f"{self.win_rate_pct:.2f}%"),
        ]


def equity_curve_df(portfolio: Portfolio) -> pd.DataFrame:
    if not portfolio.equity_curve:
        return pd.DataFrame(columns=["equity"])
    df = pd.DataFrame(portfolio.equity_curve, columns=["timestamp", "equity"])
    df = df.set_index("timestamp")
    return df


def _round_trip_pnls(portfolio: Portfolio) -> list[float]:
    """Pair BUYs with subsequent SELLs (FIFO) and return PnL per closed round trip.

    Assumes single-direction (long-only) trading. Short-handling can be added later.
    """
    lots_by_symbol: dict[str, list[tuple[float, float]]] = {}
    pnls: list[float] = []
    for fill in portfolio.fills:
        lots = lots_by_symbol.setdefault(fill.symbol, [])
        if fill.side.value == "BUY":
            lots.append((fill.quantity, fill.price))
        else:
            remaining = fill.quantity
            while remaining > 1e-9 and lots:
                lot_qty, lot_price = lots[0]
                take = min(lot_qty, remaining)
                pnls.append((fill.price - lot_price) * take)
                if take >= lot_qty - 1e-9:
                    lots.pop(0)
                else:
                    lots[0] = (lot_qty - take, lot_price)
                remaining -= take
    return pnls


def compute_metrics(portfolio: Portfolio) -> Metrics:
    eq = equity_curve_df(portfolio)
    if eq.empty or len(eq) < 2:
        return Metrics(
            starting_equity=portfolio.starting_cash,
            ending_equity=portfolio.starting_cash,
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            max_drawdown_pct=0.0,
            num_fills=len(portfolio.fills),
            num_round_trips=0,
            win_rate_pct=0.0,
        )

    starting = float(eq["equity"].iloc[0])
    ending = float(eq["equity"].iloc[-1])
    total_return = (ending / starting) - 1.0

    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (ending / starting) ** (1 / years) - 1.0 if starting > 0 else 0.0

    daily_ret = eq["equity"].pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0

    running_max = eq["equity"].cummax()
    drawdown = (eq["equity"] / running_max) - 1.0
    max_dd = float(drawdown.min())

    pnls = _round_trip_pnls(portfolio)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = (wins / len(pnls) * 100) if pnls else 0.0

    return Metrics(
        starting_equity=starting,
        ending_equity=ending,
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        sharpe=sharpe,
        max_drawdown_pct=max_dd * 100,
        num_fills=len(portfolio.fills),
        num_round_trips=len(pnls),
        win_rate_pct=win_rate,
    )
