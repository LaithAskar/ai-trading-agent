from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import yfinance as yf

from ..config import CACHE_DIR
from ..core.events import Bar


def _cache_path(symbol: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{start}_{end}_1d.pkl"


def load_bars(
    symbol: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(symbol, start, end)
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol} {start} to {end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.to_pickle(cache)
    return df


def iter_bars(symbol: str, df: pd.DataFrame) -> Iterator[Bar]:
    for ts, row in df.iterrows():
        yield Bar(
            symbol=symbol,
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.fromisoformat(str(ts)),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
