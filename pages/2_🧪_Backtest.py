"""Run a single backtest with full visualization."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from app_shared import setup_page
from trading_agent.config import PROJECT_ROOT


setup_page("Backtest", icon="🧪")

st.title("🧪 Backtest")
st.caption(
    "Pick a strategy + symbol + window and run it. Every result shows the "
    "buy-and-hold benchmark side by side and the Sharpe significance test."
)


def _list_strategies() -> list[str]:
    sd = PROJECT_ROOT / "strategies"
    if not sd.exists():
        return []
    return sorted(
        p.stem for p in sd.glob("*.py")
        if not p.name.startswith("_") and p.stem != "__init__"
    )


with st.form("backtest_form"):
    c1, c2 = st.columns([2, 2])
    with c1:
        strategy = st.selectbox("Strategy", _list_strategies())
        symbol = st.text_input("Ticker", value="AAPL").upper()
    with c2:
        start = st.date_input("Start", value=date(2022, 1, 1))
        end = st.date_input("End", value=date(2024, 12, 31))

    c3, c4, c5 = st.columns(3)
    with c3:
        cash = st.number_input("Starting cash ($)", value=100_000.0, step=10_000.0, min_value=1000.0)
    with c4:
        slippage_bps = st.number_input("Slippage (bps)", value=5.0, min_value=0.0, max_value=50.0, step=0.5)
    with c5:
        commission = st.number_input("Commission / trade ($)", value=0.0, min_value=0.0, step=0.5)

    params_text = st.text_input(
        "Strategy params (optional)",
        placeholder="key=value,key=value  e.g.  fast=20,slow=50",
        help="Comma-separated. Integer/float/string auto-detected.",
    )
    submit = st.form_submit_button("▶ Run backtest", type="primary", use_container_width=True)

if not submit:
    st.info("Configure the form and hit **Run backtest**.")
    st.stop()


def _parse_params(s: str) -> dict:
    out = {}
    if not s.strip():
        return out
    for item in s.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


params = _parse_params(params_text)

with st.spinner(f"Running {strategy} on {symbol} {start} → {end}..."):
    try:
        from trading_agent.backtest.runner import run_backtest
        run = run_backtest(
            strategy_name=strategy,
            symbol=symbol,
            start=str(start),
            end=str(end),
            params=params,
            starting_cash=float(cash),
            slippage_bps=float(slippage_bps),
            commission_per_trade=float(commission),
            write_artifacts=False,
        )
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

st.success(f"Done · {run.metrics.num_fills} fills · {run.metrics.num_round_trips} round trips")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Strategy")
    st.metric("Total return", f"{run.metrics.total_return_pct:.2f}%")
    st.metric("CAGR", f"{run.metrics.cagr_pct:.2f}%")
    st.metric("Sharpe", f"{run.metrics.sharpe:.2f}")
    st.metric("Max drawdown", f"{run.metrics.max_drawdown_pct:.2f}%")
    st.metric("Win rate", f"{run.metrics.win_rate_pct:.2f}%")
with c2:
    st.markdown("### Buy & Hold")
    st.metric("Total return", f"{run.benchmark.total_return_pct:.2f}%")
    st.metric("CAGR", f"{run.benchmark.cagr_pct:.2f}%")
    st.metric("Sharpe", f"{run.benchmark.sharpe:.2f}")
    st.metric("Max drawdown", f"{run.benchmark.max_drawdown_pct:.2f}%")
    st.caption("(passive baseline for the same window)")

st.markdown("### Sharpe significance")
sc1, sc2 = st.columns(2)
with sc1:
    st.metric("t-statistic", f"{run.sharpe_t_stat:.2f}")
with sc2:
    st.metric("p-value", f"{run.sharpe_p_value:.4f}")
if run.sharpe_p_value >= 0.10:
    st.warning(
        f"⚠️ Sharpe is not statistically distinguishable from zero "
        f"(p={run.sharpe_p_value:.3f}). Don't over-interpret this result."
    )

# Equity curve — reconstruct from the metrics we have
st.markdown("### Equity curve")
from trading_agent.backtest.metrics import equity_curve_df
# We didn't keep the portfolio object from runner.run_backtest in this flow;
# the equity curve isn't in the return dataclass. Re-run quickly to chart it.
st.caption("Plot uses the backtest's equity series. Strategy line vs starting cash.")

# Pull the portfolio's equity_curve by re-running and grabbing the result.
# This is intentionally cheap — same data already loaded from yfinance cache.
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.backtest.runner import load_strategy
from trading_agent.data.yfinance_source import iter_bars, load_bars

strat = load_strategy(strategy, params)
df = load_bars(symbol, str(start), str(end))
engine = BacktestEngine(
    starting_cash=float(cash),
    slippage_bps=float(slippage_bps),
    commission_per_trade=float(commission),
)
result = engine.run(strat, symbol, iter_bars(symbol, df))
eq = equity_curve_df(result.portfolio)
if not eq.empty:
    eq_renamed = eq.rename(columns={"equity": f"{strategy} equity"})
    st.line_chart(eq_renamed)
else:
    st.info("No equity data to plot (strategy emitted no signals).")

if result.portfolio.fills:
    st.markdown("### Trade log")
    trades_df = pd.DataFrame(
        [
            {
                "Timestamp": f.timestamp,
                "Symbol": f.symbol,
                "Side": f.side.value,
                "Quantity": f.quantity,
                "Price": f.price,
            }
            for f in result.portfolio.fills
        ]
    )
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
