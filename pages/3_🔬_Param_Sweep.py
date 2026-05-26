"""Grid search a strategy's parameters with optional walk-forward validation."""
from __future__ import annotations

import statistics
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from app_shared import setup_page
from trading_agent.config import PROJECT_ROOT


setup_page("Param Sweep", icon="🔬")

st.title("🔬 Parameter Sweep")
st.caption(
    "Cartesian-product grid search over a strategy's parameters. "
    "Optionally validate each combo with walk-forward to rank by *out-of-sample* "
    "stability instead of in-sample fit."
)


def _list_strategies() -> list[str]:
    sd = PROJECT_ROOT / "strategies"
    if not sd.exists():
        return []
    return sorted(
        p.stem for p in sd.glob("*.py")
        if not p.name.startswith("_") and p.stem != "__init__"
    )


with st.form("sweep_form"):
    c1, c2 = st.columns([2, 2])
    with c1:
        strategy = st.selectbox("Strategy", _list_strategies())
        symbol = st.text_input("Ticker", value="AAPL").upper()
    with c2:
        start = st.date_input("Start", value=date(2020, 1, 1))
        end = st.date_input("End", value=date(2024, 12, 31))

    grid_text = st.text_area(
        "Grid (one axis per line — key=v1,v2,v3)",
        height=110,
        value="fast=10,20,50\nslow=50,100,200",
        help="Each line is one parameter and its values to try. All combos are run.",
    )

    walk_forward = st.checkbox(
        "Walk-forward validate each combo",
        help="Runs each combo through rolling train/test splits. Slower, more honest.",
    )
    train_years = st.number_input("train_years (walk-forward only)", value=2.0, min_value=0.5, max_value=10.0, step=0.5)
    test_years = st.number_input("test_years (walk-forward only)", value=1.0, min_value=0.25, max_value=5.0, step=0.25)
    stride_years = st.number_input("stride_years (walk-forward only)", value=1.0, min_value=0.25, max_value=5.0, step=0.25)

    submit = st.form_submit_button("▶ Run sweep", type="primary", use_container_width=True)

if not submit:
    st.info("Configure the form and hit **Run sweep**. The default grid is a 3×3 fast/slow combo for sma_cross.")
    st.stop()


def _expand(grid_text: str) -> list[dict]:
    """Parse multi-line 'key=v1,v2' grid input into cartesian product of dicts."""
    axes: list[tuple[str, list]] = []
    for line in grid_text.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, raw = line.split("=", 1)
        vs = []
        for v in raw.split(","):
            v = v.strip()
            try:
                vs.append(int(v))
            except ValueError:
                try:
                    vs.append(float(v))
                except ValueError:
                    vs.append(v)
        axes.append((k.strip(), vs))
    combos = [{}]
    for k, vs in axes:
        combos = [{**c, k: v} for c in combos for v in vs]
    return combos


combos = _expand(grid_text)
st.info(f"Running {len(combos)} combination(s)" + (" through walk-forward splits" if walk_forward else ""))

progress = st.progress(0.0)
status = st.empty()
rows = []

if walk_forward:
    from trading_agent.backtest.rigor import walk_forward_splits
    from trading_agent.backtest.runner import run_backtest

    splits = walk_forward_splits(str(start), str(end), train_years, test_years, stride_years)
    if not splits:
        st.error(f"No valid walk-forward splits between {start} and {end} with these settings.")
        st.stop()

    for i, combo in enumerate(combos):
        cagrs, sharpes, pvalues = [], [], []
        for s in splits:
            try:
                r = run_backtest(
                    strategy_name=strategy, symbol=symbol,
                    start=s.test_start, end=s.test_end,
                    params=combo, write_artifacts=False,
                )
                cagrs.append(r.metrics.cagr_pct)
                sharpes.append(r.metrics.sharpe)
                pvalues.append(r.sharpe_p_value)
            except Exception:
                continue
        if not cagrs:
            continue
        rows.append({
            "Params": ", ".join(f"{k}={v}" for k, v in combo.items()),
            "OOS CAGR mean": statistics.mean(cagrs),
            "OOS CAGR stdev": statistics.stdev(cagrs) if len(cagrs) > 1 else 0.0,
            "OOS Sharpe mean": statistics.mean(sharpes),
            "Splits": len(cagrs),
        })
        progress.progress((i + 1) / len(combos))
        status.text(f"{i + 1}/{len(combos)} combos · {combo}")
    progress.empty(); status.empty()
    if not rows:
        st.warning("No combos produced results.")
        st.stop()
    df = pd.DataFrame(rows).sort_values("OOS Sharpe mean", ascending=False).reset_index(drop=True)
else:
    from trading_agent.backtest.runner import run_backtest
    for i, combo in enumerate(combos):
        try:
            r = run_backtest(
                strategy_name=strategy, symbol=symbol,
                start=str(start), end=str(end),
                params=combo, write_artifacts=False,
            )
            rows.append({
                "Params": ", ".join(f"{k}={v}" for k, v in combo.items()),
                "CAGR": r.metrics.cagr_pct,
                "Sharpe": r.metrics.sharpe,
                "Max DD": r.metrics.max_drawdown_pct,
                "B&H CAGR": r.benchmark.cagr_pct,
                "p-value": r.sharpe_p_value,
                "Round trips": r.metrics.num_round_trips,
            })
        except Exception as e:
            status.warning(f"{combo}: {e}")
        progress.progress((i + 1) / len(combos))
    progress.empty(); status.empty()
    if not rows:
        st.warning("No combos produced results.")
        st.stop()
    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

st.dataframe(
    df.style.format({
        c: "{:.2f}" for c in df.select_dtypes(include="float").columns
    }),
    use_container_width=True,
    hide_index=True,
)

if not walk_forward and "B&H CAGR" in df.columns:
    bh = df["B&H CAGR"].iloc[0]
    beats = df[df["CAGR"] > bh]
    st.caption(
        f"**Buy-and-hold benchmark for this window: {bh:.2f}% CAGR.** "
        f"{len(beats)} of {len(df)} combos beat it. "
        f"({len(df) - len(beats)} did not.)"
    )
