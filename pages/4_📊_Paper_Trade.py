"""Alpaca paper-trading control panel.

Live trading is *physically impossible* from this codebase. The broker class
refuses to construct in live mode. The "Submit" button below talks to Alpaca's
paper API only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_shared import require_alpaca_creds, setup_page
from trading_agent.config import PROJECT_ROOT


setup_page("Paper Trade", icon="📊")

st.title("📊 Paper Trade")
st.caption(
    "Sandboxed against Alpaca's paper trading API. "
    "The broker class raises `NotImplementedError` if you try to enable live mode."
)

st.warning(
    "**Paper trading only.** This entire codebase is hardcoded against live trading "
    "by design. Before considering live, run paper for 3+ months, measure the "
    "realized-vs-backtest drift, and cap any future live position at 1-2% of capital."
)

api_key, api_secret = require_alpaca_creds()

from trading_agent.broker.alpaca import AlpacaPaperBroker

try:
    broker = AlpacaPaperBroker(api_key, api_secret)
except Exception as e:
    st.error(f"Failed to connect to Alpaca paper: {type(e).__name__}: {e}")
    st.stop()

# Account snapshot
acct = broker.account()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cash", f"${acct.cash:,.2f}")
c2.metric("Portfolio value", f"${acct.portfolio_value:,.2f}")
c3.metric("Buying power", f"${acct.buying_power:,.2f}")
c4.metric("Mode", "Paper ✓" if acct.is_paper else "LIVE 🚫")

st.divider()

# Positions
st.markdown("### Open positions")
positions = broker.positions()
if positions:
    pos_df = pd.DataFrame(
        [
            {
                "Symbol": p.symbol,
                "Quantity": p.quantity,
                "Avg entry": f"${p.avg_entry_price:.2f}",
                "Market value": f"${p.market_value:,.2f}",
                "Unrealized P/L": f"${p.unrealized_pl:,.2f}",
            }
            for p in positions
        ]
    )
    st.dataframe(pos_df, use_container_width=True, hide_index=True)
else:
    st.caption("No open positions.")

# Open orders
st.markdown("### Open orders")
open_orders = broker.open_orders()
if open_orders:
    ord_df = pd.DataFrame(
        [
            {
                "Order ID": o.order_id[:12] + "...",
                "Symbol": o.symbol,
                "Side": o.side,
                "Quantity": o.quantity,
                "Status": o.status,
                "Submitted": o.submitted_at,
            }
            for o in open_orders
        ]
    )
    st.dataframe(ord_df, use_container_width=True, hide_index=True)
else:
    st.caption("No open orders.")

st.divider()

# Strategy tick
st.markdown("### Run a strategy tick")
st.caption(
    "Replays a strategy across the last N days of bars to rebuild its internal "
    "state, then asks for orders on today's bar. Default is dry-run."
)


def _list_strategies() -> list[str]:
    sd = PROJECT_ROOT / "strategies"
    return sorted(
        p.stem for p in sd.glob("*.py")
        if not p.name.startswith("_") and p.stem != "__init__"
    )


with st.form("paper_tick_form"):
    strategy = st.selectbox("Strategy", _list_strategies())
    symbol = st.text_input("Ticker", value="AAPL").upper()
    lookback_days = st.number_input("Lookback days", value=365, min_value=30, max_value=3650)
    params_text = st.text_input(
        "Strategy params (optional)",
        placeholder="key=value,key=value  e.g.  fast=20,slow=50",
    )
    submit_dry = st.form_submit_button("▶ Dry run (no orders submitted)", type="primary", use_container_width=True)


def _parse_params(s: str) -> dict:
    out = {}
    for item in s.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


if submit_dry:
    from trading_agent.broker.paper_runner import paper_tick

    with st.spinner(f"Replaying {strategy} on {symbol}..."):
        result = paper_tick(
            strategy_name=strategy,
            symbol=symbol,
            params=_parse_params(params_text),
            lookback_days=int(lookback_days),
            broker=broker,
            dry_run=True,
        )

    st.success(f"Replayed {result.bars_seen} bars.")
    if result.proposed_orders:
        st.markdown("#### Proposed orders (NOT submitted)")
        prop_df = pd.DataFrame(
            [{"Symbol": o.symbol, "Side": o.side.value, "Quantity": o.quantity} for o in result.proposed_orders]
        )
        st.dataframe(prop_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Submit these to Alpaca paper?")
        st.caption("This will actually place market orders on your paper account.")
        if st.checkbox("I understand these will be submitted to my paper account."):
            if st.button("🟢 Submit now", type="primary"):
                with st.spinner("Submitting..."):
                    live_result = paper_tick(
                        strategy_name=strategy,
                        symbol=symbol,
                        params=_parse_params(params_text),
                        lookback_days=int(lookback_days),
                        broker=broker,
                        dry_run=False,
                    )
                if live_result.submitted:
                    st.success(f"Submitted {len(live_result.submitted)} order(s).")
                    sub_df = pd.DataFrame(
                        [{"Order ID": o.order_id[:12] + "...", "Symbol": o.symbol, "Status": o.status}
                         for o in live_result.submitted]
                    )
                    st.dataframe(sub_df, use_container_width=True, hide_index=True)
                elif live_result.skipped_reason:
                    st.warning(live_result.skipped_reason)
    else:
        st.caption(result.skipped_reason or "Strategy emitted no orders on the most recent bar.")
