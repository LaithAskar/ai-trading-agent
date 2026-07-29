"""Streamlit Cloud entry point. Multi-page app — see pages/."""
from __future__ import annotations

import streamlit as st

from app_shared import setup_page


setup_page("Home", icon="📈")

st.title("📈 ai-trading-agent")
st.markdown(
    """
    A research-first backtest + AI agent for stock-strategy experiments.
    Hand-written ReAct loop on the Anthropic SDK — no LangChain, no agent
    frameworks.
    """
)

st.info(
    "Recruiter/demo framing: this app is for research, backtests, and paper-only "
    "experiments. It is not investment advice or evidence of market-beating "
    "performance; every backtest should be read against its buy-and-hold baseline."
)

st.markdown("### What you can do here")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 🤖 Run Agent")
    st.write(
        "Give the agent a natural-language goal — backtest a strategy, "
        "compare two strategies, analyze a 10-Q — and watch its ReAct loop "
        "unfold panel by panel."
    )
with c2:
    st.markdown("#### 🧪 Backtest")
    st.write(
        "Run any of the reference strategies on any US ticker over any "
        "window. Side-by-side buy-and-hold comparison, Sharpe p-value, "
        "and equity curve plot."
    )
with c3:
    st.markdown("#### 📋 Browse")
    st.write(
        "Browse past agent sessions and the strategy gallery. "
        "Drill into a session to see exactly what the agent did."
    )

st.divider()

st.markdown("### Quick start")
st.markdown(
    """
    **No-key demo path:** open **Backtest** and run `sma_cross` on `AAPL`,
    `2020-01-01` → `2024-12-31`, params `fast=20,slow=50`. This shows the
    core artifact: strategy metrics, buy-and-hold comparison, p-value, and trade log.

    **Agent path:**
    1. Click **Sign in with OpenRouter** in the sidebar and authorize — no raw API
       key to paste. (Prefer Anthropic directly? Use the **Advanced** expander.)
    2. Open the **Run Agent** page from the sidebar on the left.
    3. Type a research goal, e.g.:
       > "Compare sma_cross and rsi_mean_rev on AAPL for 2022-2024 and tell me which performed better."
    4. Hit **Run** and watch.
    """
)

st.divider()

with st.container():
    st.markdown("### Notes")
    st.markdown(
        """
        - **Sign in with OpenRouter** so you never type a raw secret. The minted
          key is scoped to your own OpenRouter account and lives in this browser
          session only — it's not written to any file on the server. (It is held
          in server memory for the duration of a run; this is better key hygiene,
          not zero server-side custody.)
        - **Paper trading** requires Alpaca paper credentials (separate sidebar
          expander) and is sandboxed against the live API by design.
        - On the hosted Streamlit Cloud deployment, the file system is ephemeral —
          past sessions and backtest artifacts persist only locally. Run locally
          with `streamlit run streamlit_app.py` for full data persistence.
        """
    )

st.caption(
    "Source: [github.com/LaithAskar/ai-trading-agent](https://github.com/LaithAskar/ai-trading-agent)"
)
