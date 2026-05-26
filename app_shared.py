"""Shared Streamlit helpers used by every page.

Keeps sidebar, key-handling, and CSS consistent across the multi-page app.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from trading_agent.config import Config


CSS = """
<style>
:root { --thought:#0a84ff; --action-text:#0a7ea4; --obs-ok-text:#248a3d; --obs-err-text:#c62a1f; --final-text:#1f7a30; }
.tagging-panel {
    background: #fff;
    border-left: 4px solid #d2d2d7;
    border-top: 1px solid #d2d2d7;
    border-right: 1px solid #d2d2d7;
    border-bottom: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.tagging-panel .lbl {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    margin-bottom: 8px;
    color: #6e6e73;
}
.tagging-panel.thought  { border-left-color: var(--thought); }
.tagging-panel.thought  .lbl { color: var(--thought); }
.tagging-panel.action   { border-left-color: #5ac8fa; }
.tagging-panel.action   .lbl { color: var(--action-text); }
.tagging-panel.obs-ok   { border-left-color: #34c759; }
.tagging-panel.obs-ok   .lbl { color: var(--obs-ok-text); }
.tagging-panel.obs-err  { border-left-color: #ff3b30; background: #fff5f4; }
.tagging-panel.obs-err  .lbl { color: var(--obs-err-text); }
.tagging-panel.final    { border-left-color: #34c759; background: #f3fdf4; }
.tagging-panel.final    .lbl { color: var(--final-text); }
.iter-divider {
    font-size: 11px;
    color: #6e6e73;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin: 24px 0 8px 4px;
}
</style>
"""


def setup_page(title: str, icon: str = "📈") -> None:
    """Run at the top of every page: theme, title, sidebar key input."""
    st.set_page_config(
        page_title=f"{title} · ai-trading-agent",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    cfg = Config.load()
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = cfg.anthropic_api_key or ""
    if "alpaca_api_key" not in st.session_state:
        st.session_state.alpaca_api_key = cfg.alpaca_api_key or ""
    if "alpaca_api_secret" not in st.session_state:
        st.session_state.alpaca_api_secret = cfg.alpaca_api_secret or ""

    with st.sidebar:
        st.markdown("### ai-trading-agent")
        st.caption("Backtest + AI research agent for stock strategies")
        st.markdown(
            "[Repo](https://github.com/LaithAskar/ai-trading-agent) · "
            "[Project page](https://laithaskar.github.io/ai-trading-agent/)"
        )
        st.divider()

        st.markdown("#### API keys")
        st.caption("Stored in this browser session only. Never persisted.")
        st.session_state.anthropic_api_key = st.text_input(
            "ANTHROPIC_API_KEY",
            value=st.session_state.anthropic_api_key,
            type="password",
            placeholder="sk-ant-...",
            help="Required for any agent or LLM-strategy action.",
        )
        with st.expander("Optional: Alpaca paper credentials"):
            st.session_state.alpaca_api_key = st.text_input(
                "ALPACA_API_KEY",
                value=st.session_state.alpaca_api_key,
                type="password",
                help="Only required for the Paper Trade tab.",
            )
            st.session_state.alpaca_api_secret = st.text_input(
                "ALPACA_API_SECRET",
                value=st.session_state.alpaca_api_secret,
                type="password",
            )

        st.divider()
        st.caption("Powered by Claude via the Anthropic SDK. No frameworks.")


def require_anthropic_key() -> str:
    key = st.session_state.get("anthropic_api_key") or ""
    if not key:
        st.warning(
            "Set ANTHROPIC_API_KEY in the left sidebar before running. "
            "Get one at console.anthropic.com."
        )
        st.stop()
    return key


def require_alpaca_creds() -> tuple[str, str]:
    k = st.session_state.get("alpaca_api_key") or ""
    s = st.session_state.get("alpaca_api_secret") or ""
    if not (k and s):
        st.warning(
            "Set ALPACA_API_KEY and ALPACA_API_SECRET in the left sidebar before running. "
            "Get paper credentials at app.alpaca.markets/paper/dashboard/overview."
        )
        st.stop()
    return k, s
