"""Shared Streamlit helpers used by every page.

Keeps sidebar, key-handling, and CSS consistent across the multi-page app.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from trading_agent.config import Config
from trading_agent.llm.client import build_auth_url, exchange_code_for_key


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
    if "openrouter_key" not in st.session_state:
        st.session_state.openrouter_key = cfg.openrouter_api_key or ""
    if "provider" not in st.session_state:
        st.session_state.provider = "openrouter" if st.session_state.openrouter_key else "anthropic"
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = cfg.anthropic_api_key or ""
    if "alpaca_api_key" not in st.session_state:
        st.session_state.alpaca_api_key = cfg.alpaca_api_key or ""
    if "alpaca_api_secret" not in st.session_state:
        st.session_state.alpaca_api_secret = cfg.alpaca_api_secret or ""

    _handle_oauth_callback()

    with st.sidebar:
        st.markdown("### ai-trading-agent")
        st.caption("Backtest + AI research agent for stock strategies")
        st.markdown(
            "[Repo](https://github.com/LaithAskar/ai-trading-agent) · "
            "[Project page](https://laithaskar.github.io/ai-trading-agent/)"
        )
        st.divider()

        st.markdown("#### Sign in")
        if st.session_state.openrouter_key:
            st.success("Signed in with OpenRouter")
            if st.button("Sign out", use_container_width=True):
                st.session_state.openrouter_key = ""
                st.session_state.provider = "anthropic"
                st.rerun()
        else:
            st.link_button(
                "Sign in with OpenRouter",
                build_auth_url(cfg.openrouter_callback_url),
                type="primary",
                use_container_width=True,
            )
            st.caption(
                "Authorize once on OpenRouter — nothing to paste. The agent runs on "
                "your OpenRouter credits."
            )
            if st.session_state.get("oauth_error"):
                st.error(st.session_state.oauth_error)

        with st.expander("Advanced: use your own Anthropic key"):
            st.session_state.anthropic_api_key = st.text_input(
                "ANTHROPIC_API_KEY",
                value=st.session_state.anthropic_api_key,
                type="password",
                placeholder="sk-ant-...",
                help="Fallback if you'd rather call Anthropic directly instead of via OpenRouter.",
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
        st.caption("Keys live in this browser session only. Never written to disk.")
        st.caption("Direct Anthropic SDK integration. No agent frameworks.")


def _handle_oauth_callback() -> None:
    """If we just returned from OpenRouter with ?code=, exchange it for a key.

    Bare authorization-code flow: the one-time code arrives as a query param; we
    swap it for a user-controlled OpenRouter key, store it in session state only,
    then strip the code from the URL so a refresh can't replay it.
    """
    code = st.query_params.get("code")
    if not code or st.session_state.get("openrouter_key"):
        return
    try:
        st.session_state.openrouter_key = exchange_code_for_key(code)
        st.session_state.provider = "openrouter"
        st.session_state.pop("oauth_error", None)
    except Exception as e:  # surface, don't crash the page
        st.session_state.oauth_error = f"OpenRouter sign-in failed: {type(e).__name__}: {e}"
    finally:
        st.query_params.clear()
        st.rerun()


def require_llm() -> tuple[str, str]:
    """Return (api_key, provider), preferring OpenRouter sign-in over a pasted key.

    Stops the page with a prompt if neither is available.
    """
    or_key = st.session_state.get("openrouter_key") or ""
    if or_key:
        return or_key, "openrouter"
    anth = st.session_state.get("anthropic_api_key") or ""
    if anth:
        return anth, "anthropic"
    st.warning(
        "Sign in with OpenRouter in the left sidebar before running "
        "(or expand “Advanced” to paste an Anthropic key)."
    )
    st.stop()


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
