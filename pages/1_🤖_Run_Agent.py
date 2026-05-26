"""Run the AI agent on a natural-language goal, with live-streamed iterations."""
from __future__ import annotations

import json

import streamlit as st

from app_shared import require_anthropic_key, setup_page


setup_page("Run Agent", icon="🤖")

st.title("🤖 Run Agent")
st.caption(
    "Type a goal. The agent runs a ReAct loop with eight tools and streams "
    "its Thought → Action → Observation panels back to you as it works."
)

api_key = require_anthropic_key()

with st.form("agent_form"):
    goal = st.text_area(
        "Goal",
        height=120,
        placeholder=(
            "e.g. Compare sma_cross and rsi_mean_rev on AAPL for 2022-01-01 to "
            "2024-12-31. Use search_memory first. End with a single best pick "
            "and one honest caveat."
        ),
        help="Plain English. The agent has 8 tools; let it decide which to call.",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        model = st.selectbox(
            "Model",
            ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
            index=0,
        )
    with c2:
        max_iters = st.number_input("Max iterations", min_value=1, max_value=50, value=15)
    with c3:
        max_dollars = st.number_input("Max $ spend", min_value=0.10, max_value=10.0, value=0.50, step=0.10)
    submit = st.form_submit_button("▶ Run agent", type="primary", use_container_width=True)

if not submit:
    st.info(
        "**Try these goals:** ",
    )
    examples = [
        "List the strategies available and run sma_cross on AAPL for 2023.",
        "Compare sma_cross and rsi_mean_rev on AAPL 2022-2024 and tell me which performed better risk-adjusted.",
        "Look at Apple's most recent 10-Q and tell me whether management's tone has improved or worsened versus the prior quarter.",
        "Backtest filings_sentiment on MSFT for the last 3 years and tell me if its Sharpe is statistically distinguishable from zero.",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["_prefill_goal"] = ex
            st.rerun()
    if "_prefill_goal" in st.session_state:
        st.text("(prefilled — paste into the box above and hit Run)")
    st.stop()

if not goal.strip():
    st.error("Type a goal first.")
    st.stop()

# Live streaming: use a placeholder container that grows per iteration.
stream_container = st.container()
status = st.empty()
status.info(f"Starting agent with model={model}, max_iters={max_iters}, max_dollars=${max_dollars:.2f}...")


def _panel_html(kind: str, label: str, body: str, duration_ms=None) -> str:
    chip = ""
    if duration_ms is not None:
        chip = f"<span style='background:#ececef;color:#6e6e73;font-size:11px;padding:1px 7px;border-radius:999px;margin-left:8px;font-weight:500'>{duration_ms} ms</span>"
    import html as _html
    safe_body = _html.escape(body)
    return (
        f'<div class="tagging-panel {kind}">'
        f'<div class="lbl">{label}{chip}</div>'
        f'<div style="white-space:pre-wrap; word-break:break-word;">{safe_body}</div>'
        f"</div>"
    )


def render_entry(entry, container):
    """Render a single TranscriptEntry into the given Streamlit container."""
    with container:
        container.markdown(
            f'<div class="iter-divider">Iteration {entry.iteration}</div>',
            unsafe_allow_html=True,
        )
        if entry.thought:
            container.markdown(_panel_html("thought", "Thought", entry.thought), unsafe_allow_html=True)

        if entry.is_final:
            container.markdown(_panel_html("final", "Final Answer", entry.thought or ""), unsafe_allow_html=True)
            return

        if entry.tool_name:
            tool_input_text = json.dumps(entry.tool_input or {}, indent=2, default=str)
            container.markdown(
                _panel_html("action", f"Action — {entry.tool_name}", tool_input_text),
                unsafe_allow_html=True,
            )
            # Observation
            kind = "obs-err" if entry.is_error else "obs-ok"
            label = "Observation (error)" if entry.is_error else "Observation"
            try:
                pretty = json.dumps(json.loads(entry.tool_result or ""), indent=2, default=str)
            except (json.JSONDecodeError, TypeError):
                pretty = entry.tool_result or ""
            if len(pretty) > 4000:
                pretty = pretty[:4000] + "\n... (truncated)"
            container.markdown(
                _panel_html(kind, label, pretty, duration_ms=entry.tool_duration_ms),
                unsafe_allow_html=True,
            )


# Reserve a slot for each future iteration so the layout grows visibly.
slots: list = []
for i in range(max_iters):
    slots.append(stream_container.empty())

call_count = {"n": 0}

def on_iteration_cb(entry):
    if call_count["n"] >= len(slots):
        # Defensive: more iterations than reserved slots — append a new one.
        slots.append(stream_container.empty())
    render_entry(entry, slots[call_count["n"]])
    call_count["n"] += 1


from trading_agent.agent.loop import run_agent

try:
    session = run_agent(
        goal=goal,
        mode="auto",
        model=model,
        max_iters=int(max_iters),
        max_session_dollars=float(max_dollars),
        api_key=api_key,
        on_iteration=on_iteration_cb,
    )
except Exception as e:
    status.error(f"{type(e).__name__}: {e}")
    st.stop()

if session.finished:
    status.success(
        f"Finished in {len(session.transcript)} iter · "
        f"{session.input_tokens:,} in / {session.output_tokens:,} out · "
        f"≈ ${session.cost_dollars:.4f}"
    )
else:
    status.warning(
        f"Stopped: {session.stopped_by or 'unknown'} · "
        f"≈ ${session.cost_dollars:.4f}"
    )

st.caption(f"Session ID: `{session.session_id}` · log: `data/logs/agent_runs/{session.session_id}.json`")
