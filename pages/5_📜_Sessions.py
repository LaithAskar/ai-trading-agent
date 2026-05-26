"""Browse past agent sessions."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app_shared import setup_page
from trading_agent.config import AGENT_LOGS_DIR


setup_page("Sessions", icon="📜")

st.title("📜 Past Agent Sessions")
st.caption("Sessions persist locally under `data/logs/agent_runs/`. On Streamlit Cloud the file system is ephemeral.")

sessions = sorted(AGENT_LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if not sessions:
    st.info(
        "No sessions recorded yet on this server. Run the agent on the **Run Agent** page first, "
        "or run the app locally to access your full session history."
    )
    st.stop()

# Aggregate stats
rows = []
for path in sessions:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    rows.append({
        "session_id": data.get("session_id", path.stem),
        "started_at": data.get("started_at", "?"),
        "model": data.get("model", "?"),
        "mode": data.get("mode", "?"),
        "iterations": len(data.get("transcript", [])),
        "input_tokens": data.get("input_tokens", 0),
        "output_tokens": data.get("output_tokens", 0),
        "cost_dollars": data.get("cost_dollars", 0.0),
        "finished": data.get("finished", False),
        "goal_short": (data.get("goal", "") or "")[:80] + ("..." if len(data.get("goal", "")) > 80 else ""),
    })

df = pd.DataFrame(rows)
st.markdown("### Overview")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

session_id = st.selectbox(
    "Open a session",
    [r["session_id"] for r in rows],
    format_func=lambda sid: f"{sid} — {next((r['goal_short'] for r in rows if r['session_id'] == sid), '?')}",
)

if not session_id:
    st.stop()

session_path = AGENT_LOGS_DIR / f"{session_id}.json"
data = json.loads(session_path.read_text(encoding="utf-8"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Iterations", len(data.get("transcript", [])))
c2.metric("Cost", f"${data.get('cost_dollars', 0):.4f}")
c3.metric("Input tokens", f"{data.get('input_tokens', 0):,}")
c4.metric("Output tokens", f"{data.get('output_tokens', 0):,}")

st.markdown("**Goal**")
st.code(data.get("goal", ""), language="markdown")

import html as _html


def _panel_html(kind: str, label: str, body: str, duration_ms=None) -> str:
    chip = ""
    if duration_ms is not None:
        chip = f"<span style='background:#ececef;color:#6e6e73;font-size:11px;padding:1px 7px;border-radius:999px;margin-left:8px;font-weight:500'>{duration_ms} ms</span>"
    safe_body = _html.escape(body)
    return (
        f'<div class="tagging-panel {kind}">'
        f'<div class="lbl">{label}{chip}</div>'
        f'<div style="white-space:pre-wrap; word-break:break-word;">{safe_body}</div>'
        f"</div>"
    )


st.markdown("### Transcript")
for entry in data.get("transcript", []):
    st.markdown(
        f'<div class="iter-divider">Iteration {entry.get("iteration", "?")}</div>',
        unsafe_allow_html=True,
    )
    if entry.get("thought"):
        st.markdown(_panel_html("thought", "Thought", entry["thought"]), unsafe_allow_html=True)

    if entry.get("is_final"):
        st.markdown(_panel_html("final", "Final Answer", entry.get("thought") or ""), unsafe_allow_html=True)
        continue

    if entry.get("tool_name"):
        tool_input = json.dumps(entry.get("tool_input", {}), indent=2, default=str)
        st.markdown(
            _panel_html("action", f"Action — {entry['tool_name']}", tool_input),
            unsafe_allow_html=True,
        )
        kind = "obs-err" if entry.get("is_error") else "obs-ok"
        label = "Observation (error)" if entry.get("is_error") else "Observation"
        try:
            pretty = json.dumps(json.loads(entry.get("tool_result", "") or ""), indent=2, default=str)
        except (json.JSONDecodeError, TypeError):
            pretty = entry.get("tool_result", "") or ""
        if len(pretty) > 6000:
            pretty = pretty[:6000] + "\n... (truncated)"
        st.markdown(
            _panel_html(kind, label, pretty, duration_ms=entry.get("tool_duration_ms")),
            unsafe_allow_html=True,
        )

st.caption(f"Source: `data/logs/agent_runs/{session_id}.json`")
