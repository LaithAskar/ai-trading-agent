from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


CSS = """
:root {
    color-scheme: light only;
}
* { box-sizing: border-box; }
body {
    background: #f5f5f7 !important;
    color: #1d1d1f !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    padding: 32px 16px;
    line-height: 1.5;
}
.container {
    max-width: 980px;
    margin: 0 auto;
}
.header {
    background: #ffffff !important;
    color: #1d1d1f !important;
    border: 1px solid #d2d2d7;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
}
.header h1 {
    margin: 0 0 8px 0;
    font-size: 20px;
    font-weight: 600;
}
.header .meta {
    color: #6e6e73 !important;
    font-size: 13px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 6px 24px;
}
.meta span b {
    color: #1d1d1f !important;
    font-weight: 500;
}
.iteration {
    margin: 24px 0;
}
.iteration-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6e6e73 !important;
    margin: 0 0 10px 4px;
    font-weight: 600;
}
.panel {
    background: #ffffff !important;
    color: #1d1d1f !important;
    border-left: 4px solid #d2d2d7;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-top: 1px solid #d2d2d7;
    border-right: 1px solid #d2d2d7;
    border-bottom: 1px solid #d2d2d7;
}
.panel-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: #6e6e73 !important;
}
.panel-body {
    white-space: pre-wrap;
    word-break: break-word;
    color: #1d1d1f !important;
}
.panel.thought  { border-left-color: #0a84ff; }
.panel.thought .panel-title { color: #0a84ff !important; }
.panel.action   { border-left-color: #5ac8fa; }
.panel.action .panel-title { color: #0a7ea4 !important; }
.panel.obs-ok   { border-left-color: #34c759; }
.panel.obs-ok .panel-title { color: #248a3d !important; }
.panel.obs-err  { border-left-color: #ff3b30; background: #fff5f4 !important; }
.panel.obs-err .panel-title { color: #c62a1f !important; }
.panel.final    { border-left-color: #34c759; background: #f3fdf4 !important; }
.panel.final .panel-title { color: #1f7a30 !important; }
.panel.goal { border-left-color: #1d1d1f; }
pre.code {
    background: #fafafa !important;
    color: #1d1d1f !important;
    border: 1px solid #e3e3e7;
    border-radius: 6px;
    padding: 10px 12px;
    font-family: "SF Mono", "Consolas", "Monaco", monospace;
    font-size: 13px;
    overflow-x: auto;
    margin: 0;
    line-height: 1.45;
}
.footer {
    margin-top: 32px;
    padding: 16px 0;
    border-top: 1px solid #d2d2d7;
    color: #6e6e73 !important;
    font-size: 12px;
    text-align: center;
}
"""


def _pretty_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return html.escape(value)
    return html.escape(json.dumps(value, indent=2, default=str))


def _panel(kind: str, title: str, body_html: str) -> str:
    return (
        f'<div class="panel {kind}">'
        f'<p class="panel-title">{html.escape(title)}</p>'
        f'<div class="panel-body">{body_html}</div>'
        f"</div>"
    )


def _format_thought(text: str) -> str:
    return html.escape(text.strip())


def render_html(session: dict) -> str:
    """Render a session dict to a self-contained styled HTML page."""
    header = (
        f'<div class="header">'
        f'<h1>Agent session {html.escape(session.get("session_id", "?"))}</h1>'
        f'<div class="meta">'
        f'<span><b>Model:</b> {html.escape(session.get("model", "?"))}</span>'
        f'<span><b>Mode:</b> {html.escape(session.get("mode", "?"))}</span>'
        f'<span><b>Started:</b> {html.escape(session.get("started_at", "?"))}</span>'
        f'<span><b>Iterations:</b> {len(session.get("transcript", []))}</span>'
        f'<span><b>Input tokens:</b> {session.get("input_tokens", 0):,}</span>'
        f'<span><b>Output tokens:</b> {session.get("output_tokens", 0):,}</span>'
        f'<span><b>Finished:</b> {session.get("finished", False)}</span>'
        f"</div></div>"
    )

    goal_panel = _panel("goal", "Goal", html.escape(session.get("goal", "")))

    iters: list[str] = []
    for entry in session.get("transcript", []):
        parts: list[str] = []
        if entry.get("thought"):
            parts.append(_panel("thought", "Thought", _format_thought(entry["thought"])))

        if entry.get("is_final"):
            parts.append(
                _panel("final", "Final Answer", _format_thought(entry["thought"] or ""))
            )
        elif entry.get("tool_name"):
            action_body = (
                f'<p style="margin:0 0 8px 0;">'
                f'<b>Tool:</b> <code>{html.escape(entry["tool_name"])}</code>'
                f"</p>"
                f'<pre class="code">{_pretty_json(entry.get("tool_input"))}</pre>'
            )
            parts.append(_panel("action", "Action", action_body))

            is_error = False
            result_str = entry.get("tool_result") or ""
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, dict) and "error" in parsed:
                    is_error = True
            except (json.JSONDecodeError, TypeError):
                pass

            obs_body = f'<pre class="code">{_pretty_json(result_str)}</pre>'
            parts.append(
                _panel(
                    "obs-err" if is_error else "obs-ok",
                    "Observation (error)" if is_error else "Observation",
                    obs_body,
                )
            )

        iters.append(
            f'<div class="iteration">'
            f'<p class="iteration-label">Iteration {entry.get("iteration", "?")}</p>'
            + "".join(parts)
            + "</div>"
        )

    footer = (
        '<div class="footer">'
        f"Goal completed: {session.get('final_summary', '(no summary)') and 'yes' or 'no'} · "
        "Generated by trading-agent render-transcript"
        "</div>"
    )

    return (
        "<!DOCTYPE html>"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="color-scheme" content="light only">'
        f"<title>Agent session {html.escape(session.get('session_id', ''))}</title>"
        f"<style>{CSS}</style>"
        "</head><body>"
        '<div class="container">'
        + header
        + goal_panel
        + "".join(iters)
        + footer
        + "</div></body></html>"
    )


def render_session_file(json_path: Path) -> Path:
    """Read a session JSON file, render to HTML, write next to it. Returns the HTML path."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = json_path.with_suffix(".html")
    out.write_text(render_html(data), encoding="utf-8")
    return out
