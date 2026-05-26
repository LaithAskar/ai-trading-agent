from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


CSS = """
:root {
    color-scheme: light only;
    --bg: #f5f5f7;
    --surface: #ffffff;
    --text: #1d1d1f;
    --muted: #6e6e73;
    --border: #d2d2d7;
    --code-bg: #fafafa;
    --code-border: #e3e3e7;
    --thought: #0a84ff;
    --action: #5ac8fa;
    --action-text: #0a7ea4;
    --obs-ok: #34c759;
    --obs-ok-text: #248a3d;
    --obs-err: #ff3b30;
    --obs-err-text: #c62a1f;
    --obs-err-bg: #fff5f4;
    --final: #34c759;
    --final-bg: #f3fdf4;
    --final-text: #1f7a30;
    --pending: #c7c7cc;
}
* { box-sizing: border-box; }
html, body { background: var(--bg) !important; color: var(--text) !important; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    padding: 32px 16px 96px;
    line-height: 1.5;
}
.container { max-width: 980px; margin: 0 auto; }

.header {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.header h1 { margin: 0 0 8px 0; font-size: 20px; font-weight: 600; }
.header .meta {
    color: var(--muted) !important;
    font-size: 13px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 6px 24px;
}
.meta span b { color: var(--text) !important; font-weight: 500; }

.controls {
    position: sticky;
    top: 8px;
    z-index: 10;
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.controls button {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text) !important;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    border-radius: 999px;
    cursor: pointer;
    font-family: inherit;
}
.controls button:hover { background: #ececef; }
.controls button:active { transform: scale(0.97); }
.controls button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}
.controls .play { background: #0a84ff; color: white !important; border-color: #0a84ff; }
.controls .play:hover { background: #007aff; }
.controls .progress {
    font-size: 13px;
    color: var(--muted) !important;
    margin-left: auto;
    font-variant-numeric: tabular-nums;
}
.controls .speed { font-size: 12px; color: var(--muted) !important; }
.controls select {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text) !important;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: inherit;
}

.iteration {
    margin: 20px 0;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.35s ease, transform 0.35s ease;
    pointer-events: none;
}
.iteration.visible {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
}
.iteration-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted) !important;
    margin: 0 0 10px 4px;
    font-weight: 600;
}
.panel {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-left: 4px solid var(--border);
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.panel-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: var(--muted) !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.panel-body {
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text) !important;
}
.panel.thought  { border-left-color: var(--thought); }
.panel.thought .panel-title { color: var(--thought) !important; }
.panel.action   { border-left-color: var(--action); }
.panel.action .panel-title { color: var(--action-text) !important; }
.panel.obs-ok   { border-left-color: var(--obs-ok); }
.panel.obs-ok .panel-title { color: var(--obs-ok-text) !important; }
.panel.obs-err  { border-left-color: var(--obs-err); background: var(--obs-err-bg) !important; }
.panel.obs-err .panel-title { color: var(--obs-err-text) !important; }
.panel.final    { border-left-color: var(--final); background: var(--final-bg) !important; }
.panel.final .panel-title { color: var(--final-text) !important; }
.panel.goal { border-left-color: var(--text); }

pre.code {
    background: var(--code-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--code-border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: "SF Mono", "Consolas", "Monaco", monospace;
    font-size: 13px;
    overflow-x: auto;
    margin: 0;
    line-height: 1.45;
    max-height: 240px;
    overflow-y: auto;
}
pre.code.expanded { max-height: none; }

.expand-toggle {
    background: none;
    border: none;
    color: var(--muted) !important;
    font-size: 11px;
    cursor: pointer;
    padding: 0;
    font-family: inherit;
}
.expand-toggle:hover { color: var(--text) !important; text-decoration: underline; }

.footer {
    margin-top: 32px;
    padding: 16px 0;
    border-top: 1px solid var(--border);
    color: var(--muted) !important;
    font-size: 12px;
    text-align: center;
}
.duration-chip {
    display: inline-block;
    background: #ececef;
    color: var(--muted) !important;
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 999px;
    margin-left: 8px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
"""


JS = """
(function() {
    const iters = Array.from(document.querySelectorAll('.iteration'));
    const total = iters.length;
    let cursor = 0;
    let playing = false;
    let timer = null;

    const playBtn   = document.getElementById('btn-play');
    const nextBtn   = document.getElementById('btn-next');
    const resetBtn  = document.getElementById('btn-reset');
    const endBtn    = document.getElementById('btn-end');
    const speedSel  = document.getElementById('sel-speed');
    const progress  = document.getElementById('progress');

    function updateProgress() {
        progress.textContent = `${cursor} / ${total}`;
        nextBtn.disabled = cursor >= total;
        endBtn.disabled = cursor >= total;
    }
    function revealNext() {
        if (cursor >= total) { stop(); return; }
        iters[cursor].classList.add('visible');
        // Scroll into view smoothly
        iters[cursor].scrollIntoView({ behavior: 'smooth', block: 'center' });
        cursor++;
        updateProgress();
    }
    function reset() {
        stop();
        cursor = 0;
        iters.forEach(it => it.classList.remove('visible'));
        updateProgress();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    function skipToEnd() {
        stop();
        iters.forEach(it => it.classList.add('visible'));
        cursor = total;
        updateProgress();
    }
    function start() {
        if (cursor >= total) reset();
        playing = true;
        playBtn.textContent = '⏸ Pause';
        playBtn.classList.remove('play');
        const interval = parseInt(speedSel.value, 10);
        timer = setInterval(() => {
            revealNext();
            if (cursor >= total) stop();
        }, interval);
    }
    function stop() {
        playing = false;
        playBtn.textContent = '▶ Play';
        playBtn.classList.add('play');
        if (timer) { clearInterval(timer); timer = null; }
    }
    function togglePlay() {
        if (playing) stop(); else start();
    }

    playBtn.addEventListener('click', togglePlay);
    nextBtn.addEventListener('click', () => { stop(); revealNext(); });
    resetBtn.addEventListener('click', reset);
    endBtn.addEventListener('click', skipToEnd);
    speedSel.addEventListener('change', () => {
        if (playing) { stop(); start(); }
    });

    // Keyboard shortcuts: Space = play/pause, → = next, R = reset
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'BUTTON') return;
        if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
        else if (e.code === 'ArrowRight') { e.preventDefault(); stop(); revealNext(); }
        else if (e.key === 'r' || e.key === 'R') { reset(); }
    });

    // Click-to-expand long code blocks
    document.querySelectorAll('.expand-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const pre = btn.closest('.panel').querySelector('pre.code');
            if (!pre) return;
            const isExpanded = pre.classList.toggle('expanded');
            btn.textContent = isExpanded ? 'collapse' : 'expand';
        });
    });

    updateProgress();
})();
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


def _panel(kind: str, title: str, body_html: str, duration_ms: int | None = None, expandable: bool = False) -> str:
    duration = ""
    if duration_ms is not None:
        duration = f'<span class="duration-chip">{duration_ms} ms</span>'
    expand = ""
    if expandable:
        expand = '<button class="expand-toggle">expand</button>'
    return (
        f'<div class="panel {kind}">'
        f'<p class="panel-title"><span>{html.escape(title)}{duration}</span>{expand}</p>'
        f'<div class="panel-body">{body_html}</div>'
        f"</div>"
    )


def _format_thought(text: str) -> str:
    return html.escape(text.strip())


def render_html(session: dict) -> str:
    transcript = session.get("transcript", [])
    iteration_count = len(transcript)

    header = (
        f'<div class="header">'
        f'<h1>Agent session {html.escape(session.get("session_id", "?"))}</h1>'
        f'<div class="meta">'
        f'<span><b>Model:</b> {html.escape(session.get("model", "?"))}</span>'
        f'<span><b>Mode:</b> {html.escape(session.get("mode", "?"))}</span>'
        f'<span><b>Started:</b> {html.escape(session.get("started_at", "?"))}</span>'
        f'<span><b>Iterations:</b> {iteration_count}</span>'
        f'<span><b>Input tokens:</b> {session.get("input_tokens", 0):,}</span>'
        f'<span><b>Output tokens:</b> {session.get("output_tokens", 0):,}</span>'
        f'<span><b>Est. cost:</b> ${session.get("cost_dollars", 0):.4f}</span>'
        f'<span><b>Finished:</b> {session.get("finished", False)}</span>'
        f"</div></div>"
    )

    controls = (
        '<div class="controls">'
        '<button id="btn-play" class="play">▶ Play</button>'
        '<button id="btn-next">⏭ Next</button>'
        '<button id="btn-reset">⏮ Reset</button>'
        '<button id="btn-end">⏩ Skip to end</button>'
        '<span class="speed">speed:</span>'
        '<select id="sel-speed">'
          '<option value="600">fast</option>'
          '<option value="1400" selected>normal</option>'
          '<option value="2500">slow</option>'
        '</select>'
        '<span class="progress" id="progress">0 / 0</span>'
        '</div>'
    )

    goal_panel = _panel("goal", "Goal", html.escape(session.get("goal", "")))

    iters: list[str] = []
    for entry in transcript:
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

            is_error = bool(entry.get("is_error"))
            if not is_error:
                result_str = entry.get("tool_result") or ""
                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, dict) and "error" in parsed:
                        is_error = True
                except (json.JSONDecodeError, TypeError):
                    pass

            result_pretty_html = _pretty_json(entry.get("tool_result"))
            looks_long = len(result_pretty_html) > 1500
            obs_body = f'<pre class="code">{result_pretty_html}</pre>'
            parts.append(
                _panel(
                    "obs-err" if is_error else "obs-ok",
                    "Observation (error)" if is_error else "Observation",
                    obs_body,
                    duration_ms=entry.get("tool_duration_ms"),
                    expandable=looks_long,
                )
            )

        iters.append(
            f'<div class="iteration" data-iter="{entry.get("iteration", "?")}">'
            f'<p class="iteration-label">Iteration {entry.get("iteration", "?")}</p>'
            + "".join(parts)
            + "</div>"
        )

    footer = (
        '<div class="footer">'
        'Press <kbd>Space</kbd> to play/pause · <kbd>→</kbd> for next · <kbd>R</kbd> to reset · '
        'Generated by trading-agent render-transcript'
        "</div>"
    )

    return (
        "<!DOCTYPE html>"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="color-scheme" content="light only">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Agent session {html.escape(session.get('session_id', ''))}</title>"
        f"<style>{CSS}</style>"
        "</head><body>"
        '<div class="container">'
        + header
        + controls
        + goal_panel
        + "".join(iters)
        + footer
        + "</div>"
        f"<script>{JS}</script>"
        "</body></html>"
    )


def render_session_file(json_path: Path) -> Path:
    """Read a session JSON file, render to HTML, write next to it. Returns the HTML path."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = json_path.with_suffix(".html")
    out.write_text(render_html(data), encoding="utf-8")
    return out
