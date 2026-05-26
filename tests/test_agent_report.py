from __future__ import annotations

import json

from trading_agent.agent.report import render_html, render_session_file


SAMPLE_SESSION = {
    "session_id": "abc123",
    "goal": "Test the renderer",
    "mode": "auto",
    "model": "claude-sonnet-4-6",
    "started_at": "2026-05-25T12:00:00+00:00",
    "input_tokens": 1234,
    "output_tokens": 567,
    "finished": True,
    "final_summary": "All done.",
    "transcript": [
        {
            "iteration": 1,
            "thought": "I should list strategies first.",
            "tool_name": "list_strategies",
            "tool_input": {},
            "tool_result": json.dumps(["sma_cross"]),
            "is_final": False,
        },
        {
            "iteration": 2,
            "thought": "Got it. All done.",
            "tool_name": None,
            "tool_input": None,
            "tool_result": None,
            "is_final": True,
        },
    ],
}


def test_render_html_includes_key_sections():
    out = render_html(SAMPLE_SESSION)
    assert "<!DOCTYPE html>" in out
    assert "abc123" in out
    assert "list_strategies" in out
    assert "All done" in out
    assert 'class="panel thought"' in out
    assert 'class="panel action"' in out
    assert 'class="panel final"' in out


def test_render_html_forces_light_theme():
    out = render_html(SAMPLE_SESSION)
    assert "color-scheme: light only" in out
    assert 'name="color-scheme" content="light only"' in out


def test_render_html_escapes_user_content():
    session = {**SAMPLE_SESSION, "goal": "<script>alert(1)</script>"}
    out = render_html(session)
    assert "<script>alert(1)" not in out
    assert "&lt;script&gt;" in out


def test_render_session_file_writes_html(tmp_path):
    src = tmp_path / "sess.json"
    src.write_text(json.dumps(SAMPLE_SESSION))
    out = render_session_file(src)
    assert out.exists()
    assert out.suffix == ".html"
    assert "abc123" in out.read_text()


def test_render_html_marks_error_observations():
    session = {
        **SAMPLE_SESSION,
        "transcript": [
            {
                "iteration": 1,
                "thought": "Try a bad call",
                "tool_name": "read_strategy_code",
                "tool_input": {"name": "../etc/passwd"},
                "tool_result": json.dumps({"error": "Invalid strategy name"}),
                "is_final": False,
            }
        ],
    }
    out = render_html(session)
    assert 'class="panel obs-err"' in out
    assert "Observation (error)" in out
