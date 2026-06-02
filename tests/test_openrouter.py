"""V8c tests: OpenRouter provider routing + OAuth code exchange.

No network: the code exchange is tested with a fake httpx.post, and the loop
pass-through reuses the scripted FakeAnthropic from the loop tests.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_agent.llm import client as orc
from trading_agent.llm.client import (
    OPENROUTER_BASE_URL,
    build_auth_url,
    client_kwargs,
    exchange_code_for_key,
    resolve_model,
)

from tests.test_agent_loop import FakeAnthropic, FakeResponse, FakeTextBlock


# ---------- model slug resolution ----------

def test_resolve_model_anthropic_is_passthrough():
    assert resolve_model("claude-sonnet-4-6", "anthropic") == "claude-sonnet-4-6"


def test_resolve_model_openrouter_maps_known_slugs():
    assert resolve_model("claude-opus-4-7", "openrouter") == "anthropic/claude-opus-4.7"
    assert resolve_model("claude-sonnet-4-6", "openrouter") == "anthropic/claude-sonnet-4.6"
    assert resolve_model("claude-haiku-4-5", "openrouter") == "anthropic/claude-haiku-4.5"


def test_resolve_model_openrouter_unknown_falls_back_to_anthropic_namespace():
    assert resolve_model("claude-future-9", "openrouter") == "anthropic/claude-future-9"


# ---------- client kwargs ----------

def test_client_kwargs_anthropic_with_key():
    kw = client_kwargs("sk-ant-x", "anthropic")
    assert kw == {"api_key": "sk-ant-x"}
    assert "base_url" not in kw


def test_client_kwargs_anthropic_without_key_is_empty():
    assert client_kwargs(None, "anthropic") == {}


def test_client_kwargs_openrouter_sets_base_url_and_bearer():
    kw = client_kwargs("sk-or-abc", "openrouter")
    assert kw["base_url"] == OPENROUTER_BASE_URL
    assert kw["default_headers"]["Authorization"] == "Bearer sk-or-abc"
    assert kw["api_key"] == "sk-or-abc"


def test_client_kwargs_openrouter_requires_key():
    with pytest.raises(ValueError):
        client_kwargs(None, "openrouter")


# ---------- auth URL ----------

def test_build_auth_url_encodes_callback():
    url = build_auth_url("https://example.streamlit.app/")
    assert url.startswith("https://openrouter.ai/auth?")
    assert "callback_url=https%3A%2F%2Fexample.streamlit.app%2F" in url


# ---------- OAuth code exchange ----------

class _FakeResp:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 400")

    def json(self):
        return self._payload


def test_exchange_code_returns_key():
    with patch.object(orc.httpx, "post", return_value=_FakeResp({"key": "sk-or-minted"})) as p:
        key = exchange_code_for_key("one-time-code")
    assert key == "sk-or-minted"
    # body carries the code, nothing else required
    assert p.call_args.kwargs["json"] == {"code": "one-time-code"}


def test_exchange_code_raises_when_no_key_in_response():
    with patch.object(orc.httpx, "post", return_value=_FakeResp({"user_id": "u1"})):
        with pytest.raises(RuntimeError):
            exchange_code_for_key("code")


def test_exchange_code_propagates_http_error():
    with patch.object(orc.httpx, "post", return_value=_FakeResp({}, status_ok=False)):
        with pytest.raises(RuntimeError):
            exchange_code_for_key("code")


# ---------- loop routes the resolved slug through to the API ----------

def test_run_agent_openrouter_sends_resolved_slug(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_agent.agent.loop.MEMORY_DB", tmp_path / "mem.sqlite3")
    monkeypatch.setattr("trading_agent.agent.loop.AGENT_LOGS_DIR", tmp_path / "logs")

    fake = FakeAnthropic([
        FakeResponse(content=[FakeTextBlock("done")], stop_reason="end_turn"),
    ])
    with patch("trading_agent.agent.loop.anthropic.Anthropic", return_value=fake):
        from trading_agent.agent.loop import run_agent

        session = run_agent(
            goal="hi",
            mode="auto",
            model="claude-sonnet-4-6",
            api_key="sk-or-x",
            provider="openrouter",
        )

    # the API was called with the OpenRouter slug, but the session records the
    # canonical model name (so pricing/logging stay keyed on Anthropic ids).
    assert fake.messages.calls[0]["model"] == "anthropic/claude-sonnet-4.6"
    assert session.model == "claude-sonnet-4-6"
