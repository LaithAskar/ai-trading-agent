"""Single place that knows how to talk to an LLM provider.

The agent loop and the filing analyzer both construct an ``anthropic.Anthropic``
client. We support two providers:

- ``anthropic``  — the SDK's default endpoint, authenticated with an Anthropic key.
- ``openrouter`` — OpenRouter's *native Anthropic Messages* endpoint, which speaks
  the same wire format, so the hand-written ReAct loop is unchanged. Only the
  base URL, the auth header, and the model slug differ.

``client_kwargs`` returns kwargs rather than the client itself on purpose: call
sites build the client in their own module namespace, which keeps existing test
patch points (``patch("trading_agent.agent.loop.anthropic.Anthropic")``) working.
"""
from __future__ import annotations

from urllib.parse import urlencode

import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/auth/keys"

# Canonical Anthropic model id -> OpenRouter slug. OpenRouter uses dotted
# versions ("anthropic/claude-sonnet-4.6"); we use hyphenated ("claude-sonnet-4-6").
OPENROUTER_MODEL_MAP: dict[str, str] = {
    "claude-opus-4-7": "anthropic/claude-opus-4.7",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
}


def resolve_model(model: str, provider: str = "anthropic") -> str:
    """Translate a canonical model id to the slug the provider expects."""
    if provider == "openrouter":
        return OPENROUTER_MODEL_MAP.get(model, f"anthropic/{model}")
    return model


def client_kwargs(api_key: str | None, provider: str = "anthropic") -> dict:
    """Build kwargs for ``anthropic.Anthropic(**kwargs)`` for the given provider."""
    if provider == "openrouter":
        if not api_key:
            raise ValueError("OpenRouter provider requires an API key")
        # OpenRouter authenticates via bearer token. We still pass api_key so the
        # SDK doesn't fall back to reading ANTHROPIC_API_KEY from the environment.
        return {
            "api_key": api_key,
            "base_url": OPENROUTER_BASE_URL,
            "default_headers": {"Authorization": f"Bearer {api_key}"},
        }
    return {"api_key": api_key} if api_key else {}


def build_auth_url(callback_url: str) -> str:
    """The URL to send the user to so they authorize the app on OpenRouter.

    Bare authorization-code flow: no PKCE ``code_challenge`` (see V8c plan — the
    verifier can't survive Streamlit Cloud's redirect, and the code is single-use
    over HTTPS to a registered callback).
    """
    return f"{OPENROUTER_AUTH_URL}?{urlencode({'callback_url': callback_url})}"


def exchange_code_for_key(code: str, *, timeout: float = 30.0) -> str:
    """Exchange a one-time OAuth code for a user-controlled OpenRouter API key.

    Never logs the code or the returned key.
    """
    resp = httpx.post(OPENROUTER_KEYS_URL, json={"code": code}, timeout=timeout)
    resp.raise_for_status()
    key = resp.json().get("key")
    if not key:
        raise RuntimeError("OpenRouter code exchange returned no key")
    return key
