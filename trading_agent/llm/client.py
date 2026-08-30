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

import os
from urllib.parse import urlencode

import httpx

# Host root only, NOT ".../api/v1": the Anthropic SDK appends "/v1/messages"
# itself, so a base_url ending in /v1 would produce a double /v1 and 404.
# Matches OpenRouter's native Anthropic Messages integration (ANTHROPIC_BASE_URL).
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/auth/keys"

# OpenCode "go" gateway (Nous-ecosystem aggregator): speaks BOTH OpenAI
# chat-completions and the Anthropic Messages wire format, with ~30 models
# behind one key (kimi-k3, glm-5.x, deepseek-v4, qwen3.8, gpt-5.6-luna...).
# The Anthropic SDK appends /v1/messages to this root, so the root must NOT
# end in /v1 (double /v1 -> 404); strip any such suffix from the env override.
_RAW_OPENCODE_BASE = os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
OPENCODE_BASE_URL = _RAW_OPENCODE_BASE.rstrip("/").removesuffix("/v1")

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
        if "/" in model:
            # Vendor-qualified slug (e.g. "openai/gpt-5.6-sol",
            # "qwen/qwen3.8-flash") passes through unchanged. Only bare
            # Anthropic ids get the anthropic/ namespace applied.
            return model
        return OPENROUTER_MODEL_MAP.get(model, f"anthropic/{model}")
    # opencode (and any future passthrough provider): bare slugs verbatim.
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
    if provider == "opencode":
        if not api_key:
            raise ValueError("opencode provider requires OPENCODE_GO_API_KEY")
        return {
            "api_key": api_key,
            "base_url": OPENCODE_BASE_URL,
            "default_headers": {"Authorization": f"Bearer {api_key}"},
        }
    return {"api_key": api_key} if api_key else {}


def provider_credentials(cfg, provider: str):
    """Return the credential for a provider from a Config-like object.

    Single mapping of provider -> Config field, so the CLI never needs to
    know field names. Returns None when that provider isn't configured.
    Local providers (ollama) need no credential and return "".
    """
    fields = {
        "opencode": "opencode_api_key",
        "opencode-openai": "opencode_api_key",
        "openrouter": "openrouter_api_key",
        "anthropic": "anthropic_api_key",
    }
    if provider == "ollama":
        return ""
    field = fields.get(provider)
    return getattr(cfg, field, None) if field else None


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
