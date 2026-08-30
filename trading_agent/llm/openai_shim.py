"""Anthropic-shaped client shim backed by the OpenAI chat/completions wire.

The agent loop speaks Anthropic Messages objects (content blocks, tool_use,
tool_result, stop_reason). Some providers only speak OpenAI chat completions:
the opencode gateway's OpenAI-family models (gpt-5.6-luna, grok-4.x,
minimax-m3) and local Ollama. Rather than fork the loop, this shim duck-types
the surface the loop actually uses:

    client.messages.create(model=, max_tokens=, system=, tools=, messages=)
      -> response with .content (list of text/tool_use blocks),
         .stop_reason ("end_turn" | "tool_use" | ...), .usage
         (.input_tokens, .output_tokens, .cache_read_input_tokens,
          .cache_creation_input_tokens)

Conversion rules (Ch2 "use the standard API format" — each wire gets its
native shapes, no hand-rolled concatenation):
  - system blocks  -> single system message (block list flattened)
  - tool schemas   -> {"type":"function","function":{name,description,parameters}}
  - user tool_result blocks -> role:"tool" messages keyed by tool_call_id
  - assistant tool_use blocks -> assistant message with tool_calls
  - finish_reason tool_calls/stop -> stop_reason tool_use/end_turn
  - usage.prompt_tokens_details.cached_tokens -> cache_read_input_tokens
    (OpenAI-family caching is implicit: no separate cache-write field)
"""
from __future__ import annotations

import json
from typing import Any

import httpx

_FINISH_REASON_MAP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


class _Usage:
    def __init__(self, data: dict):
        self.input_tokens = data.get("prompt_tokens", 0) or 0
        self.output_tokens = data.get("completion_tokens", 0) or 0
        details = data.get("prompt_tokens_details") or {}
        self.cache_read_input_tokens = details.get("cached_tokens", 0) or 0
        # OpenAI-family caching is implicit; there is no separate write bucket.
        self.cache_creation_input_tokens = 0


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, tool_id: str, name: str, arguments: str):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        try:
            self.input = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            self.input = {"_raw_arguments": arguments}


class _Response:
    def __init__(self, content: list, stop_reason: str, usage: _Usage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


def _blocks_to_openai_content(content) -> Any:
    """Anthropic content (str or block list) -> OpenAI content (str or parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[dict] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
            continue
        btype = block.get("type")
        if btype == "text":
            part = {"type": "text", "text": block.get("text", "")}
            if "cache_control" in block:
                part["cache_control"] = block["cache_control"]
            parts.append(part)
        # tool_result blocks are handled by the message converter, not here
    if parts and all(p.get("type") == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _system_to_content(system) -> Any:
    """Anthropic system (str or text-block list) -> OpenAI system content."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    texts: list[str] = []
    cache_control = None
    for block in system:
        if isinstance(block, dict):
            texts.append(block.get("text", ""))
            if "cache_control" in block:
                cache_control = block["cache_control"]
        else:
            texts.append(str(block))
    body = "\n".join(texts)
    if cache_control is not None:
        return [{"type": "text", "text": body, "cache_control": cache_control}]
    return body


class _MessagesNamespace:
    """Mirrors the SDK's ``client.messages.create(...)`` call surface."""

    def __init__(self, client: "OpenAICompatClient"):
        self._client = client

    def create(self, **kwargs) -> "_Response":
        return self._client.messages_create(**kwargs)


class OpenAICompatClient:
    """Duck-typed Anthropic client backed by POST {base}/chat/completions."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 180.0):
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        self._url = base + "/chat/completions"
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.messages = _MessagesNamespace(self)

    # -- request conversion -------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for message in messages:
            role = message["role"]
            content = message.get("content")
            if role == "user" and isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                # Anthropic returns tool results inside a user message; the
                # OpenAI wire wants one role:"tool" message per result.
                for block in content:
                    if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                        continue
                    result = block.get("content", "")
                    if not isinstance(result, str):
                        result = json.dumps(result)
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": result,
                        }
                    )
                continue
            if role == "assistant" and isinstance(content, list):
                # Blocks may be plain dicts (hand-built history) or the shim's
                # attribute-style response objects (loop round-trip: the loop
                # appends response.content verbatim). Normalize both.
                def _btype(b):
                    return b.get("type") if isinstance(b, dict) else getattr(b, "type", None)

                def _bget(b, key, default=None):
                    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)

                text = "\n".join(
                    _bget(b, "text", "") or "" for b in content if _btype(b) == "text"
                ).strip()
                converted: dict = {"role": "assistant", "content": text or None}
                tool_calls = [
                    {
                        "id": _bget(b, "id"),
                        "type": "function",
                        "function": {
                            "name": _bget(b, "name"),
                            "arguments": json.dumps(_bget(b, "input", {}) or {}),
                        },
                    }
                    for b in content
                    if _btype(b) == "tool_use"
                ]
                if tool_calls:
                    converted["tool_calls"] = tool_calls
                out.append(converted)
                continue
            out.append(
                {
                    "role": role,
                    "content": content if isinstance(content, str) else _blocks_to_openai_content(content),
                }
            )
        return out

    def _build_payload(self, *, model, max_tokens, system, tools, messages) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages),
        }
        system_content = _system_to_content(system)
        if system_content:
            payload["messages"] = [{"role": "system", "content": system_content}] + payload["messages"]
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
        return payload

    # -- response conversion -------------------------------------------------

    @staticmethod
    def _convert_response(data: dict) -> _Response:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        blocks: list = []
        if message.get("content"):
            blocks.append(_TextBlock(message["content"]))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            blocks.append(
                _ToolUseBlock(
                    call.get("id") or f"call_{fn.get('name', 'unknown')}",
                    fn.get("name", "unknown"),
                    fn.get("arguments") or "{}",
                )
            )
        stop_reason = _FINISH_REASON_MAP.get(choice.get("finish_reason") or "", "end_turn")
        return _Response(blocks, stop_reason, _Usage(data.get("usage") or {}))

    # -- the surface the loop calls -------------------------------------------

    def messages_create(self, **kwargs) -> _Response:
        payload = self._build_payload(
            model=kwargs["model"],
            max_tokens=kwargs["max_tokens"],
            system=kwargs.get("system"),
            tools=kwargs.get("tools") or [],
            messages=kwargs.get("messages") or [],
        )
        response = httpx.post(self._url, headers=self._headers, json=payload, timeout=self._timeout)
        response.raise_for_status()
        return self._convert_response(response.json())
