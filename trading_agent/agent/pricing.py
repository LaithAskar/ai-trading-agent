from __future__ import annotations

from dataclasses import dataclass

"""Approximate Anthropic pricing per million tokens (USD).

These are *approximate* and *can change*. Verify against Anthropic's
current pricing page before relying on the dollar caps in production.
Unknown models fall back to Sonnet pricing.

Cache economics (Ch2 context engineering): Anthropic bills prompt-cache
reads at ~0.1x the base input rate and 5-minute cache writes at ~1.25x.
The agent loop leans on prompt caching, so the dollar cap must include
those multipliers or it stops matching the real invoice.
"""

PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-4-7":            {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-opus-4-6":            {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet-4-6":          {"input":  3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-sonnet-4-5":          {"input":  3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5-20251001":  {"input":  1.0, "output":  5.0, "cache_read": 0.1, "cache_write": 1.25},
    "claude-haiku-4-5":           {"input":  1.0, "output":  5.0, "cache_read": 0.1, "cache_write": 1.25},
}

_FALLBACK = PRICING_PER_MTOK["claude-sonnet-4-6"]

# OpenRouter slugs (per-token prices x1e6 from OpenRouter's public models API,
# 2026-08-29). Cache multipliers: gpt-5.6-sol 0.1x read / 1.25x write (same
# shape as Anthropic); qwen3.8-flash ~0.107x read / ~1.33x write. Models not
# listed fall back to Sonnet pricing, which MISprices non-Anthropic models —
# add a row before running a new provider/model combination.
OPENROUTER_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "openai/gpt-5.6-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
    "qwen/qwen3.8-flash": {"input": 0.15, "output": 0.47, "cache_read": 0.016, "cache_write": 0.2},
}

# OpenCode "go" gateway slugs (bare ids). Prices mirror the same underlying
# models on OpenRouter (per-token x1e6, 2026-08-29); the gateway's own billing
# may differ — treat these as honest estimates, not invoices.
OPENCODE_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-5.6-luna":       {"input": 0.20, "output": 1.20, "cache_read": 0.02, "cache_write": 0.25},
    "kimi-k3":            {"input": 3.00, "output": 15.0, "cache_read": 0.30, "cache_write": 0.0},
    "glm-5.3":            {"input": 1.40, "output": 4.40, "cache_read": 0.26, "cache_write": 0.0},
    "deepseek-v4-flash":  {"input": 0.08092, "output": 0.16184, "cache_read": 0.016184, "cache_write": 0.0},
    "minimax-m3":         {"input": 0.30, "output": 1.20, "cache_read": 0.06, "cache_write": 0.0},
    "qwen3.8-flash":      {"input": 0.15, "output": 0.47, "cache_read": 0.02, "cache_write": 0.20},
}


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_dollars: float
    output_dollars: float
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_dollars(self) -> float:
        return self.input_dollars + self.output_dollars


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> CostEstimate:
    pricing = (
        PRICING_PER_MTOK.get(model)
        or OPENROUTER_PRICING_PER_MTOK.get(model)
        or OPENCODE_PRICING_PER_MTOK.get(model)
        or _FALLBACK
    )
    in_dollars = input_tokens * pricing["input"] / 1_000_000
    out_dollars = output_tokens * pricing["output"] / 1_000_000
    if cache_read_tokens or cache_write_tokens:
        in_dollars += (
            cache_read_tokens * pricing["cache_read"] / 1_000_000
            + cache_write_tokens * pricing["cache_write"] / 1_000_000
        )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_dollars=in_dollars,
        output_dollars=out_dollars,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
