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
    pricing = PRICING_PER_MTOK.get(model, _FALLBACK)
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
