from __future__ import annotations

from dataclasses import dataclass

"""Approximate Anthropic pricing per million tokens (USD).

These are *approximate* and *can change*. Verify against Anthropic's
current pricing page before relying on the dollar caps in production.
Unknown models fall back to Sonnet pricing.
"""

PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-4-7":            {"input": 15.0, "output": 75.0},
    "claude-opus-4-6":            {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6":          {"input":  3.0, "output": 15.0},
    "claude-sonnet-4-5":          {"input":  3.0, "output": 15.0},
    "claude-haiku-4-5-20251001":  {"input":  1.0, "output":  5.0},
    "claude-haiku-4-5":           {"input":  1.0, "output":  5.0},
}

_FALLBACK = PRICING_PER_MTOK["claude-sonnet-4-6"]


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_dollars: float
    output_dollars: float

    @property
    def total_dollars(self) -> float:
        return self.input_dollars + self.output_dollars


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    pricing = PRICING_PER_MTOK.get(model, _FALLBACK)
    in_dollars = input_tokens * pricing["input"] / 1_000_000
    out_dollars = output_tokens * pricing["output"] / 1_000_000
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_dollars=in_dollars,
        output_dollars=out_dollars,
    )
