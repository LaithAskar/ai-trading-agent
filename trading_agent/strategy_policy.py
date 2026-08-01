from __future__ import annotations

RESEARCH_ONLY_STRATEGIES = frozenset({"ai_oligopoly_leaders"})


def execution_block_reason(strategy_name: str) -> str | None:
    """Return a mandatory execution-boundary rejection for research-only strategies."""
    if strategy_name in RESEARCH_ONLY_STRATEGIES:
        return (
            f"strategy {strategy_name!r} is research-only and cannot submit orders; "
            "hard-notional or bounded-price execution is not implemented"
        )
    return None
