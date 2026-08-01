from __future__ import annotations

from typing import Any

STRATEGY_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "sma_cross": {"fast": 20, "slow": 50},
    "rsi_mean_rev": {"period": 14, "oversold": 30.0, "overbought": 70.0},
    "ai_oligopoly_leaders": {
        "fast": 20,
        "slow": 50,
        "momentum_lookback": 20,
        "min_momentum_pct": 5.0,
        "leaders": "NVDA|MSFT|GOOGL|AMZN|META|AVGO|TSM|ASML",
        "target_notional": 20.0,
    },
    "filings_sentiment": {"threshold": 0.0, "form": "10-Q", "max_filings": 30},
    "filings_sentiment_llm": {
        "threshold": 0.05,
        "form": "10-Q",
        "max_filings": 20,
        "cache_only": True,
    },
    "news_sentiment": {
        "window": 5,
        "enter_threshold": 0.15,
        "exit_threshold": 0.05,
        "min_articles_per_day": 1,
    },
    "news_event_burst": {
        "relevance_floor": 0.3,
        "score_threshold": 0.15,
        "burst_count": 3,
        "burst_window": 3,
        "hold_days": 5,
        "lookback_days": 600,
    },
}


def strategy_default_params_text(strategy: str) -> str:
    """Return form-ready defaults that only belong to the selected strategy."""
    params = STRATEGY_DEFAULT_PARAMS.get(strategy, {})
    return ",".join(f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in params.items())


def parse_strategy_params_text(text: str) -> dict[str, Any]:
    """Parse the UI's comma-delimited key/value form without corrupting values.

    Collection-valued strategy parameters use ``|`` inside their value so the
    outer comma remains an unambiguous parameter delimiter.
    """
    parsed: dict[str, Any] = {}
    if not text.strip():
        return parsed
    for item in text.split(","):
        if "=" not in item:
            continue
        key, value = (part.strip() for part in item.split("=", 1))
        if value.lower() in {"true", "false"}:
            parsed[key] = value.lower() == "true"
            continue
        try:
            parsed[key] = int(value)
        except ValueError:
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
    return parsed
