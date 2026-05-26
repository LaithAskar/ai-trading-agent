from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = DATA_DIR / "logs"
AGENT_LOGS_DIR = LOGS_DIR / "agent_runs"
MEMORY_DB = DATA_DIR / "memory.sqlite3"
MCP_DIR = DATA_DIR / "mcp"


@dataclass(frozen=True)
class Config:
    live_trading: bool
    log_level: str
    anthropic_api_key: str | None
    agent_model: str
    agent_max_iters: int
    agent_max_tokens_per_call: int
    agent_max_session_tokens: int
    agent_max_session_dollars: float
    sec_user_agent: str
    alpaca_api_key: str | None
    alpaca_api_secret: str | None
    alpaca_paper: bool

    @classmethod
    def load(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            live_trading=os.getenv("LIVE_TRADING", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            agent_model=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"),
            agent_max_iters=int(os.getenv("AGENT_MAX_ITERS", "20")),
            agent_max_tokens_per_call=int(os.getenv("AGENT_MAX_TOKENS_PER_CALL", "4096")),
            agent_max_session_tokens=int(os.getenv("AGENT_MAX_SESSION_TOKENS", "200000")),
            agent_max_session_dollars=float(os.getenv("AGENT_MAX_SESSION_DOLLARS", "1.00")),
            sec_user_agent=os.getenv("SEC_USER_AGENT", "trading-agent research dev@example.com"),
            alpaca_api_key=os.getenv("ALPACA_API_KEY") or None,
            alpaca_api_secret=os.getenv("ALPACA_API_SECRET") or None,
            alpaca_paper=os.getenv("ALPACA_PAPER", "true").lower() == "true",
        )


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RESULTS_DIR, LOGS_DIR, AGENT_LOGS_DIR, MCP_DIR):
        d.mkdir(parents=True, exist_ok=True)
