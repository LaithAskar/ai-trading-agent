from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import anthropic

from ..config import DATA_DIR


CACHE_DB = DATA_DIR / "llm_cache.sqlite3"

ANALYSIS_PROMPT = """\
You are analyzing the text of a U.S. SEC filing for a publicly-traded company.

Return ONLY a JSON object with these fields (no preamble, no markdown):
{
  "sentiment_score": <float in [-1, 1]>,
  "confidence": <float in [0, 1]>,
  "key_drivers": [<short phrase>, ...]   // 1-3 items, each <= 12 words
}

Scoring rules:
- SEC filings are mandated to disclose risks and use cautious language. Score the *relative* tone, not absolute optimism.
- -1.0 = severely negative (going-concern doubts, large losses, material adverse events).
- 0.0 = neutral / typical SEC language.
- +1.0 = strongly positive (record results, raised guidance, broadly favorable outlook with limited new risks disclosed).
- confidence reflects how certain you are given the text excerpt (short or truncated excerpts -> lower confidence).
- key_drivers are the specific phrases or themes from the text driving your score.

Be honest. A neutral score is fine. Do not invent positive or negative signals not in the text.

Filing text:
\"\"\"
{text}
\"\"\"
"""


SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_filing_scores (
    ticker        TEXT NOT NULL,
    accession_no  TEXT NOT NULL,
    model         TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    score         REAL NOT NULL,
    confidence    REAL NOT NULL,
    drivers_json  TEXT NOT NULL,
    response_raw  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, accession_no, model, prompt_hash)
);
"""


@contextmanager
def _conn(db_path: Path = CACHE_DB) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


@dataclass(frozen=True)
class FilingAnalysis:
    score: float
    confidence: float
    drivers: list[str]
    from_cache: bool


def _prompt_hash(template: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(template.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _get_cached(
    ticker: str, accession_no: str, model: str, prompt_hash: str, db_path: Path = CACHE_DB
) -> FilingAnalysis | None:
    with _conn(db_path) as c:
        row = c.execute(
            """
            SELECT score, confidence, drivers_json
            FROM llm_filing_scores
            WHERE ticker = ? AND accession_no = ? AND model = ? AND prompt_hash = ?
            """,
            (ticker, accession_no, model, prompt_hash),
        ).fetchone()
    if not row:
        return None
    return FilingAnalysis(
        score=row["score"],
        confidence=row["confidence"],
        drivers=json.loads(row["drivers_json"]),
        from_cache=True,
    )


def _save_cached(
    *,
    ticker: str,
    accession_no: str,
    model: str,
    prompt_hash: str,
    score: float,
    confidence: float,
    drivers: list[str],
    response_raw: str,
    db_path: Path = CACHE_DB,
) -> None:
    with _conn(db_path) as c:
        c.execute(
            """
            INSERT OR REPLACE INTO llm_filing_scores
            (ticker, accession_no, model, prompt_hash, score, confidence,
             drivers_json, response_raw, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, accession_no, model, prompt_hash,
                score, confidence, json.dumps(drivers), response_raw,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )


def _parse_response(raw: str) -> tuple[float, float, list[str]]:
    """Extract {score, confidence, drivers} from a model response.

    Tolerates fenced code blocks and stray prefix/suffix text — we look for the
    first balanced JSON object containing 'sentiment_score'.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{[^{}]*\"sentiment_score\"[^{}]*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)

    obj = json.loads(text)
    score = float(obj["sentiment_score"])
    if not -1.0 <= score <= 1.0:
        raise ValueError(f"score {score} out of [-1, 1]")
    conf = float(obj.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))
    drivers = obj.get("key_drivers") or []
    if not isinstance(drivers, list):
        drivers = [str(drivers)]
    drivers = [str(d)[:200] for d in drivers][:5]
    return score, conf, drivers


def analyze_filing(
    *,
    ticker: str,
    accession_no: str,
    text: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 400,
    cache_only: bool = False,
    db_path: Path = CACHE_DB,
    client: anthropic.Anthropic | None = None,
) -> FilingAnalysis:
    """Get a structured sentiment analysis for a filing excerpt.

    cache_only=True skips the API and returns None-equivalent (raises) if uncached
    — useful for backtest runs you want to be free + deterministic.
    """
    prompt = ANALYSIS_PROMPT.replace("{text}", text[:30_000])
    p_hash = _prompt_hash(ANALYSIS_PROMPT, text[:30_000])

    cached = _get_cached(ticker, accession_no, model, p_hash, db_path=db_path)
    if cached is not None:
        return cached

    if cache_only:
        raise LookupError(
            f"No cached LLM analysis for {ticker} {accession_no} (cache_only=True)"
        )

    if client is None:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    score, conf, drivers = _parse_response(raw)

    _save_cached(
        ticker=ticker, accession_no=accession_no, model=model, prompt_hash=p_hash,
        score=score, confidence=conf, drivers=drivers, response_raw=raw,
        db_path=db_path,
    )
    return FilingAnalysis(score=score, confidence=conf, drivers=drivers, from_cache=False)
