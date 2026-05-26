"""Tests for the LLM filing analyzer + cache.

We don't call Claude — we inject a mock client and verify:
  - The first call hits the API, the second reads the cache.
  - The cache survives a fresh module-level reload (it's on disk).
  - The JSON parser tolerates fenced code blocks.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from trading_agent.llm.filing_analyzer import (
    FilingAnalysis,
    _get_cached,
    _parse_response,
    analyze_filing,
)


class _FakeUsage:
    def __init__(self, inp=100, out=50):
        self.input_tokens = inp
        self.output_tokens = out


class _FakeTextBlock:
    type = "text"
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage()


def _fake_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _FakeResponse(response_text)
    return client


def test_parse_response_plain_json():
    raw = '{"sentiment_score": 0.4, "confidence": 0.8, "key_drivers": ["record revenue", "raised guidance"]}'
    score, conf, drivers = _parse_response(raw)
    assert score == 0.4
    assert conf == 0.8
    assert drivers == ["record revenue", "raised guidance"]


def test_parse_response_fenced_json():
    raw = """Sure, here's my analysis:

```json
{"sentiment_score": -0.2, "confidence": 0.6, "key_drivers": ["margin compression"]}
```
"""
    score, conf, drivers = _parse_response(raw)
    assert score == -0.2
    assert drivers == ["margin compression"]


def test_parse_response_clamps_confidence():
    score, conf, drivers = _parse_response(
        '{"sentiment_score": 0.5, "confidence": 1.7, "key_drivers": []}'
    )
    assert conf == 1.0


def test_parse_response_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        _parse_response('{"sentiment_score": 1.5, "confidence": 0.5, "key_drivers": []}')


def test_analyze_filing_calls_api_then_caches(tmp_path):
    db = tmp_path / "cache.sqlite3"
    raw = '{"sentiment_score": 0.3, "confidence": 0.7, "key_drivers": ["growth"]}'
    client = _fake_client(raw)

    first = analyze_filing(
        ticker="AAPL", accession_no="0001-23-00001",
        text="Revenue grew this quarter, margins strong.",
        client=client, db_path=db,
    )
    assert first.score == 0.3
    assert not first.from_cache
    assert client.messages.create.call_count == 1

    second = analyze_filing(
        ticker="AAPL", accession_no="0001-23-00001",
        text="Revenue grew this quarter, margins strong.",
        client=client, db_path=db,
    )
    assert second.score == 0.3
    assert second.from_cache
    # API was NOT called the second time
    assert client.messages.create.call_count == 1


def test_analyze_filing_cache_only_raises_when_missing(tmp_path):
    db = tmp_path / "cache.sqlite3"
    with pytest.raises(LookupError):
        analyze_filing(
            ticker="AAPL", accession_no="0001-23-00001",
            text="text", cache_only=True, db_path=db,
        )


def test_analyze_filing_cache_only_returns_when_present(tmp_path):
    db = tmp_path / "cache.sqlite3"
    raw = '{"sentiment_score": -0.1, "confidence": 0.5, "key_drivers": ["mixed"]}'
    client = _fake_client(raw)
    analyze_filing(
        ticker="NVDA", accession_no="x-1", text="some text",
        client=client, db_path=db,
    )

    # Now cache_only=True should succeed without a client
    result = analyze_filing(
        ticker="NVDA", accession_no="x-1", text="some text",
        cache_only=True, db_path=db,
    )
    assert result.score == -0.1


def test_cache_keyed_by_text_so_different_excerpt_misses(tmp_path):
    db = tmp_path / "cache.sqlite3"
    raw = '{"sentiment_score": 0.1, "confidence": 0.5, "key_drivers": ["a"]}'
    client = _fake_client(raw)

    analyze_filing(ticker="X", accession_no="a-1", text="text version one",
                   client=client, db_path=db)
    # Same ticker + accession but DIFFERENT text -> different prompt hash -> miss
    analyze_filing(ticker="X", accession_no="a-1", text="text version TWO",
                   client=client, db_path=db)
    assert client.messages.create.call_count == 2
