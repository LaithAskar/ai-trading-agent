"""Guard tests for the Windows cp1252 console crash.

The agent prints model responses (often containing emoji / non-Latin-1 chars)
to the terminal via rich. On Windows the default stdout codec is cp1252, which
raises UnicodeEncodeError and kills the run. loop._reconfigure_utf8 switches a
stream to UTF-8; these tests pin that behavior so the regression can't return.
"""
from __future__ import annotations

import io

from trading_agent.agent.loop import _reconfigure_utf8


def test_reconfigure_switches_cp1252_stream_to_utf8():
    # Simulate the headless Windows stdout: cp1252, strict.
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    assert stream.encoding.lower() in ("cp1252", "windows-1252")

    applied = _reconfigure_utf8(stream)

    assert applied is True
    assert stream.encoding.lower() == "utf-8"
    # The exact char from the production crash: ✅ (U+2705). Must not raise now.
    stream.write("✅ sma_cross did NOT beat buy-and-hold ≈ 0.44")
    stream.flush()


def test_reconfigure_is_a_noop_on_streams_without_reconfigure():
    # io.StringIO has no reconfigure(); the guard must skip it, not crash.
    assert _reconfigure_utf8(io.StringIO()) is False
