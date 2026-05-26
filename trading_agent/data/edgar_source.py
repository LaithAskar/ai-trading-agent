from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from edgar import Company, set_identity

from ..config import Config

_identity_set = False


def _ensure_identity() -> None:
    """SEC EDGAR's fair-use policy requires a User-Agent identifying the requester.

    We set this once per process via edgartools' set_identity().
    """
    global _identity_set
    if _identity_set:
        return
    cfg = Config.load()
    set_identity(cfg.sec_user_agent)
    _identity_set = True


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    form: str
    filing_date: str
    accession_no: str
    primary_doc_url: str | None


def list_filings(ticker: str, form: str = "10-Q", limit: int = 4) -> list[FilingRef]:
    """Most-recent filings of the given form for the ticker."""
    _ensure_identity()
    company = Company(ticker.upper())
    filings = company.get_filings(form=form).head(limit)
    out: list[FilingRef] = []
    for f in filings:
        primary_doc = getattr(f, "primary_doc_url", None) or getattr(f, "homepage_url", None)
        out.append(
            FilingRef(
                ticker=ticker.upper(),
                form=str(f.form),
                filing_date=str(f.filing_date),
                accession_no=str(f.accession_no),
                primary_doc_url=str(primary_doc) if primary_doc else None,
            )
        )
    return out


@lru_cache(maxsize=128)
def _fetch_filing(ticker: str, accession_no: str):
    _ensure_identity()
    company = Company(ticker.upper())
    for f in company.get_filings():
        if str(f.accession_no) == accession_no:
            return f
    raise KeyError(f"accession {accession_no} not found for {ticker}")


def filing_excerpt(
    ticker: str,
    accession_no: str,
    max_chars: int = 8000,
) -> dict:
    """Return key excerpts from a filing — header info + truncated text body.

    We cap output so a single tool call doesn't dump 200 pages of XBRL into
    the agent's context. The MD&A / discussion sections are the parts the
    LLM should reason over; full-text retrieval is out of scope here.
    """
    f = _fetch_filing(ticker, accession_no)

    try:
        text = f.text() if hasattr(f, "text") else ""
    except Exception:
        text = ""

    if not text:
        try:
            text = f.markdown() if hasattr(f, "markdown") else ""
        except Exception:
            text = ""

    truncated = text[:max_chars]
    return {
        "ticker": ticker.upper(),
        "form": str(f.form),
        "filing_date": str(f.filing_date),
        "accession_no": accession_no,
        "char_count_full": len(text),
        "char_count_returned": len(truncated),
        "text_excerpt": truncated,
    }
