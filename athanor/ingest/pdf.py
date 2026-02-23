"""
athanor.ingest.pdf — extract full text from open-access PDFs.

Uses pypdf for extraction. Falls back gracefully to abstract-only
if the PDF is unavailable or extraction fails — downstream code never
needs to know.

The interface matches what parse_papers() already expects: Paper.full_text
is either None (→ use abstract) or the extracted body text.
"""
from __future__ import annotations

import io
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

from athanor.ingest.arxiv_client import Paper

log = logging.getLogger(__name__)

_ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
_TIMEOUT = 30
_MAX_PAGES = 12  # intro + methods; enough for concept extraction


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = _MAX_PAGES) -> str:
    """Return cleaned text from *pdf_bytes*, up to *max_pages* pages."""
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed — `pip install pypdf`")
        return ""

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages[:max_pages]
    chunks = []
    for page in pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            pass
    return _clean(("\n\n".join(chunks)))


def _clean(text: str) -> str:
    """Remove common PDF extraction noise."""
    # collapse hyphenated line-breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip page headers/footers (lines shorter than 4 words that repeat)
    return text.strip()


def fetch_pdf_bytes(url: str, session: Optional[requests.Session] = None) -> Optional[bytes]:
    """Download a PDF from *url*, returning bytes or None on failure."""
    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=_TIMEOUT, headers={"User-Agent": "athanor/1.0"})
        if resp.status_code == 200 and "pdf" in resp.headers.get("content-type", "").lower():
            return resp.content
        log.debug("Non-PDF response from %s: %s %s", url, resp.status_code, resp.headers.get("content-type"))
        return None
    except requests.RequestException as exc:
        log.debug("PDF download failed for %s: %s", url, exc)
        return None


def arxiv_pdf_url(arxiv_id: str) -> str:
    """Construct the arXiv open-access PDF URL for a paper ID."""
    # Strip version suffix for the URL if present
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    return _ARXIV_PDF.format(arxiv_id=base_id)


def enrich_papers_with_fulltext(
    papers: List[Paper],
    max_papers: Optional[int] = None,
    sleep: float = 1.0,
) -> List[Paper]:
    """Attempt to download and extract full text for each paper in-place.

    Papers that already have full_text (e.g. from Semantic Scholar PDF URL)
    are skipped. Papers where extraction fails retain their abstract.

    Args:
        papers:     list to enrich (modified in-place, also returned)
        max_papers: cap for API politeness; None = all
        sleep:      seconds between requests

    Returns:
        The same list with .full_text populated where possible.
    """
    session = requests.Session()
    session.headers["User-Agent"] = "athanor/1.0 (open-source research tool)"

    targets = papers[:max_papers] if max_papers else papers
    succeeded = 0

    for paper in targets:
        # Skip papers that already have extracted full text
        if paper.full_text:
            continue

        # Determine URL: prefer explicit pdf_url (from S2), fall back to arXiv
        url = paper.pdf_url or arxiv_pdf_url(paper.arxiv_id)

        log.info("Fetching PDF for %s from %s", paper.arxiv_id, url[:60])
        pdf_bytes = fetch_pdf_bytes(url, session)

        if pdf_bytes:
            text = extract_pdf_text(pdf_bytes)
            if len(text) > 500:  # sanity check
                paper.full_text = text
                succeeded += 1
                log.info("✓ Extracted %d chars from %s", len(text), paper.arxiv_id)
            else:
                log.debug("Extraction too short for %s — keeping abstract", paper.arxiv_id)
                paper.full_text = None
        else:
            paper.full_text = None

        time.sleep(sleep)

    log.info("Full-text extraction: %d/%d succeeded", succeeded, len(targets))
    return papers
