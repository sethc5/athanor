"""Tests for ingest-layer fixes: cache slugs, parallel PDF enrichment."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from athanor.ingest.arxiv_client import ArxivClient, Paper
from athanor.ingest.semantic_scholar import SemanticScholarClient
from athanor.ingest.pdf import enrich_papers_with_fulltext, _fetch_one


# ── cache slug collision resistance ──────────────────────────────────────────

class TestCacheSlugCollision:
    """_cache_path must produce distinct filenames for queries that share a
    long common prefix."""

    def test_arxiv_different_long_queries(self, tmp_path: Path):
        client = ArxivClient(cache_dir=tmp_path)
        prefix = "a" * 80
        p1 = client._cache_path(prefix + " alpha")
        p2 = client._cache_path(prefix + " beta")
        assert p1 != p2, "Distinct queries must not collide"

    def test_arxiv_slug_contains_hash(self, tmp_path: Path):
        client = ArxivClient(cache_dir=tmp_path)
        path = client._cache_path("some query")
        h = hashlib.sha256("some query".encode()).hexdigest()[:12]
        assert h in path.name

    def test_s2_different_long_queries(self, tmp_path: Path):
        client = SemanticScholarClient(cache_dir=tmp_path)
        prefix = "b" * 80
        p1 = client._cache_path("s2", prefix + " one")
        p2 = client._cache_path("s2", prefix + " two")
        assert p1 != p2

    def test_s2_slug_contains_hash(self, tmp_path: Path):
        client = SemanticScholarClient(cache_dir=tmp_path)
        path = client._cache_path("s2", "another query")
        h = hashlib.sha256("another query".encode()).hexdigest()[:12]
        assert h in path.name

    def test_case_insensitive_same_slug(self, tmp_path: Path):
        client = ArxivClient(cache_dir=tmp_path)
        assert client._cache_path("Hello World") == client._cache_path("hello world")


# ── parallel PDF enrichment ──────────────────────────────────────────────────

def _make_paper(arxiv_id: str, **kw) -> Paper:
    defaults = dict(
        arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", abstract="Abstract",
        authors=[], categories=[], published="2024-01-01", url="http://x",
    )
    defaults.update(kw)
    return Paper(**defaults)


class TestParallelPdfEnrichment:
    def test_skips_already_enriched(self):
        p = _make_paper("1", full_text="already here")
        result = enrich_papers_with_fulltext([p], max_workers=2)
        assert result[0].full_text == "already here"

    @patch("athanor.ingest.pdf.fetch_pdf_bytes", return_value=None)
    def test_sets_none_on_failure(self, mock_fetch):
        p = _make_paper("2")
        result = enrich_papers_with_fulltext([p], sleep=0, max_workers=1)
        assert result[0].full_text is None

    @patch("athanor.ingest.pdf.extract_pdf_text", return_value="x" * 600)
    @patch("athanor.ingest.pdf.fetch_pdf_bytes", return_value=b"fake-pdf")
    def test_populates_full_text(self, mock_fetch, mock_extract):
        p = _make_paper("3")
        result = enrich_papers_with_fulltext([p], sleep=0, max_workers=1)
        assert result[0].full_text is not None
        assert len(result[0].full_text) == 600

    @patch("athanor.ingest.pdf.extract_pdf_text", return_value="x" * 600)
    @patch("athanor.ingest.pdf.fetch_pdf_bytes", return_value=b"fake-pdf")
    def test_parallel_multiple_papers(self, mock_fetch, mock_extract):
        papers = [_make_paper(str(i)) for i in range(5)]
        result = enrich_papers_with_fulltext(papers, sleep=0, max_workers=3)
        assert all(p.full_text is not None for p in result)
        assert len(result) == 5

    def test_max_papers_limit(self):
        papers = [_make_paper(str(i)) for i in range(10)]
        with patch("athanor.ingest.pdf.fetch_pdf_bytes", return_value=None):
            result = enrich_papers_with_fulltext(papers, max_papers=3, sleep=0)
        # Only first 3 should have been attempted
        assert len(result) == 10  # original list returned


# ── _fetch_one helper ────────────────────────────────────────────────────────

class TestFetchOne:
    def test_skip_if_already_has_text(self):
        p = _make_paper("x", full_text="existing")
        paper, text = _fetch_one(p, sleep=0)
        assert text == "existing"

    @patch("athanor.ingest.pdf.fetch_pdf_bytes", return_value=None)
    def test_returns_none_on_download_failure(self, _):
        p = _make_paper("y")
        paper, text = _fetch_one(p, sleep=0)
        assert text is None
