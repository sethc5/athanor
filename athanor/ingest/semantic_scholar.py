"""
athanor.ingest.semantic_scholar — fetch papers from the Semantic Scholar API.

Complements the arXiv client. Better citation counts, open-access full text
links, and stronger coverage for biology/medicine.

API docs: https://api.semanticscholar.org/graph/v1
No API key required for low-volume use; set S2_API_KEY in .env for higher limits.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import requests

from athanor.config import cfg
from athanor.ingest.arxiv_client import Paper  # reuse the same Paper model

log = logging.getLogger(__name__)

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,"
    "publicationDate,fieldsOfStudy,openAccessPdf,citationCount"
)


class SemanticScholarClient:
    """Thin wrapper around the Semantic Scholar Graph API with local caching.

    Usage:
        client = SemanticScholarClient()
        papers = client.fetch("information theory entropy", max_results=15)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._cache_dir = cache_dir or cfg.data_raw
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key or cfg.s2_api_key
        self._session = requests.Session()
        if self._api_key:
            self._session.headers["x-api-key"] = self._api_key

    # ── public ───────────────────────────────────────────────────────────────

    def fetch(
        self,
        query: str,
        max_results: int = cfg.arxiv_max_results,
        use_cache: bool = True,
        open_access_only: bool = False,
    ) -> List[Paper]:
        """Search Semantic Scholar for *query*, return Paper objects."""
        cache_path = self._cache_path("s2", query)
        if use_cache and cache_path.exists():
            log.info("Loading S2 cached results from %s", cache_path)
            return self._load_cache(cache_path)

        log.info("Fetching Semantic Scholar results for query: %r", query)
        papers: List[Paper] = []
        offset = 0
        batch = 100

        while len(papers) < max_results:
            resp = self._session.get(
                f"{_BASE}/paper/search",
                params={
                    "query": query,
                    "fields": _FIELDS,
                    "limit": min(batch, max_results - len(papers)),
                    "offset": offset,
                    "openAccessPdf": open_access_only or "",
                },
                timeout=30,
            )
            if resp.status_code == 429:
                log.warning("S2 rate limit — sleeping 10s")
                time.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                paper = self._to_paper(item)
                if paper:
                    papers.append(paper)

            offset += len(items)
            if len(items) < batch:
                break
            time.sleep(0.2)

        log.info("Fetched %d S2 papers", len(papers))
        self._save_cache(cache_path, papers)
        return papers

    def fetch_by_arxiv_ids(self, arxiv_ids: List[str]) -> List[Paper]:
        """Enrich arXiv papers with S2 metadata (cite counts, open-access PDF)."""
        results = []
        for aid in arxiv_ids:
            try:
                resp = self._session.get(
                    f"{_BASE}/paper/arXiv:{aid}",
                    params={"fields": _FIELDS},
                    timeout=15,
                )
                if resp.status_code == 404:
                    log.debug("arXiv:%s not in S2", aid)
                    continue
                resp.raise_for_status()
                paper = self._to_paper(resp.json())
                if paper:
                    results.append(paper)
                time.sleep(0.1)
            except requests.RequestException as exc:
                log.warning("S2 lookup failed for %s: %s", aid, exc)
        return results

    # ── private ──────────────────────────────────────────────────────────────

    def _to_paper(self, item: dict) -> Optional[Paper]:
        title = item.get("title", "").strip()
        abstract = item.get("abstract") or ""
        if not title or not abstract:
            return None

        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv", item.get("paperId", ""))
        pub_date = item.get("publicationDate") or str(item.get("year", ""))
        authors = [a.get("name", "") for a in (item.get("authors") or [])]
        cats = item.get("fieldsOfStudy") or []

        # Store open-access PDF URL in full_text slot temporarily
        oa = item.get("openAccessPdf") or {}
        pdf_url = oa.get("url", "")

        return Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=cats,
            published=pub_date[:10] if len(pub_date) >= 10 else pub_date,
            url=f"https://www.semanticscholar.org/paper/{item.get('paperId','')}",
            full_text=pdf_url or None,  # repurpose field for PDF URL
        )

    def _cache_path(self, prefix: str, query: str) -> Path:
        slug = query.lower().replace(" ", "_")[:60]
        return self._cache_dir / f"{prefix}_{slug}.json"

    def _save_cache(self, path: Path, papers: List[Paper]) -> None:
        path.write_text(
            json.dumps([p.to_dict() for p in papers], indent=2),
            encoding="utf-8",
        )

    def _load_cache(self, path: Path) -> List[Paper]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Paper.from_dict(d) for d in data]
