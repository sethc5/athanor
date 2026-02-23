"""
athanor.ingest.arxiv_client — fetch papers from the arXiv API.

Design goals:
- Domain-agnostic: accepts any query string.
- Returns rich Paper objects, not raw API types.
- Cacheable: writes JSON to data/raw/ so repeated runs are free.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import time

import arxiv

from athanor.config import cfg

log = logging.getLogger(__name__)


@dataclass
class Paper:
    """Normalised representation of a single paper.

    Intentionally flat and serialisable — this is the boundary object
    between the ingest layer and everything downstream.
    """
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: str          # ISO-8601 date string
    url: str
    full_text: Optional[str] = None  # populated by parser if available

    @property
    def digest(self) -> str:
        """The text we actually reason over — abstract now, full text later."""
        return self.full_text or self.abstract

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(**d)


class ArxivClient:
    """Thin wrapper around the arxiv library with local caching.

    Usage:
        client = ArxivClient()
        papers = client.fetch("information theory entropy", max_results=15)
    """

    _SORT_MAP = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = cache_dir or cfg.data_raw
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────────

    def fetch(
        self,
        query: str,
        max_results: int = cfg.arxiv_max_results,
        sort_by: str = cfg.arxiv_sort_by,
        use_cache: bool = True,
    ) -> List[Paper]:
        """Fetch papers for *query*, returning a list of Paper objects.

        Results are cached to data/raw/<slug>.json so the API is only
        hit once per unique query.
        """
        cache_path = self._cache_path(query)
        if use_cache and cache_path.exists():
            log.info("Loading cached results from %s", cache_path)
            return self._load_cache(cache_path)

        log.info("Fetching arXiv results for query: %r", query)
        criterion = self._SORT_MAP.get(sort_by, arxiv.SortCriterion.Relevance)

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=criterion,
        )

        papers: List[Paper] = []
        for result in client.results(search):
            paper = Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[str(a) for a in result.authors],
                categories=result.categories,
                published=result.published.date().isoformat(),
                url=result.entry_id,
            )
            papers.append(paper)

        log.info("Fetched %d papers", len(papers))
        self._save_cache(cache_path, papers)
        return papers

    # ── private ──────────────────────────────────────────────────────────────

    def _cache_path(self, query: str) -> Path:
        slug = query.lower().replace(" ", "_")[:60]
        return self._cache_dir / f"arxiv_{slug}.json"

    def _save_cache(self, path: Path, papers: List[Paper]) -> None:
        path.write_text(
            json.dumps([p.to_dict() for p in papers], indent=2),
            encoding="utf-8",
        )
        log.info("Saved %d papers to cache: %s", len(papers), path)

    def _load_cache(self, path: Path) -> List[Paper]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Paper.from_dict(d) for d in data]

    def fetch_by_ids(self, arxiv_ids: List[str]) -> List[Paper]:
        """Fetch specific papers by their arXiv IDs (no cache — deterministic).

        IDs can be bare (``2309.12345``) or include the version (``2309.12345v2``).
        """
        if not arxiv_ids:
            return []
        log.info("Fetching %d papers by arXiv ID", len(arxiv_ids))
        client = arxiv.Client()
        search = arxiv.Search(id_list=arxiv_ids)
        papers: List[Paper] = []
        for result in client.results(search):
            papers.append(Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[str(a) for a in result.authors],
                categories=result.categories,
                published=result.published.date().isoformat(),
                url=result.entry_id,
            ))
        log.info("Fetched %d papers by ID", len(papers))
        return papers
