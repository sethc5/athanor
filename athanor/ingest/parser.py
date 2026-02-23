"""
athanor.ingest.parser — convert raw Paper objects into analysable text.

Currently uses abstract + metadata. Structured to accept full PDF text
when that pipeline is added (e.g. via pypdf or GROBID).
"""
from __future__ import annotations

import re
from typing import List

from athanor.ingest.arxiv_client import Paper


def clean_text(text: str) -> str:
    """Normalise whitespace and strip arXiv LaTeX artefacts.

    Math expressions ($...$) are kept as-is so downstream models
    (Claude, embedders) can use the mathematical content.
    Only non-math LaTeX commands like \\textbf{...} are stripped.
    """
    # collapse newlines and extra spaces
    text = re.sub(r"\s+", " ", text)
    # strip non-math LaTeX commands that survive into abstracts
    # (keep $...$ math expressions intact)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    return text.strip()


def paper_to_text(paper: Paper) -> str:
    """Produce a single text blob suitable for concept extraction.

    When full_text is available it is preferred; otherwise abstract.
    The title is always prepended so concept extractors have context.
    """
    body = paper.full_text if paper.full_text else paper.abstract
    return f"Title: {paper.title}\n\nAbstract: {clean_text(body)}"


def parse_papers(papers: List[Paper]) -> List[dict]:
    """Return a list of dicts ready for embedding and graph extraction.

    Each dict carries the paper identity alongside its cleaned text,
    so downstream steps never need to re-join.
    """
    results = []
    for p in papers:
        results.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "categories": p.categories,
                "published": p.published,
                "url": p.url,
                "text": paper_to_text(p),
            }
        )
    return results
