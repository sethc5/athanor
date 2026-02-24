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

    Math expressions (``$...$`` and ``$$...$$``) are kept as-is so downstream
    models (Claude, embedders) can use the mathematical content.
    Only non-math LaTeX commands outside of math delimiters are stripped.
    """
    # collapse newlines and extra spaces
    text = re.sub(r"\s+", " ", text)
    # split on both $$...$$ (display) and $...$ (inline) math,
    # trying $$ first so it doesn't get consumed as two $'s
    parts = re.split(r"(\$\$[^$]+\$\$|\$[^$]+\$)", text)
    cleaned = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside math delimiters — keep as-is
            cleaned.append(part)
        else:
            # Outside math — strip LaTeX commands
            cleaned.append(re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", part))
    return "".join(cleaned).strip()


def paper_to_text(paper: Paper) -> str:
    """Produce a single text blob suitable for concept extraction.

    When full_text is available it is preferred; otherwise abstract.
    The title is always prepended so concept extractors have context.
    """
    if paper.full_text:
        return f"Title: {paper.title}\n\nFull text:\n{clean_text(paper.full_text)}"
    return f"Title: {paper.title}\n\nAbstract: {clean_text(paper.abstract)}"


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
