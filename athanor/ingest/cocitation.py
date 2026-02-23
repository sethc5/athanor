"""Bibliographic coupling (co-citation seeding).

Two papers that cite many of the same references are likely topically related
even if they never directly cite each other.  This module computes a Jaccard-
normalised coupling score for all paper pairs that share ≥ 2 references.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def compute_biblio_coupling(
    paper_references: Dict[str, List[str]],
    min_shared: int = 2,
) -> Dict[Tuple[str, str], float]:
    """Return a dict of ``(paper_id_a, paper_id_b) → Jaccard coupling score``.

    Parameters
    ----------
    paper_references:
        Mapping of ``{paper_id: [cited_paper_id, …]}``.
    min_shared:
        Minimum number of shared references required to emit a pair.

    Returns
    -------
    scores dict, keys are *sorted* 2-tuples so each pair appears once.
    """
    papers = list(paper_references)
    ref_sets: Dict[str, frozenset] = {
        pid: frozenset(refs) for pid, refs in paper_references.items()
    }

    scores: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(papers):
        ra = ref_sets[a]
        if not ra:
            continue
        for b in papers[i + 1 :]:
            rb = ref_sets[b]
            if not rb:
                continue
            shared = len(ra & rb)
            if shared >= min_shared:
                union = len(ra | rb)
                score = round(shared / union, 4) if union else 0.0
                key = (a, b) if a < b else (b, a)
                scores[key] = score

    return scores


def top_coupled_pairs(
    scores: Dict[Tuple[str, str], float],
    n: int = 50,
) -> List[Tuple[Tuple[str, str], float]]:
    """Return the top-*n* coupled paper pairs sorted by descending score."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
