"""
athanor.gaps.dedup — cluster semantically similar candidate gaps before
sending them to Claude, preventing redundant analyses.

Strategy:
  1. Embed each gap as "concept_a + concept_b" text
  2. Cluster by cosine similarity with a greedy threshold algorithm
  3. Keep one representative per cluster (highest embedding similarity to centroid)
  4. Return deduplicated list and a cluster membership map

Domain-agnostic: no field-specific logic.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Dict

import numpy as np

from athanor.gaps.models import CandidateGap

log = logging.getLogger(__name__)


def deduplicate_gaps(
    gaps: List[CandidateGap],
    threshold: float = 0.85,
    embedder=None,
) -> Tuple[List[CandidateGap], Dict[int, List[int]]]:
    """Remove near-duplicate concept pairs.

    Two gaps are considered duplicates if the cosine similarity of their
    embedding (label + description text) exceeds *threshold*.

    Args:
        gaps:      candidate gaps to deduplicate
        threshold: cosine similarity cutoff for merging (0–1)
        embedder:  an athanor.embed.Embedder instance;  if None one is created

    Returns:
        (deduplicated_gaps, cluster_map)
        cluster_map: {representative_index → [member_indices]}
    """
    if len(gaps) <= 1:
        return gaps, {0: [0]}

    if embedder is None:
        from athanor.embed import Embedder
        embedder = Embedder()

    texts = [
        f"{g.concept_a} {g.description_a} — {g.concept_b} {g.description_b}"
        for g in gaps
    ]
    vecs = embedder.embed(texts)  # (N, D)

    # Greedy clustering
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(vecs)

    assigned: List[int] = [-1] * len(gaps)  # cluster id per gap
    clusters: Dict[int, List[int]] = {}      # cluster_id → member indices
    cluster_id = 0

    for i in range(len(gaps)):
        if assigned[i] != -1:
            continue
        clusters[i] = [i]
        assigned[i] = i
        for j in range(i + 1, len(gaps)):
            if assigned[j] != -1:
                continue
            if sim[i, j] >= threshold:
                clusters[i].append(j)
                assigned[j] = i

    representatives = list(clusters.keys())
    deduped = [gaps[r] for r in representatives]

    n_removed = len(gaps) - len(deduped)
    log.info(
        "Gap deduplication: %d → %d (removed %d near-duplicates, threshold=%.2f)",
        len(gaps), len(deduped), n_removed, threshold,
    )
    return deduped, clusters
