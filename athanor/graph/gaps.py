"""
athanor.graph.gaps — compute candidate structural gaps from a ConceptGraph.

A candidate gap is a pair of concepts that are semantically close
(high embedding cosine similarity) but structurally distant in the
literature graph (shortest path > 2).  These are the raw inputs for
Stage 2 gap analysis.

Moved here from pipeline.py because it is graph-analysis logic, not
pipeline orchestration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from athanor.embed import Embedder
from athanor.graph.models import ConceptGraph

log = logging.getLogger(__name__)


def compute_candidate_gaps(
    concept_graph: ConceptGraph,
    threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """Compute candidate gaps from a ConceptGraph.

    A candidate gap is a concept pair that is semantically close (cosine
    similarity >= *threshold*) but graph-distant (shortest path > 2).

    Returns a list of dicts sorted by similarity descending.
    """
    G = concept_graph.to_networkx()
    embedder = Embedder()

    concept_texts = [
        c.label + ". " + c.description for c in concept_graph.concepts
    ]
    concept_embs = embedder.embed(concept_texts)
    csim = cosine_similarity(concept_embs)

    labels = [c.label for c in concept_graph.concepts]
    concept_list = concept_graph.concepts
    candidate_gaps: List[Dict[str, Any]] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if csim[i, j] < threshold:
                continue
            try:
                dist = nx.shortest_path_length(G, labels[i], labels[j])
            except nx.NetworkXNoPath:
                dist = 999
            if dist > 2:
                sh_a = 1.0 - concept_list[i].burt_constraint
                sh_b = 1.0 - concept_list[j].burt_constraint
                sh_score = round((sh_a + sh_b) / 2, 4)
                candidate_gaps.append({
                    "concept_a": labels[i],
                    "concept_b": labels[j],
                    "similarity": float(csim[i, j]),
                    "graph_distance": dist,
                    "structural_hole_score": sh_score,
                })

    candidate_gaps.sort(key=lambda x: x["similarity"], reverse=True)
    log.info("Found %d candidate gaps (threshold=%.2f)", len(candidate_gaps), threshold)
    return candidate_gaps
