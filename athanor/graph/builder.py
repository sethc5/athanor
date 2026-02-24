"""
athanor.graph.builder — assemble per-paper extractions into a unified
concept graph and compute graph metrics.

Merging strategy:
  1. Concepts with the same label (case-insensitive) are unified.
  2. Aliases are cross-referenced: if a concept from paper B uses a label
     that is an alias in paper A, they are merged.
  3. Semantic deduplication: concepts with embedding cosine-similarity ≥ 0.92
     are merged via union-find (catches abbreviations like CY3 ↔ Calabi-Yau
     threefold that the alias graph misses).
  4. Edge weights are accumulated across papers.
  5. Centrality is computed over the final merged graph.
"""
from __future__ import annotations

import json
import logging
import math
import re as _re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from athanor.config import cfg
from athanor.graph.extractor import ConceptExtractor
from athanor.graph.models import Concept, ConceptGraph, Edge

log = logging.getLogger(__name__)

# Pre-compute the set of valid Edge model field names (including aliases)
# so build_from_raw() can filter unknown keys without per-call overhead.
_EDGE_FIELDS: frozenset[str] = frozenset(
    f.alias or f_name for f_name, f in Edge.model_fields.items()
)


def normalize_label(s: str) -> str:
    """Canonical dedup key: lowercase, collapse hyphens/underscores/spaces.

    'Calabi-Yau three-fold' and 'Calabi-Yau threefold' → same key.
    Exported so tests can import and verify the same logic the merge uses.
    """
    return _re.sub(r"[-_\s]+", "", s).lower()


class GraphBuilder:
    """Orchestrates extraction across a corpus and merges results."""

    def __init__(self, extractor: Optional[ConceptExtractor] = None) -> None:
        self._extractor = extractor  # lazily resolved on first build() call
        self._extractor_override = extractor is not None

    # ── public ───────────────────────────────────────────────────────────────

    def build(
        self,
        parsed_papers: List[dict],
        domain: str = "general",
        query: str = "",
        save_path: Optional[Path] = None,
        max_workers: int = 2,
        domain_context: str = "",
    ) -> ConceptGraph:
        """Extract and merge concept graphs from *parsed_papers*.

        Args:
            parsed_papers: output of athanor.ingest.parse_papers()
            domain:        human-readable domain tag (for metadata)
            query:         original search query (for metadata)
            save_path:     optional JSON path to persist the graph

        Returns:
            A fully merged and annotated ConceptGraph.
        """
        # Lazily initialize the extractor so GraphBuilder() can be constructed
        # without an API key (useful for offline/unit tests of graph utilities).
        if self._extractor is None:
            self._extractor = ConceptExtractor()

        all_concepts: List[Concept] = []
        all_edges: List[Edge] = []

        def _extract_one(paper: dict):
            log.info("Processing paper: %s — %s", paper["arxiv_id"], paper["title"][:60])
            return self._extractor.extract(
                text=paper["text"],
                arxiv_id=paper["arxiv_id"],
                domain_context=domain_context,
            )

        # Parallelise Claude extraction across papers (I/O-bound, safe to thread).
        # Default 8 workers: at 3000 max_tokens each that's ~24K OPM burst,
        # comfortably under the 90K Haiku OPM Tier-1 limit.
        _workers = min(max_workers, len(parsed_papers))
        with ThreadPoolExecutor(max_workers=_workers) as pool:
            results = list(pool.map(_extract_one, parsed_papers))

        for concepts, edges in results:
            all_concepts.extend(concepts)
            all_edges.extend(edges)

        merged_concepts, merged_edges, alias_map = self._merge(
            all_concepts, all_edges
        )
        # Semantic near-duplicate merge — catches label variants the alias
        # graph misses (e.g. "CY3" ↔ "Calabi-Yau threefold")
        merged_concepts, merged_edges = self._semantic_dedup(merged_concepts, merged_edges)
        graph = ConceptGraph(
            domain=domain,
            query=query,
            concepts=merged_concepts,
            edges=merged_edges,
        )
        self._compute_centrality(graph)
        self._compute_structural_holes(graph)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                graph.model_dump_json(indent=2), encoding="utf-8"
            )
            log.info("Saved concept graph to %s", save_path)

        return graph

    # ── private ──────────────────────────────────────────────────────────────

    def build_from_raw(
        self,
        raw: dict,
        domain: str = "general",
        query: str = "",
    ) -> ConceptGraph:
        """Build and annotate a ConceptGraph from a raw dict (no API needed).

        Args:
            raw: dict with ``concepts`` (list of dicts with at least ``label``
                 and ``description``) and ``edges`` (list of dicts with
                 ``source``, ``target``, and optional ``weight``).

        Returns:
            A ConceptGraph with centrality and structural-hole fields populated.
        """
        concepts = [Concept(**c) for c in raw.get("concepts", [])]
        edges = []
        for e in raw.get("edges", []):
            filtered = {k: v for k, v in e.items() if k in _EDGE_FIELDS}
            filtered.setdefault("relation", "related_to")
            edges.append(Edge(**filtered))
        graph = ConceptGraph(
            domain=domain, query=query, concepts=concepts, edges=edges
        )
        self._compute_centrality(graph)
        self._compute_structural_holes(graph)
        return graph

    # ── semantic deduplication ────────────────────────────────────────────

    def _semantic_dedup(
        self,
        concepts: List[Concept],
        edges: List[Edge],
        threshold: float = 0.88,
    ) -> Tuple[List[Concept], List[Edge]]:
        """Merge near-duplicate concepts using embedding cosine similarity.

        After string-based alias merging, concepts like “Calabi-Yau threefold”,
        “CY3”, and “Calabi-Yau three-fold” still land as separate nodes when
        neither Claude-provided alias covers the other form.  This pass uses
        sentence embeddings (all-MiniLM-L6-v2, mean-centered) to find
        near-duplicates at cosine-sim ≥ threshold and merges them via
        union-find.

        threshold=0.88 is calibrated on raw (uncentered) all-MiniLM-L6-v2
        embeddings where hyphenation variants score ~0.90 and
        threefold ↔ fourfold scores ~0.77 (safely below merge threshold).
        """
        from sklearn.metrics.pairwise import cosine_similarity as _csim
        from athanor.embed import Embedder

        n = len(concepts)
        if n < 2:
            return concepts, edges

        embedder = Embedder()
        texts = [f"{c.label}. {c.description[:200]}" for c in concepts]
        # Use raw (uncentered) embeddings: mean-centering is great for gap
        # detection but compresses near-duplicate pairs below any useful
        # threshold (even "Calabi-Yau threefold" vs "CY3" scores ~0.63 centred).
        embs = embedder.embed(texts, center=False)
        sim = _csim(embs)

        # Union-Find — canonical = most source_papers; tie-break = label sort
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            cx, cy = concepts[px], concepts[py]
            # Root = most source_papers; alphabetical tie-break
            if (len(cy.source_papers), cx.label) > (len(cx.source_papers), cy.label):
                parent[px] = py
            else:
                parent[py] = px

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        # Recompute roots after all unions (path compression may shift them)
        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        remap: Dict[str, str] = {}     # non-canonical label → canonical label
        merged_concepts: List[Concept] = []

        for root, members in clusters.items():
            base = concepts[root]
            for idx in members:
                if idx == root:
                    continue
                other = concepts[idx]
                remap[other.label] = base.label
                self._merge_concept(base, other)
                if other.label not in base.aliases and other.label != base.label:
                    base.aliases.append(other.label)
                log.debug(
                    "Semantic dedup: '%s' → '%s' (sim=%.3f)",
                    other.label, base.label, float(sim[root, idx]),
                )
            merged_concepts.append(base)

        if remap:
            log.info(
                "Semantic dedup merged %d duplicate concept(s) out of %d",
                len(remap), n,
            )

        # Remap edge endpoints
        canon_labels = {c.label for c in merged_concepts}
        deduped: Dict[Tuple[str, str, str], Edge] = {}
        for e in edges:
            src = remap.get(e.source, e.source)
            tgt = remap.get(e.target, e.target)
            if src == tgt:
                continue
            if src not in canon_labels or tgt not in canon_labels:
                continue
            key = (src, tgt, e.relation)
            if key in deduped:
                ex = deduped[key]
                ex.weight = min(1.0, ex.weight + e.weight * 0.5)
                ex.source_papers = list(set(ex.source_papers + e.source_papers))
            else:
                deduped[key] = Edge(
                    source=src, target=tgt, relation=e.relation,
                    weight=e.weight, evidence=e.evidence,
                    source_papers=e.source_papers, edge_type=e.edge_type,
                )

        return merged_concepts, list(deduped.values())

    def _merge(
        self,
        concepts: List[Concept],
        edges: List[Edge],
    ) -> Tuple[List[Concept], List[Edge], Dict[str, str]]:
        """Unify duplicate concepts and remap edge endpoints."""

        # Build label → canonical label map (norm-key and case-insensitive)
        canon: Dict[str, str] = {}   # raw label → canonical label
        merged: Dict[str, Concept] = {}  # canonical label → Concept

        for c in concepts:
            key = c.label.lower()
            norm_key = normalize_label(c.label)
            # Check if any alias already registered as a canonical concept.
            # Try both raw-lowercase and normalised key (catches hyphenation).
            resolved = None
            for alias in [c.label] + c.aliases:
                for variant in (alias.lower(), normalize_label(alias)):
                    if variant in canon:
                        resolved = canon[variant]
                        break
                if resolved:
                    break

            if resolved is None:
                # New concept — register both key forms
                canon[key] = c.label
                canon[norm_key] = c.label
                for alias in c.aliases:
                    canon[alias.lower()] = c.label
                    canon[normalize_label(alias)] = c.label
                if c.label not in merged:
                    merged[c.label] = c
                else:
                    self._merge_concept(merged[c.label], c)
            else:
                # Merge into existing canonical concept
                self._merge_concept(merged[resolved], c)
                canon[key] = resolved
                canon[norm_key] = resolved

        # Remap edge endpoints to canonical labels (try normalised key too)
        merged_edges: Dict[Tuple[str, str, str], Edge] = {}
        for e in edges:
            src = canon.get(e.source.lower()) or canon.get(normalize_label(e.source), e.source)
            tgt = canon.get(e.target.lower()) or canon.get(normalize_label(e.target), e.target)
            if src == tgt:
                continue  # skip self-loops
            if src not in merged or tgt not in merged:
                log.debug("Dropping edge (%s→%s): unknown concept", src, tgt)
                continue
            key = (src, tgt, e.relation)
            if key in merged_edges:
                existing = merged_edges[key]
                existing.weight = min(1.0, existing.weight + e.weight * 0.5)
                existing.source_papers = list(
                    set(existing.source_papers + e.source_papers)
                )
            else:
                merged_edges[key] = Edge(
                    source=src,
                    target=tgt,
                    relation=e.relation,
                    edge_type=e.edge_type,
                    weight=e.weight,
                    evidence=e.evidence,
                    source_papers=e.source_papers,
                )

        return list(merged.values()), list(merged_edges.values()), canon

    def _merge_concept(self, base: Concept, other: Concept) -> None:
        """Accumulate metadata from *other* into *base* in-place."""
        if not base.description and other.description:
            base.description = other.description
        for alias in other.aliases:
            if alias not in base.aliases and alias != base.label:
                base.aliases.append(alias)
        for pid in other.source_papers:
            if pid not in base.source_papers:
                base.source_papers.append(pid)

    def _compute_centrality(self, graph: ConceptGraph) -> None:
        """Annotate each Concept with its betweenness centrality score."""
        G = graph.to_networkx()
        if len(G.nodes) < 2:
            return
        centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
        label_map = {c.label: c for c in graph.concepts}
        for label, score in centrality.items():
            if label in label_map:
                label_map[label].centrality = round(score, 4)

    def _compute_structural_holes(self, graph: ConceptGraph) -> None:  # noqa: C901
        """Annotate each Concept with Burt's structural hole metrics.

        Burt constraint (C_i): how redundant a node's contacts are.
          0 = pure broker straddling otherwise disconnected clusters
          1 = fully embedded — all contacts know each other

        Effective size: number of non-redundant contacts in ego network.
          High = spans many disconnected clusters (good broker signal).

        A node is flagged as `structural_hole=True` when:
          - burt_constraint < 0.5  (low redundancy in neighbourhood)
          - centrality         > 0.05 (appears on many shortest paths)
        """
        G = graph.to_networkx()
        if len(G.nodes) < 3:
            return

        label_map = {c.label: c for c in graph.concepts}

        try:
            constraint = nx.constraint(G, weight="weight")
            for label, c_score in constraint.items():
                if label in label_map:
                    val = float(c_score)
                    # NaN means isolated node — treat as fully embedded (no hole)
                    label_map[label].burt_constraint = round(1.0 if math.isnan(val) else val, 4)
        except Exception:
            log.debug("Burt constraint computation failed — skipping")

        try:
            eff_size = nx.effective_size(G, weight="weight")
            for label, eff in eff_size.items():
                if label in label_map and eff is not None:
                    val = float(eff)
                    # NaN / None means degree-0/1 node — treat as single contact
                    label_map[label].effective_size = round(1.0 if math.isnan(val) else val, 4)
        except Exception:
            log.debug("Effective size computation failed — skipping")

        # Flag structural hole brokers
        for c in graph.concepts:
            c.structural_hole = (
                c.burt_constraint < 0.5 and c.centrality > 0.05
            )
