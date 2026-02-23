"""
athanor.graph.models — data structures for the concept graph.

All models are Pydantic v2 for validation + easy JSON serialisation.
Domain-agnostic: no information-theory-specific fields here.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Concept(BaseModel):
    """A node in the concept graph.

    Label is the canonical name. Aliases capture variant spellings.
    Source papers track provenance.
    """
    label: str
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    source_papers: List[str] = Field(default_factory=list)  # arxiv IDs
    centrality: float = 0.0      # betweenness centrality (GraphBuilder)
    # Structural hole metrics (Burt)
    burt_constraint: float = 1.0    # 0=pure broker, 1=fully embedded cluster
    effective_size: float = 0.0     # non-redundant contacts in ego network
    structural_hole: bool = False   # True if key broker between clusters


class Edge(BaseModel):
    """A directed or undirected relationship between two concepts."""
    source: str          # concept label
    target: str          # concept label
    relation: str        # e.g. "extends", "depends_on", "contradicts"
    weight: float = 1.0  # strength of the relationship
    evidence: str = ""   # excerpt supporting this edge
    source_papers: List[str] = Field(default_factory=list)


class ConceptGraph(BaseModel):
    """The full concept graph for a corpus."""
    domain: str = "general"
    query: str = ""
    concepts: List[Concept] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    # ── convenience ──────────────────────────────────────────────────────────

    def concept_labels(self) -> List[str]:
        return [c.label for c in self.concepts]

    def to_networkx(self):
        """Return a networkx.Graph for analysis and visualization."""
        import networkx as nx
        G = nx.Graph()
        for c in self.concepts:
            G.add_node(
                c.label,
                description=c.description,
                centrality=c.centrality,
                source_papers=c.source_papers,
            )
        for e in self.edges:
            if G.has_edge(e.source, e.target):
                # accumulate weight on duplicate edges
                G[e.source][e.target]["weight"] += e.weight
                G[e.source][e.target]["relations"].append(e.relation)
            else:
                G.add_edge(
                    e.source,
                    e.target,
                    weight=e.weight,
                    relation=e.relation,
                    relations=[e.relation],
                    evidence=e.evidence,
                    source_papers=e.source_papers,
                )
        return G

    def sparse_connections(self, top_k: int = 10) -> List[Edge]:
        """Return edges that connect otherwise distant graph clusters.

        A 'sparse connection' is an edge whose endpoints have low
        common-neighbour count relative to their total neighbourhoods —
        i.e. the bridge edges.  These are signal for Stage 2 gap-finding.
        """
        import networkx as nx
        G = self.to_networkx()
        edge_scores = []
        for u, v, data in G.edges(data=True):
            nu = set(G.neighbors(u))
            nv = set(G.neighbors(v))
            jaccard = len(nu & nv) / len(nu | nv) if (nu | nv) else 0.0
            edge_scores.append((jaccard, u, v))
        edge_scores.sort()  # ascending: lowest jaccard = most bridging
        sparse: List[Edge] = []
        edge_map = {(e.source, e.target): e for e in self.edges}
        edge_map.update({(e.target, e.source): e for e in self.edges})
        for _, u, v in edge_scores[:top_k]:
            if (u, v) in edge_map:
                sparse.append(edge_map[(u, v)])
        return sparse
