"""
athanor.gaps.models — data structures for Stage 2 gap analysis.

CandidateGap   : a pair of concepts flagged as structurally sparse
GapAnalysis    : Claude's structured assessment of one gap
GapReport      : the full ranked report over a corpus
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, computed_field


class CandidateGap(BaseModel):
    """A concept pair that is semantically close but graph-distant.

    Produced by Stage 1; read by Stage 2.
    """
    concept_a: str
    concept_b: str
    similarity: float          # cosine similarity of embeddings
    graph_distance: int        # shortest-path length; 999 = disconnected

    # Context injected at analysis time (not in Stage 1 JSON)
    description_a: str = ""
    description_b: str = ""
    shared_papers: List[str] = Field(default_factory=list)
    papers_a: List[str] = Field(default_factory=list)
    papers_b: List[str] = Field(default_factory=list)

    # Structural hole signal: average broker score of the two endpoints.
    # High = gap bridges two otherwise disconnected clusters (Burt framework).
    structural_hole_score: float = 0.0


class GapAnalysis(BaseModel):
    """Claude's structured analysis of a single candidate gap."""
    concept_a: str
    concept_b: str

    # The core research question
    research_question: str

    # Explanatory narrative
    why_unexplored: str        # why the community has missed this
    intersection_opportunity: str  # what work at this intersection could achieve

    # Methodology sketch
    methodology: str           # how you'd actually investigate this
    computational: bool        # True if it can be tested computationally

    # Scores (1–5 integers; 5 = highest)
    novelty: int        = Field(ge=1, le=5)
    tractability: int   = Field(ge=1, le=5)
    impact: int         = Field(ge=1, le=5)

    # Derived
    @computed_field
    @property
    def composite_score(self) -> float:
        """Weighted composite with bonuses for causal and structural-hole gaps.

        Base: impact × 0.40 + novelty × 0.35 + tractability × 0.25  (1–5 scale)
        Bonus:
          +0.50 for causal gaps (directional mechanistic link)
          +0.25 for integrative gaps (cross-scale / cross-field)
          +0.25 if structural_hole_score > 0.5 (Burt broker gap)
        Capped at 5.0.
        """
        base = self.impact * 0.40 + self.novelty * 0.35 + self.tractability * 0.25
        bonus = 0.0
        if self.bridge_type == "causal":
            bonus += 0.50
        elif self.bridge_type == "integrative":
            bonus += 0.25
        if self.structural_hole_score > 0.5:
            bonus += 0.25
        return round(min(5.0, base + bonus), 3)

    # Gap classification — set by GapFinder
    # causal        : a missing *mechanistic* link (A causes B or vice versa)
    # methodological: gap is a missing *tool* or *technique* bridge
    # semantic      : conceptual proximity not yet formalised in literature
    # integrative   : gap bridges two different sub-fields or scales
    bridge_type: str = "semantic"

    # Tags for downstream filtering
    keywords: List[str] = Field(default_factory=list)

    # Provenance
    similarity: float = 0.0
    graph_distance: int = 999
    structural_hole_score: float = 0.0

    # Human review — set in notebook or CLI before Stage 3
    approved: Optional[bool] = None  # None = unreviewed; True = approved; False = rejected


class GapReport(BaseModel):
    """Full ranked gap report over a corpus — Stage 3 input."""
    domain: str
    query: str
    n_candidates: int = 0
    n_analyzed: int = 0
    analyses: List[GapAnalysis] = Field(default_factory=list)

    @property
    def ranked(self) -> List[GapAnalysis]:
        """Return analyses sorted by composite_score descending."""
        return sorted(self.analyses, key=lambda a: a.composite_score, reverse=True)

    def top(self, n: int = 5) -> List[GapAnalysis]:
        """Return the top-*n* analyses by composite score."""
        return self.ranked[:n]
