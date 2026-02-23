"""
athanor.hypotheses.models — data structures for Stage 3.

Hypothesis       : a single falsifiable claim derived from a gap
ExperimentDesign : how to test a hypothesis
HypothesisReport : ranked collection of hypotheses → final Stage 3 output
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, computed_field


class ExperimentDesign(BaseModel):
    """A concrete experiment proposal for testing a hypothesis."""

    # What to do
    approach: str          # high-level strategy
    steps: List[str]       # ordered concrete steps
    tools: List[str]       # specific methods, datasets, libraries

    # Feasibility
    computational: bool    # True → can run now; False → needs wet lab/field
    estimated_effort: str  # e.g. "1-2 weeks compute", "6-month wet lab study"
    data_requirements: str # what data/resources are needed

    # What success looks like
    expected_positive: str  # what result confirms the hypothesis
    expected_negative: str  # what result refutes it
    null_hypothesis: str    # the formal H₀
    statistical_test: str = ""   # e.g. "two-sided t-test, alpha=0.05, power=0.80"

    # Flag any validation gaps
    limitations: List[str] = Field(default_factory=list)
    requires_followup: Optional[str] = None  # what computational result needs wet lab to confirm


class Hypothesis(BaseModel):
    """A falsifiable scientific hypothesis derived from a research gap."""

    # Origin
    gap_concept_a: str
    gap_concept_b: str
    source_question: str   # the research question from Stage 2

    # The hypothesis
    statement: str         # "We hypothesize that X because Y"
    mechanism: str         # proposed causal/mathematical mechanism
    prediction: str        # specific, measurable prediction

    # Falsifiability
    falsifiable: bool = True
    falsification_criteria: str   # what would definitively refute this
    minimum_effect_size: str = ""  # e.g. 'r > 0.3', '>2-fold change'

    # Convenience alias used in notebooks / CLI
    @property
    def source_gap(self) -> str:
        return f"{self.gap_concept_a} ⇔ {self.gap_concept_b}"

    # Scoring (1–5)
    novelty: int      = Field(ge=1, le=5)
    rigor: int        = Field(ge=1, le=5)   # how well-formed / testable
    impact: int       = Field(ge=1, le=5)
    replication_risk: str = "medium"  # 'low' | 'medium' | 'high'

    @computed_field
    @property
    def composite_score(self) -> float:
        """Weighted composite: impact×0.4 + novelty×0.35 + rigor×0.25"""
        return round(self.impact * 0.4 + self.novelty * 0.35 + self.rigor * 0.25, 3)

    # Experiment
    experiment: Optional[ExperimentDesign] = None

    # Provenance
    keywords: List[str] = Field(default_factory=list)
    gap_similarity: float = 0.0
    gap_distance: int = 999

    # Human review
    approved: Optional[bool] = None   # None = unreviewed; True/False = reviewed


class HypothesisReport(BaseModel):
    """Full ranked hypothesis report — the terminal Stage 3 output."""
    domain: str
    query: str
    n_gaps_considered: int = 0
    hypotheses: List[Hypothesis] = Field(default_factory=list)

    @property
    def ranked(self) -> List["Hypothesis"]:
        return sorted(self.hypotheses, key=lambda h: h.composite_score, reverse=True)

    def top(self, n: int = 5) -> List["Hypothesis"]:
        """Return the top-n hypotheses by composite score."""
        return self.ranked[:n]

    @property
    def computational(self) -> List[Hypothesis]:
        """Hypotheses with computational experiment designs — runnable immediately."""
        return [
            h for h in self.ranked
            if h.experiment and h.experiment.computational
        ]

    @property
    def pending_review(self) -> List[Hypothesis]:
        """Hypotheses not yet human-approved/rejected."""
        return [h for h in self.ranked if h.approved is None]
