"""Tests for Pydantic models: round-trip JSON, computed fields, scoring."""
from __future__ import annotations

import json
import pytest

from athanor.graph.models import Concept, Edge, ConceptGraph
from athanor.gaps.models import CandidateGap, GapAnalysis, GapReport
from athanor.hypotheses.models import (
    ExperimentDesign,
    Hypothesis,
    HypothesisReport,
)


# ────────────────────────── helpers ──────────────────────────────────────────

def _make_concept(**kw):
    defaults = dict(label="Test Concept", description="A test concept")
    defaults.update(kw)
    return Concept(**defaults)


def _make_edge(**kw):
    defaults = dict(source="A", target="B", relation="relates to")
    defaults.update(kw)
    return Edge(**defaults)


def _make_gap_analysis(**kw):
    defaults = dict(
        concept_a="A", concept_b="B",
        research_question="How does A relate to B?",
        why_unexplored="Nobody looked", intersection_opportunity="Great synergy",
        methodology="Compute stuff", computational=True,
        novelty=4, tractability=3, impact=5,
    )
    defaults.update(kw)
    return GapAnalysis(**defaults)


def _make_hypothesis(**kw):
    defaults = dict(
        gap_concept_a="A", gap_concept_b="B",
        source_question="How does A relate to B?",
        statement="We hypothesize X because Y",
        mechanism="X causes Y via Z",
        prediction="X should increase Y by 50%",
        falsification_criteria="If Y doesn't increase, reject",
        novelty=4, rigor=3, impact=5,
        experiment=ExperimentDesign(
            approach="Run simulation",
            steps=["Step 1", "Step 2"],
            tools=["Python", "NumPy"],
            computational=True,
            estimated_effort="1 week",
            data_requirements="None",
            expected_positive="Y increases",
            expected_negative="Y stays flat",
            null_hypothesis="No effect of X on Y",
        ),
    )
    defaults.update(kw)
    return Hypothesis(**defaults)


# ────────────────────────── Concept ──────────────────────────────────────────

class TestConceptRoundTrip:
    def test_basic_roundtrip(self):
        c = _make_concept()
        c2 = Concept.model_validate(c.model_dump())
        assert c2 == c

    def test_json_roundtrip(self):
        c = _make_concept(aliases=["TC", "test-concept"], source_papers=["2401.00001"])
        j = c.model_dump_json()
        c2 = Concept.model_validate_json(j)
        assert c2 == c

    def test_defaults(self):
        c = Concept(label="X", description="desc")
        assert c.aliases == []
        assert c.source_papers == []
        assert c.centrality == 0.0
        assert c.burt_constraint == 1.0  # 1.0 = fully embedded (default)


# ────────────────────────── Edge ─────────────────────────────────────────────

class TestEdgeRoundTrip:
    def test_basic_roundtrip(self):
        e = _make_edge()
        e2 = Edge.model_validate(e.model_dump())
        assert e2 == e

    def test_edge_type_values(self):
        for et in ("causal", "inhibitory", "correlational", "methodological",
                    "definitional", "empirical", "analogical"):
            e = _make_edge(edge_type=et)
            assert e.edge_type == et


# ────────────────────────── ConceptGraph ─────────────────────────────────────

class TestConceptGraphRoundTrip:
    def test_full_roundtrip(self):
        cg = ConceptGraph(
            domain="test", query="test query",
            concepts=[_make_concept(label="A"), _make_concept(label="B")],
            edges=[_make_edge(source="A", target="B")],
        )
        j = cg.model_dump_json()
        cg2 = ConceptGraph.model_validate_json(j)
        assert cg2 == cg

    def test_to_networkx(self):
        cg = ConceptGraph(
            concepts=[_make_concept(label="A"), _make_concept(label="B")],
            edges=[_make_edge(source="A", target="B")],
        )
        G = cg.to_networkx()
        assert "A" in G and "B" in G
        assert G.has_edge("A", "B")

    def test_sparse_connections(self):
        cg = ConceptGraph(
            concepts=[_make_concept(label=c) for c in "ABCDE"],
            edges=[
                _make_edge(source="A", target="B"),
                _make_edge(source="B", target="C"),
                _make_edge(source="C", target="A"),  # tight cluster
                _make_edge(source="C", target="D"),   # bridge
                _make_edge(source="D", target="E"),
            ],
        )
        sparse = cg.sparse_connections(top_k=2)
        assert len(sparse) <= 2


# ────────────────────────── GapAnalysis ──────────────────────────────────────

class TestGapAnalysisRoundTrip:
    def test_json_roundtrip(self):
        ga = _make_gap_analysis()
        j = ga.model_dump_json()
        ga2 = GapAnalysis.model_validate_json(j)
        assert ga2.concept_a == ga.concept_a
        assert ga2.composite_score == ga.composite_score

    def test_composite_score_range(self):
        ga = _make_gap_analysis(novelty=1, tractability=1, impact=1)
        assert 1.0 <= ga.composite_score <= 5.0

    def test_composite_score_causal_bonus(self):
        base = _make_gap_analysis(bridge_type="semantic")
        causal = _make_gap_analysis(bridge_type="causal")
        assert causal.composite_score > base.composite_score

    def test_composite_score_structural_hole_bonus(self):
        base = _make_gap_analysis(structural_hole_score=0.0)
        sh = _make_gap_analysis(structural_hole_score=0.8)
        assert sh.composite_score > base.composite_score

    def test_score_capped_at_5(self):
        ga = _make_gap_analysis(
            novelty=5, tractability=5, impact=5,
            bridge_type="causal", structural_hole_score=0.9,
        )
        assert ga.composite_score <= 5.0


# ────────────────────────── GapReport ────────────────────────────────────────

class TestGapReportRoundTrip:
    def test_ranked_sorting(self):
        gr = GapReport(
            domain="test", query="test",
            analyses=[
                _make_gap_analysis(novelty=1, impact=1),
                _make_gap_analysis(novelty=5, impact=5),
            ],
        )
        assert gr.ranked[0].composite_score >= gr.ranked[1].composite_score


# ────────────────────────── Hypothesis ───────────────────────────────────────

class TestHypothesisRoundTrip:
    def test_json_roundtrip(self):
        h = _make_hypothesis()
        j = h.model_dump_json()
        h2 = Hypothesis.model_validate_json(j)
        assert h2.statement == h.statement
        assert h2.composite_score == h.composite_score

    def test_composite_score(self):
        h = _make_hypothesis(novelty=4, rigor=3, impact=5)
        expected = round(5 * 0.4 + 4 * 0.35 + 3 * 0.25, 3)
        assert h.composite_score == expected

    def test_final_score_without_critic(self):
        h = _make_hypothesis()
        assert h.final_score == h.composite_score

    def test_final_score_with_critic(self):
        h = _make_hypothesis(
            critic_novelty=3, critic_rigor=2, critic_impact=4,
        )
        crit = 4 * 0.4 + 3 * 0.35 + 2 * 0.25
        expected = round((h.composite_score + crit) / 2, 3)
        assert h.final_score == expected

    def test_source_gap_property(self):
        h = _make_hypothesis(gap_concept_a="Foo", gap_concept_b="Bar")
        assert h.source_gap == "Foo ⇔ Bar"

    def test_approved_default_none(self):
        h = _make_hypothesis()
        assert h.approved is None


# ────────────────────────── HypothesisReport ─────────────────────────────────

class TestHypothesisReportRoundTrip:
    def test_ranked_by_final_score(self):
        hr = HypothesisReport(
            domain="test", query="test",
            hypotheses=[
                _make_hypothesis(novelty=1, impact=1),
                _make_hypothesis(novelty=5, impact=5),
            ],
        )
        assert hr.ranked[0].final_score >= hr.ranked[1].final_score

    def test_top_n(self):
        hr = HypothesisReport(
            domain="test", query="test",
            hypotheses=[_make_hypothesis() for _ in range(10)],
        )
        assert len(hr.top(3)) == 3

    def test_computational_filter(self):
        hr = HypothesisReport(
            domain="test", query="test",
            hypotheses=[
                _make_hypothesis(),  # computational=True in experiment
                _make_hypothesis(experiment=None),
            ],
        )
        comp = hr.computational
        assert all(h.experiment and h.experiment.computational for h in comp)

    def test_pending_review(self):
        h1 = _make_hypothesis()
        h2 = _make_hypothesis()
        h2.approved = True
        hr = HypothesisReport(domain="t", query="t", hypotheses=[h1, h2])
        assert len(hr.pending_review) == 1

    def test_full_json_roundtrip(self):
        hr = HypothesisReport(
            domain="test", query="test",
            n_gaps_considered=5,
            hypotheses=[_make_hypothesis(), _make_hypothesis()],
        )
        j = hr.model_dump_json()
        hr2 = HypothesisReport.model_validate_json(j)
        assert len(hr2.hypotheses) == 2
        assert hr2.domain == "test"
