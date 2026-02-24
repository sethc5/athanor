"""Tests for athanor.reporting — Markdown renderer extracted from cli.py."""
from __future__ import annotations

from athanor.hypotheses.models import ExperimentDesign, Hypothesis
from athanor.reporting import render_hypothesis_markdown


def _hyp(**kw) -> Hypothesis:
    defaults = dict(
        gap_concept_a="Concept A",
        gap_concept_b="Concept B",
        source_question="How does A relate to B?",
        statement="We hypothesize X",
        mechanism="X causes Y",
        prediction="Y increases",
        falsification_criteria="If Y doesn't increase",
        novelty=4, rigor=3, impact=5,
        experiment=ExperimentDesign(
            approach="Run simulation",
            steps=["Step 1"],
            tools=["Python"],
            computational=True,
            estimated_effort="1 week",
            data_requirements="None",
            expected_positive="Y increases",
            expected_negative="Y stays same",
            null_hypothesis="No effect",
        ),
    )
    defaults.update(kw)
    return Hypothesis(**defaults)


class TestRenderHypothesisMarkdown:
    def test_contains_title(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp())],
            title="Test Report",
            subtitle="dom",
            date_str="2025-01-01",
            total=1,
        )
        assert "# Athanor — Test Report" in md

    def test_contains_hypothesis_statement(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp(statement="We hypothesize dark matter interacts"))],
            title="T", subtitle="s", date_str="2025-01-01", total=1,
        )
        assert "dark matter interacts" in md

    def test_multi_domain_adds_domain_column(self):
        md = render_hypothesis_markdown(
            [("domain_a", _hyp()), ("domain_b", _hyp())],
            title="Cross", subtitle="a, b", date_str="2025-01-01",
            total=2, multi_domain=True,
        )
        assert "Domain" in md
        assert "domain_a" in md
        assert "domain_b" in md

    def test_single_domain_no_domain_column(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp())],
            title="T", subtitle="s", date_str="2025-01-01",
            total=1, multi_domain=False,
        )
        # Summary table header shouldn't have "Domain" column
        lines = md.split("\n")
        header_line = [l for l in lines if l.startswith("| #")][0]
        assert "Domain" not in header_line

    def test_approved_only_annotation(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp())],
            title="T", subtitle="s", date_str="2025-01-01",
            total=1, approved_only=True,
        )
        assert "approved only" in md

    def test_experiment_section_present(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp())],
            title="T", subtitle="s", date_str="2025-01-01", total=1,
        )
        assert "### Experiment" in md
        assert "Step 1" in md

    def test_no_experiment_graceful(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp(experiment=None))],
            title="T", subtitle="s", date_str="2025-01-01", total=1,
        )
        assert "### Experiment" not in md

    def test_empty_candidates(self):
        md = render_hypothesis_markdown(
            [],
            title="Empty", subtitle="s", date_str="2025-01-01", total=0,
        )
        assert "# Athanor" in md
        assert "0 shown" in md

    def test_score_in_summary_table(self):
        md = render_hypothesis_markdown(
            [("dom", _hyp(novelty=5, impact=5, rigor=5))],
            title="T", subtitle="s", date_str="2025-01-01", total=1,
        )
        # Should contain a numeric score in the table
        assert "5.0" in md or "4." in md  # composite is 5*0.4+5*0.35+5*0.25=5.0

