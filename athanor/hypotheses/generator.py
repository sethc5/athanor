"""
athanor.hypotheses.generator — use Claude to convert a GapAnalysis into a
falsifiable hypothesis and concrete experiment design.

Domain-agnostic: no field-specific assumptions.
Produces computational experiments where possible;
explicitly flags what needs wet-lab or real-world validation.
"""
from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

import anthropic

from athanor.config import cfg
from athanor.gaps.models import GapAnalysis
from athanor.hypotheses.models import (
    ExperimentDesign,
    Hypothesis,
    HypothesisReport,
)

log = logging.getLogger(__name__)

# ── prompts ──────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a hypothesis generation engine inside an automated science pipeline.

Your task: given a research gap and its analysis, generate:
1. A single tight, falsifiable hypothesis
2. A concrete experiment design to test it

Output ONLY valid JSON matching this exact schema — no prose, no markdown:

{
  "statement":              "<The hypothesis as a single declarative sentence. Start with 'We hypothesize that…'>",
  "mechanism":              "<Proposed causal or mathematical mechanism — 2-3 sentences>",
  "prediction":             "<One specific, measurable prediction that follows from the hypothesis>",
  "falsifiable":            <true/false>,
  "falsification_criteria": "<What result would definitively refute this hypothesis>",
  "novelty":   <integer 1-5>,
  "rigor":     <integer 1-5; how well-formed and testable is this hypothesis>,
  "impact":    <integer 1-5>,
  "keywords":  ["<3-6 search terms>"],
  "experiment": {
    "approach":             "<High-level strategy in 1-2 sentences>",
    "steps":                ["<Step 1>", "<Step 2>", "..."],
    "tools":                ["<specific tool, dataset, or method>", "..."],
    "computational":        <true if primarily computational, false if wet-lab/observational primary>,
    "estimated_effort":     "<e.g. '1-2 weeks compute', '6-month wet lab study'>",
    "data_requirements":    "<What data or resources are needed>",
    "expected_positive":    "<What result confirms the hypothesis>",
    "expected_negative":    "<What result refutes it>",
    "null_hypothesis":      "<Formal H₀>",
    "limitations":          ["<Limitation 1>", "..."],
    "requires_followup":    "<If computational: what wet-lab step would be needed to fully confirm; null if not applicable>"
  }
}

Scoring rubric for rigor (1–5):
5 = fully operationalised, single clear prediction, obvious falsification test
1 = vague claim, untestable, or unfalsifiable

For experiment.computational:
- true  = can be substantially tested with existing public data + code + compute
- false = primarily requires new experiments, clinical trials, or field work
"""

_USER_TEMPLATE = """\
Domain: {domain}

Research gap:
  Concept A: {concept_a}
  Concept B: {concept_b}

Research question identified in Stage 2:
  {research_question}

Why this gap exists (Stage 2 analysis):
  {why_unexplored}

Opportunity (Stage 2 analysis):
  {opportunity}

Suggested methodology from Stage 2:
  {methodology}

Gap scores: novelty={novelty}, tractability={tractability}, impact={impact}

Generate a hypothesis and experiment design for this gap in {domain}.
Prefer computational experiments. If the most important test requires wet lab,
flag it clearly in requires_followup and design a computational proxy experiment
as the primary steps.
"""


class HypothesisGenerator:
    """Generates falsifiable hypotheses and experiment designs from GapAnalyses."""

    def __init__(
        self,
        domain: str,
        model: str = cfg.model,
        api_key: str = cfg.anthropic_api_key,
        max_tokens: int = 2048,
        sleep_between: float = 0.5,
    ) -> None:
        cfg.validate()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._domain = domain
        self._model = model
        self._max_tokens = max_tokens
        self._sleep = sleep_between

    # ── public ───────────────────────────────────────────────────────────────

    def generate(
        self,
        analyses: List[GapAnalysis],
        query: str = "",
        approved_only: bool = False,
    ) -> HypothesisReport:
        """Generate a hypothesis for each GapAnalysis, return a HypothesisReport.

        Args:
            analyses:      ranked GapAnalysis objects from Stage 2
            query:         original search query (for metadata)
            approved_only: if True, only process analyses with approved=True
        """
        targets = [a for a in analyses if not approved_only or getattr(a, "approved", None) is True]

        report = HypothesisReport(
            domain=self._domain,
            query=query,
            n_gaps_considered=len(targets),
        )

        for i, analysis in enumerate(targets):
            log.info(
                "[%d/%d] Generating hypothesis: %s ↔ %s",
                i + 1, len(targets),
                analysis.concept_a, analysis.concept_b,
            )
            hyp = self._generate_one(analysis)
            if hyp:
                report.hypotheses.append(hyp)
            time.sleep(self._sleep)

        log.info(
            "Hypothesis generation complete: %d/%d succeeded",
            len(report.hypotheses), len(targets),
        )
        return report

    # ── private ──────────────────────────────────────────────────────────────

    def _generate_one(self, analysis: GapAnalysis) -> Optional[Hypothesis]:
        prompt = _USER_TEMPLATE.format(
            domain=self._domain,
            concept_a=analysis.concept_a,
            concept_b=analysis.concept_b,
            research_question=analysis.research_question,
            why_unexplored=analysis.why_unexplored,
            opportunity=analysis.intersection_opportunity,
            methodology=analysis.methodology,
            novelty=analysis.novelty,
            tractability=analysis.tractability,
            impact=analysis.impact,
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        except anthropic.APIError as exc:
            log.error("API error for %s↔%s: %s", analysis.concept_a, analysis.concept_b, exc)
            return None

        return self._parse(raw, analysis)

    def _parse(self, raw: str, analysis: GapAnalysis) -> Optional[Hypothesis]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.error("JSON parse error: %s", exc)
            return None

        try:
            exp_data = data.get("experiment", {})
            experiment = ExperimentDesign(
                approach=exp_data.get("approach", ""),
                steps=exp_data.get("steps", []),
                tools=exp_data.get("tools", []),
                computational=bool(exp_data.get("computational", True)),
                estimated_effort=exp_data.get("estimated_effort", ""),
                data_requirements=exp_data.get("data_requirements", ""),
                expected_positive=exp_data.get("expected_positive", ""),
                expected_negative=exp_data.get("expected_negative", ""),
                null_hypothesis=exp_data.get("null_hypothesis", ""),
                limitations=exp_data.get("limitations", []),
                requires_followup=exp_data.get("requires_followup"),
            )

            return Hypothesis(
                gap_concept_a=analysis.concept_a,
                gap_concept_b=analysis.concept_b,
                source_question=analysis.research_question,
                statement=data["statement"],
                mechanism=data["mechanism"],
                prediction=data["prediction"],
                falsifiable=bool(data.get("falsifiable", True)),
                falsification_criteria=data["falsification_criteria"],
                novelty=int(data.get("novelty", analysis.novelty)),
                rigor=int(data.get("rigor", 3)),
                impact=int(data.get("impact", analysis.impact)),
                keywords=data.get("keywords", analysis.keywords),
                gap_similarity=analysis.similarity,
                gap_distance=analysis.graph_distance,
                experiment=experiment,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Hypothesis build error: %s", exc)
            return None
