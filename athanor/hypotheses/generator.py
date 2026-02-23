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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import anthropic

from athanor.config import cfg
from athanor.gaps.models import GapAnalysis
from athanor.hypotheses.models import (
    ExperimentDesign,
    Hypothesis,
    HypothesisReport,
)
from athanor.llm_utils import call_llm_json

log = logging.getLogger(__name__)

# ── prompts ──────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a hypothesis generation engine inside an automated science pipeline.

Your task: given a research gap and its analysis, generate:
1. A single tight, falsifiable hypothesis (Popper standard)
2. A concrete experiment design to test it

A genuine hypothesis (Popperian criteria):
- States a specific mechanistic or causal claim, NOT a redescription of an observation
- Has a logically prior falsification criterion: a result that would REFUTE it,
  specified before running the experiment
- Makes a quantified or qualifiable prediction, not just "A affects B"
- Is distinguishable from its null hypothesis

Output ONLY valid JSON matching this exact schema — no prose, no markdown:

{
  "statement":              "<The hypothesis as a single declarative sentence. Start with 'We hypothesize that…'>",
  "mechanism":              "<Proposed causal or mathematical mechanism — 2-3 sentences. Must specify direction: A causes B, B causes A, or bidirectional>",
  "prediction":             "<One specific, measurable prediction with direction and magnitude where possible. E.g.: 'Increasing X by 10%% will reduce Y by at least 5%%'>",
  "falsifiable":            <true/false — true only if falsification_criteria specifies a concrete refuting result>,
  "falsification_criteria": "<A specific, concrete result that would definitively refute this hypothesis. NOT 'if the experiment fails' — a positive result that contradicts the mechanism>",
  "minimum_effect_size":    "<The minimum detectable effect that would constitute confirmation, e.g. 'r > 0.3', 'p < 0.05 with n=100', '>2-fold change'>",
  "novelty":   <integer 1-5>,
  "rigor":     <integer 1-5; how well-formed and testable is this hypothesis>,
  "impact":    <integer 1-5>,
  "replication_risk": "<'low' | 'medium' | 'high' — how likely this result will fail to replicate>",
  "keywords":  ["<3-6 search terms>"],
  "experiment": {
    "approach":             "<High-level strategy in 1-2 sentences>",
    "steps":                ["<Step 1>", "<Step 2>", "..."],
    "tools":                ["<specific tool, dataset, or method>", "..."],
    "computational":        <true if primarily computational, false if wet-lab/observational primary>,
    "estimated_effort":     "<e.g. '1-2 weeks compute', '6-month wet lab study'>",
    "data_requirements":    "<What data or resources are needed>",
    "expected_positive":    "<What result confirms the hypothesis — match prediction above>",
    "expected_negative":    "<What result refutes it — match falsification_criteria above>",
    "null_hypothesis":      "<Formal H\u2080: the null claim that the experiment is designed to reject>",
    "statistical_test":     "<The specific statistical test + alpha threshold, e.g. 'two-sided t-test, alpha=0.05, power=0.80'>",
    "minimum_detectable_effect": "<Smallest effect size that would be scientifically meaningful, e.g. \"Cohen's d > 0.2 (~200/arm at 80% power)\" or \"≥2-fold enrichment\">",
    "statistical_power_notes": "<Sample size justification: state assumed effect size, alpha, desired power (80% or 90%), and resulting N per group. For computational experiments, state convergence criterion instead.>",
    "limitations":          ["<Limitation 1>", "..."],
    "requires_followup":    "<If computational: what wet-lab step would be needed to fully confirm; null if not applicable>"
  }
}

Scoring rubric for rigor (1–5):
5 = fully operationalised — single clear prediction, quantified effect, concrete falsification test, formal H₀
4 = well-formed — clear prediction, plausible falsification, but effect size unspecified
3 = testable but vague — no quantification, or multiple confounded predictions
2 = partially testable — important untestable component or unfalsifiable element
1 = not a genuine hypothesis — redescription, unfalsifiable, or circular

Replication risk rubric:
- low:    computational/mathematical; deterministic or well-powered statistical test;
          no single-lab dependencies; standard public datasets
- medium: well-powered study but relies on biological variability, model organisms,
          or proprietary data; effect size moderate; established methodology
- high:   small n; requires hard-to-replicate lab conditions; p-hacking risk;
          single assay; novel measurement technique with no cross-lab validation

For experiment.computational:
- true  = can be substantially tested with existing public data + code + compute
- false = primarily requires new experiments, clinical trials, or field work
"""

_USER_TEMPLATE = """\
Domain: {domain}

Research gap:
  Concept A: {concept_a}
  Concept B: {concept_b}

Gap classification: {bridge_type}
  (causal=missing mechanistic link | methodological=missing tool/framework |
   semantic=conceptual | integrative=cross-field/cross-scale)

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

For CAUSAL gaps: the hypothesis MUST specify causal direction and mechanism.
  Include in minimum_effect_size a concrete threshold (e.g. 'hazard ratio > 1.5',
  'explained variance > 10%%').
For METHODOLOGICAL gaps: the hypothesis should describe the missing framework
  and predict its performance advantage over existing approaches.
"""


class HypothesisGenerator:
    """Generates falsifiable hypotheses and experiment designs from GapAnalyses."""

    def __init__(
        self,
        domain: str,
        model: str = cfg.model,
        api_key: str = cfg.anthropic_api_key,
        max_tokens: int = 4096,
        max_workers: int = 4,
    ) -> None:
        cfg.validate()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._domain = domain
        self._model = model
        self._max_tokens = max_tokens
        self._max_workers = max_workers

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

        log.info("Generating %d hypotheses with %d workers", len(targets), self._max_workers)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            results = list(pool.map(self._generate_one, targets))

        report.hypotheses = [h for h in results if h is not None]
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
            bridge_type=getattr(analysis, "bridge_type", "semantic"),
            research_question=analysis.research_question,
            why_unexplored=analysis.why_unexplored,
            opportunity=analysis.intersection_opportunity,
            methodology=analysis.methodology,
            novelty=analysis.novelty,
            tractability=analysis.tractability,
            impact=analysis.impact,
        )

        try:
            data, _ = call_llm_json(
                self._client, self._model, self._max_tokens, _SYSTEM, prompt
            )
        except anthropic.APIError as exc:
            log.error("API error for %s\u2194%s: %s", analysis.concept_a, analysis.concept_b, exc)
            return None
        if data is None:
            log.error("Hypothesis generation failed for %s\u2194%s after retries.", analysis.concept_a, analysis.concept_b)
            return None

        return self._parse(data, analysis)

    def _parse(self, data: dict, analysis: GapAnalysis) -> Optional[Hypothesis]:
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
                statistical_test=exp_data.get("statistical_test", ""),
                minimum_detectable_effect=exp_data.get("minimum_detectable_effect", ""),
                statistical_power_notes=exp_data.get("statistical_power_notes", ""),
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
                minimum_effect_size=data.get("minimum_effect_size", ""),
                novelty=int(data.get("novelty", analysis.novelty)),
                rigor=int(data.get("rigor", 3)),
                impact=int(data.get("impact", analysis.impact)),
                replication_risk=data.get("replication_risk", "medium"),
                keywords=data.get("keywords", analysis.keywords),
                gap_similarity=analysis.similarity,
                gap_distance=analysis.graph_distance,
                experiment=experiment,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Hypothesis build error: %s", exc)
            return None
