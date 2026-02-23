"""
athanor.gaps.finder — use Claude to analyse candidate gaps and produce
ranked research questions.

For each CandidateGap Claude answers:
  1. What research question does this structural gap represent?
  2. Why has this connection been missed?
  3. What is the opportunity at this intersection?
  4. How would you investigate it? Is it computationally tractable?
  5. Score: novelty / tractability / impact (1–5)

Domain-agnostic: the prompt is parameterised by domain name.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import anthropic

from athanor.config import cfg
from athanor.gaps.models import CandidateGap, GapAnalysis, GapReport
from athanor.llm_utils import call_llm_json

log = logging.getLogger(__name__)

# ── prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a research gap analyst embedded inside an automated science pipeline.

Your task: given two scientific concepts that are semantically related but
structurally disconnected in the literature graph, identify the research gap
they represent and produce a structured analysis.

Output ONLY valid JSON matching this exact schema — no prose, no markdown:

{
  "research_question": "<one precise, testable question this gap implies>",
  "why_unexplored":    "<why the community has missed or avoided this connection — 2-3 sentences>",
  "intersection_opportunity": "<what productive work at this intersection could achieve — 2-3 sentences>",
  "methodology": "<how you would actually investigate this — concrete steps, 3-5 sentences>",
  "computational": <true if it can be substantially investigated computationally, false if wet-lab/observational primary>,
  "bridge_type": "<classify the gap: 'causal' | 'methodological' | 'semantic' | 'integrative'>",
  "novelty":       <integer 1-5; 5=highly novel>,
  "tractability":  <integer 1-5; 5=very tractable with current tools>,
  "impact":        <integer 1-5; 5=field-changing if answered>,
  "keywords":      ["<3-6 search terms that would find relevant prior work>"]
}

Bridge type definitions (choose ONE):
- "causal":         the gap is a missing mechanistic or causal link (A causes/enables/regulates B,
                    or vice versa) — the most scientifically valuable category
- "methodological": the gap is a missing technique, tool, or formal framework that would
                    connect the two concepts
- "semantic":       the two concepts are studied in parallel but never formally related;
                    the gap is definitional or taxonomic
- "integrative":    the gap bridges two different sub-fields, scales, or model systems
                    (e.g. molecular ↔ organismal, in vitro ↔ in vivo)

CRITICAL for causal gaps: explicitly state whether the causal direction is
A→B, B→A, bidirectional, or unknown, and cite what directionality evidence exists.

Scoring rubric:
- novelty 5: genuinely unasked; no paper directly addresses this intersection
- novelty 1: well-trodden, incremental
- tractability 5: answerable with existing public data + compute
- tractability 1: requires major new instrumentation or decades of data
- impact 5: resolves a foundational tension or enables a new class of methods
- impact 1: niche, confirmatory

Prefer CAUSAL and INTEGRATIVE gaps. Penalise purely semantic gaps (novelty ≤ 2)
unless the semantic clarification would unblock a significant body of research.
"""

_USER_TEMPLATE = """\
Domain: {domain}

Concept A: {concept_a}
Description A: {description_a}
Appears in papers: {papers_a}

Concept B: {concept_b}
Description B: {description_b}
Appears in papers: {papers_b}

Embedding similarity: {similarity:.3f} (high — these concepts are semantically close)
Graph distance: {graph_distance} (large — the literature rarely links them directly)
Structural hole score: {structural_hole_score:.3f} (high = gap bridges disconnected cluster boundary)

Analyze the gap between these two concepts in the context of {domain}.
Pay special attention to whether there is a directional causal relationship
between these concepts. If this is a causal gap, specify the direction (A→B,
B→A, bidirectional, or unknown) and any existing evidence of directionality.
"""


class GapFinder:
    """Calls Claude to turn candidate gaps into ranked research questions."""

    def __init__(
        self,
        domain: str,
        model: str = cfg.model,
        api_key: str = cfg.anthropic_api_key,
        max_tokens: int = 1024,
        max_gaps: int = 20,
        max_workers: int = 4,
        domain_context: str = "",
        prior_approved: Optional[List[str]] = None,
    ) -> None:
        cfg.validate()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._domain = domain
        self._model = model
        self._max_tokens = max_tokens
        self._max_gaps = max_gaps
        self._max_workers = max_workers
        self._domain_context = domain_context
        # Approved hypothesis statements from previous runs — gap finder avoids regenerating similar gaps
        self._prior_approved: List[str] = prior_approved or []

    # ── public ───────────────────────────────────────────────────────────────

    def analyse(
        self,
        gaps: List[CandidateGap],
        query: str = "",
    ) -> GapReport:
        """Analyse up to *max_gaps* candidates and return a ranked GapReport."""
        candidates = gaps[: self._max_gaps]
        report = GapReport(
            domain=self._domain,
            query=query,
            n_candidates=len(gaps),
            n_analyzed=len(candidates),
        )

        log.info("Analysing %d gaps with %d workers", len(candidates), self._max_workers)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            results = list(pool.map(self._analyse_one, candidates))

        report.analyses = [a for a in results if a is not None]
        log.info(
            "Gap analysis complete: %d/%d produced valid analyses",
            len(report.analyses),
            len(candidates),
        )
        return report

    # ── private ──────────────────────────────────────────────────────────────

    def _analyse_one(self, gap: CandidateGap) -> Optional[GapAnalysis]:
        prompt = _USER_TEMPLATE.format(
            domain=self._domain,
            concept_a=gap.concept_a,
            description_a=gap.description_a or "not available",
            papers_a=", ".join(gap.papers_a[:4]) or "unknown",
            concept_b=gap.concept_b,
            description_b=gap.description_b or "not available",
            papers_b=", ".join(gap.papers_b[:4]) or "unknown",
            similarity=gap.similarity,
            graph_distance=gap.graph_distance if gap.graph_distance < 999 else "∞",
            structural_hole_score=gap.structural_hole_score,
        )
        if self._domain_context:
            prompt = f"Domain context:\n{self._domain_context}\n\n" + prompt
        if self._prior_approved:
            approved_block = "\n".join(f"  - {s}" for s in self._prior_approved[:10])
            prompt += (
                f"\n\nAlready-approved hypotheses from previous runs (do NOT generate "
                f"a research question that would lead to a similar hypothesis):\n{approved_block}"
            )
        try:
            data, _ = call_llm_json(
                self._client, self._model, self._max_tokens, _SYSTEM, prompt
            )
        except anthropic.APIError as exc:
            log.error("Anthropic API error for gap %s\u2194%s: %s", gap.concept_a, gap.concept_b, exc)
            return None
        if data is None:
            log.error("Gap analysis failed for %s\u2194%s after retries.", gap.concept_a, gap.concept_b)
            return None
        return self._parse(data, gap)

    def _parse(self, data: dict, gap: CandidateGap) -> Optional[GapAnalysis]:
        try:
            return GapAnalysis(
                concept_a=gap.concept_a,
                concept_b=gap.concept_b,
                research_question=data["research_question"],
                why_unexplored=data["why_unexplored"],
                intersection_opportunity=data["intersection_opportunity"],
                methodology=data["methodology"],
                computational=bool(data.get("computational", True)),
                bridge_type=data.get("bridge_type", "semantic"),
                novelty=int(data.get("novelty", 3)),
                tractability=int(data.get("tractability", 3)),
                impact=int(data.get("impact", 3)),
                keywords=data.get("keywords", []),
                similarity=gap.similarity,
                graph_distance=gap.graph_distance,
                structural_hole_score=gap.structural_hole_score,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Model validation error for gap %s\u2194%s: %s", gap.concept_a, gap.concept_b, exc)
            return None
