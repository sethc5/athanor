"""
athanor.hypotheses.critic — independent blind re-scoring of generated hypotheses.

Stage 3.5: the problem with self-scored hypotheses is that the same model
that wrote them also rated their novelty/rigor/impact. A separate critic call
sees ONLY the hypothesis content (no original scores) and re-rates independently.

The final_score field on Hypothesis blends generator + critic scores, which
corrects for the systematic self-promotion bias.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import anthropic

from athanor.config import cfg
from athanor.hypotheses.models import Hypothesis, HypothesisReport
from athanor.llm_utils import call_llm_json

log = logging.getLogger(__name__)

# ── prompt ────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a rigorous, skeptical peer reviewer evaluating automated hypothesis proposals.

You will see a hypothesis (statement, mechanism, prediction, and falsification criteria)
with NO quality scores — you MUST assess it independently.

Your mandate:
- Be conservative. Most automated hypotheses are redundant, vague, or already
  addressed in the literature. Score generously only when genuinely justified.
- Apply Popper's falsifiability standard strictly: if the falsification criterion
  is just "if the experiment fails" or circular, penalise rigor heavily.
- Apply novelty skeptically: assume the intersection has been explored unless
  the phrasing specifies a concrete gap. Default to novelty ≤ 3.
- Apply impact skeptically: incremental confirmation of known patterns = 1-2.
  Reserve 5 for hypotheses that would resolve a foundational tension.

Output ONLY valid JSON — no prose, no markdown:

{
  "novelty":  <integer 1-5>,
  "rigor":    <integer 1-5>,
  "impact":   <integer 1-5>,
  "note":     "<One sentence: your most important critique or the key reason for your scores. Be specific.>",
  "already_known": <true if you believe this intersection is already well-covered in the literature>
}

Scoring rubric:

novelty:
  5 = definitively unasked — the specific mechanistic claim does not appear in
      mainstream literature; the intersection is genuinely novel
  4 = under-explored — adjacent work exists but this specific angle is not addressed
  3 = plausible gap — some prior work nearby; novelty depends on specifics
  2 = well-trodden — close parallels exist; incremental at best
  1 = already answered — the hypothesis restates something with published evidence

rigor:
  5 = fully operationalised — single clear prediction, quantified effect,
      concrete positive falsification criterion, formal H₀
  4 = well-formed — clear prediction, plausible falsification, effect size unspecified
  3 = testable but vague — no quantification, or confounded predictions
  2 = partially testable — important untestable component or unfalsifiable element
  1 = not a genuine hypothesis — redescription, unfalsifiable, or circular

impact:
  5 = field-reorienting — resolves a foundational tension or enables a new class of methods
  4 = significant — opens a productive research programme; changes consensus if true
  3 = useful — addresses a real question; modest community impact
  2 = niche — confirmatory or narrow; limited generalisability
  1 = marginal — already confirmed, or too narrow to matter
"""

_CRITIC_USER_TEMPLATE = """\
Hypothesis statement:
{statement}

Proposed mechanism:
{mechanism}

Specific prediction:
{prediction}

Falsification criterion (what would definitively refute this):
{falsification_criteria}

Source gap (the two concepts bridged):
{concept_a} ⟺ {concept_b}

Rate this hypothesis independently on novelty, rigor, and impact (integers 1-5).
"""


class HypothesisCritic:
    """Independent blind re-scorer for Stage 3 hypotheses."""

    def __init__(
        self,
        model: str = cfg.model,
        api_key: str = cfg.anthropic_api_key,
        max_tokens: int = 512,
        max_workers: int = 4,
    ) -> None:
        cfg.validate()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._max_workers = max_workers

    # ── public ───────────────────────────────────────────────────────────────

    def critique(self, report: HypothesisReport) -> HypothesisReport:
        """Run critic pass on every hypothesis in *report*, returning updated report."""
        log.info(
            "Critic pass: scoring %d hypotheses with %d workers",
            len(report.hypotheses), self._max_workers,
        )
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            updated = list(pool.map(self._critique_one, report.hypotheses))

        report.hypotheses = updated
        log.info("Critic pass complete")
        return report

    # ── private ──────────────────────────────────────────────────────────────

    def _critique_one(self, hyp: Hypothesis) -> Hypothesis:
        prompt = _CRITIC_USER_TEMPLATE.format(
            statement=hyp.statement,
            mechanism=hyp.mechanism,
            prediction=hyp.prediction,
            falsification_criteria=hyp.falsification_criteria,
            concept_a=hyp.gap_concept_a,
            concept_b=hyp.gap_concept_b,
        )
        try:
            data, _ = call_llm_json(
                self._client,
                self._model,
                self._max_tokens,
                _CRITIC_SYSTEM,
                prompt,
                use_cache=False,  # critic should never use cached generator prompts
            )
        except anthropic.APIError as exc:
            log.error("Critic API error for '%s': %s", hyp.statement[:60], exc)
            return hyp
        if data is None:
            log.warning("Critic failed for '%s' — keeping original scores", hyp.statement[:60])
            return hyp

        try:
            hyp = hyp.model_copy(update=dict(
                critic_novelty=max(1, min(5, int(data.get("novelty", hyp.novelty)))),
                critic_rigor=max(1, min(5, int(data.get("rigor", hyp.rigor)))),
                critic_impact=max(1, min(5, int(data.get("impact", hyp.impact)))),
                critic_note=str(data.get("note", ""))[:300],
            ))
            if data.get("already_known"):
                log.info(
                    "Critic flagged as already-known: '%s…'",
                    hyp.statement[:80],
                )
        except (ValueError, TypeError) as exc:
            log.error("Critic parse error: %s", exc)

        return hyp
