"""
athanor.graph.extractor — use Claude to extract concepts and relations
from a single paper's text.

This is the AI backbone of Stage 1.  The prompt is deliberately
domain-agnostic: it asks for concepts and relations without assuming any
particular field vocabulary.  Swap the query, get a different domain map.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

import anthropic

from athanor.config import cfg
from athanor.graph.models import Concept, Edge

log = logging.getLogger(__name__)

# ── prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a scientific concept extraction engine.
Your task: given the text of an academic paper, extract the key concepts
and the relationships between them.

Output ONLY valid JSON matching this exact schema — no prose, no markdown:

{
  "concepts": [
    {
      "label": "<short canonical name, 1-4 words>",
      "description": "<one sentence definition>",
      "aliases": ["<alternative names if any>"]
    }
  ],
  "edges": [
    {
      "source": "<concept label>",
      "target": "<concept label>",
      "relation": "<verb phrase, e.g. extends | depends_on | contradicts | enables | measures | approximates>",
      "weight": <float 0.1–1.0 indicating strength>,
      "evidence": "<short quote or paraphrase supporting this relation>"
    }
  ]
}

Rules:
- Extract 5–15 concepts per paper.
- Each concept label must be a canonical scientific term (not a sentence).
- Edges must only reference labels that appear in "concepts".
- Prefer specific relations over generic ones.
- weight 1.0 = central/strong, 0.1 = peripheral/weak.
- Do not include author names, institutions, or paper titles as concepts.
"""

_USER_TEMPLATE = """\
Extract the concept graph from the following paper text.

---
{text}
---
"""

# ── extractor ────────────────────────────────────────────────────────────────


class ConceptExtractor:
    """Calls Claude to extract a concept sub-graph from a single paper.

    Returns lists of Concept and Edge objects (not merged across papers —
    that is the GraphBuilder's job).
    """

    def __init__(
        self,
        model: str = cfg.model,
        api_key: str = cfg.anthropic_api_key,
        max_tokens: int = 2048,
    ) -> None:
        cfg.validate()
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    # ── public ───────────────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        arxiv_id: str = "",
    ) -> tuple[List[Concept], List[Edge]]:
        """Extract concepts and edges from *text*.

        Returns (concepts, edges) — both tagged with arxiv_id provenance.
        """
        raw = self._call_claude(text)
        return self._parse_response(raw, arxiv_id)

    # ── private ──────────────────────────────────────────────────────────────

    def _call_claude(self, text: str) -> str:
        log.info("Calling %s for concept extraction...", self._model)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM,
            messages=[
                {"role": "user", "content": _USER_TEMPLATE.format(text=text)}
            ],
        )
        return response.content[0].text

    def _parse_response(
        self,
        raw: str,
        arxiv_id: str,
    ) -> tuple[List[Concept], List[Edge]]:
        # Strip markdown fences if Claude emits them despite instructions
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.error("Failed to parse Claude JSON: %s\n%s", exc, raw[:500])
            return [], []

        concepts = []
        known_labels = set()
        for c in data.get("concepts", []):
            try:
                concept = Concept(
                    label=c["label"],
                    description=c.get("description", ""),
                    aliases=c.get("aliases", []),
                    source_papers=[arxiv_id] if arxiv_id else [],
                )
                concepts.append(concept)
                known_labels.add(concept.label)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping malformed concept %s: %s", c, exc)

        edges = []
        for e in data.get("edges", []):
            if e.get("source") not in known_labels or e.get("target") not in known_labels:
                log.debug("Skipping edge with unknown concept: %s", e)
                continue
            try:
                edge = Edge(
                    source=e["source"],
                    target=e["target"],
                    relation=e.get("relation", "related_to"),
                    weight=float(e.get("weight", 0.5)),
                    evidence=e.get("evidence", ""),
                    source_papers=[arxiv_id] if arxiv_id else [],
                )
                edges.append(edge)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping malformed edge %s: %s", e, exc)

        log.info(
            "Extracted %d concepts, %d edges from paper %s",
            len(concepts),
            len(edges),
            arxiv_id or "?",
        )
        return concepts, edges
