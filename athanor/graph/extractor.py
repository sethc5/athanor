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
from athanor.llm_utils import call_llm_json

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
      "relation": "<verb phrase — see controlled vocabulary below>",
      "edge_type": "<one of: causal | inhibitory | correlational | methodological | definitional | empirical | analogical>",
      "weight": <float 0.1–1.0 indicating strength>,
      "evidence": "<short quote or paraphrase supporting this relation>"
    }
  ]
}

Controlled edge_type vocabulary (choose the BEST fit):
- causal:          A directly causes, enables, or produces B (mechanistic link, direction known)
- inhibitory:      A suppresses, blocks, or down-regulates B (causal but negative direction)
- correlational:   A and B co-occur or co-vary; directionality unknown or not established
- methodological:  A is a technique, tool, or framework used to study/measure/derive B
- definitional:    A is a special case of B, or B is formally defined in terms of A (isa / part-of)
- empirical:       A is observed or measured in / from B (data-level link)
- analogical:      A is structurally analogous to B in a different domain

Relation verb vocabulary (pick from these or coin a domain-specific verb):
  causes · inhibits · enables · regulates · requires · produces · prevents
  correlates_with · predicts · approximates · bounds · constrains · measures
  extends · generalises · reduces_to · is-a · part-of · characterised_by
  exchanges · classifies · stabilises · compactifies_on · fibres_over
  contradicts · refutes · analogous_to

Rules:
- Extract 5–15 concepts per paper.
- Each concept label must be a canonical scientific term (not a sentence).
- Edges must only reference labels that appear in "concepts".
- Prefer specific relations over generic ones.
- weight 1.0 = central/strong, 0.1 = peripheral/weak.
- Do not include author names, institutions, or paper titles as concepts.

Examples of correct output (biology paper):
{
  "concepts": [
    {"label": "mTOR signalling", "description": "Serine/threonine kinase pathway integrating nutrient signals to regulate cell growth and autophagy.", "aliases": ["mTOR pathway"]},
    {"label": "autophagy", "description": "Lysosomal degradation pathway that recycles damaged organelles to maintain cellular homeostasis.", "aliases": ["macroautophagy"]},
    {"label": "caloric restriction", "description": "Reduction of dietary energy intake without malnutrition; the most robust lifespan-extending intervention across species.", "aliases": ["dietary restriction"]}
  ],
  "edges": [
    {"source": "caloric restriction", "target": "mTOR signalling", "relation": "inhibits", "edge_type": "causal", "weight": 0.9, "evidence": "CR reduces serum IGF-1, suppressing mTORC1 activity (Fig. 3)."},
    {"source": "mTOR signalling", "target": "autophagy", "relation": "inhibits", "edge_type": "inhibitory", "weight": 0.95, "evidence": "mTORC1 phosphorylates ULK1, blocking autophagy initiation; rapamycin restores flux (Fig. 5b)."}
  ]
}

Example (physics/mathematics paper):
{
  "concepts": [
    {"label": "Calabi-Yau manifold", "description": "Complex K\u00e4hler manifold with vanishing first Chern class used to compactify extra dimensions in string theory.", "aliases": ["CY3"]},
    {"label": "mirror symmetry", "description": "Duality between CY pairs exchanging complex-structure and K\u00e4hler moduli: h^{1,1}(X)=h^{2,1}(Y).", "aliases": []},
    {"label": "Hodge numbers", "description": "Topological invariants h^{p,q} characterising CY cohomology; h^{1,1} counts K\u00e4hler moduli.", "aliases": ["Hodge data"]}
  ],
  "edges": [
    {"source": "mirror symmetry", "target": "Hodge numbers", "relation": "exchanges", "edge_type": "definitional", "weight": 0.9, "evidence": "Mirror symmetry swaps h^{1,1}\u2194h^{2,1}, providing a shortcut for instanton enumeration (Sec. 4.2)."},
    {"source": "Calabi-Yau manifold", "target": "Hodge numbers", "relation": "characterised_by", "edge_type": "definitional", "weight": 1.0, "evidence": "Authors classify CY3 geometries by Hodge pairs, listing 30,108 distinct topologies (Table 1)."}
  ]
}
"""

_USER_TEMPLATE = """\
Extract the concept graph from the following paper text.
{domain_context_block}
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
        max_tokens: int = 3000,  # was 2048; math/physics papers need more headroom
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
        domain_context: str = "",
    ) -> tuple[List[Concept], List[Edge]]:
        """Extract concepts and edges from *text*.

        Returns (concepts, edges) — both tagged with arxiv_id provenance.
        """
        ctx_block = (
            f"\nDomain context:\n{domain_context}\n"
            if domain_context else ""
        )
        data, raw = call_llm_json(
            self._client, self._model, self._max_tokens,
            _SYSTEM, _USER_TEMPLATE.format(text=text, domain_context_block=ctx_block),
        )
        if data is None:
            log.error("Concept extraction failed for paper %s — skipping.", arxiv_id or "?")
            return [], []
        return self._parse_response(data, arxiv_id)

    # ── private ──────────────────────────────────────────────────────────────

    def _parse_response(
        self,
        data: dict,
        arxiv_id: str,
    ) -> tuple[List[Concept], List[Edge]]:
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
                    edge_type=e.get("edge_type", "empirical"),
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