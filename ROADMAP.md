# Athanor — Technical Roadmap

Knowledge-to-code mapping: each insight below is pinned to the module(s) it
affects, its implementation priority, and current status.

---

## Tier 1 — Core pipeline intelligence (highest leverage)

### 1. Structural hole theory (Burt)
**What:** Betweenness centrality marks who is central; Burt constraint marks who
is a *broker* — a concept straddling otherwise disconnected clusters. A low
constraint node sitting between two high-constraint clusters is the algorithmic
definition of a research gap. Triadic closure tells us which triangles are
missing.

**Affects:** `graph/builder.py`, `graph/models.py`, `gaps/models.py`

**Implementation:**
- `Concept.burt_constraint` — Burt's redundancy constraint (0=pure broker, 1=fully embedded)
- `Concept.effective_size` — non-redundant contacts in ego network
- `GraphBuilder._compute_structural_holes()` — networkx `constraint()` + `effective_size()`
- `CandidateGap.structural_hole_score` — combined signal: low constraint + high sim + graph distance
- `GapAnalysis.bridge_type` — classify gap as "causal" | "methodological" | "semantic" | "integrative"

**Status:** ✅ Implemented (Session 4)

---

### 2. Knowledge representation & ontology engineering
**What:** The difference between a taxonomy (A isa B) and an ontology (A *enables*
B *under condition* C) is the difference between a list and a knowledge graph.
OWL/RDF formalism informs edge type design. Concept canonicalization should
leverage alias graphs, not just case-normalization.

**Affects:** `graph/extractor.py`, `graph/models.py`

**Implementation ideas:**
- Richer edge type vocabulary per domain (currently: free text relation)
- Formal edge typing: `{is-a, part-of, regulates, enables, inhibits, correlates, causes}`
- Concept disambiguation: TF-IDF fingerprinting for same-label/different-meaning

**Status:** 🔲 Planned

---

### 3. Information retrieval & bibliometrics
**What:** Co-citation analysis (papers cited together are semantically proximate)
and bibliographic coupling (papers citing the same sources are topically linked)
give a richer signal than keyword search alone. The S2 citation graph unlocks
this entirely.

**Affects:** `ingest/semantic_scholar.py`, `graph/builder.py`

**Implementation ideas:**
- Co-citation matrix from S2 citation data → seed edges in concept graph
- Bibliographic coupling score as edge weight signal
- H-index of source papers as trustworthiness proxy for concept provenance

**Status:** ✅ Implemented (Session 5) — `SemanticScholarClient.fetch_references_batch()` + `ingest/cocitation.py` (`compute_biblio_coupling()`). Activated via `athanor run --cocite --s2`.

### 4. Philosophy of falsification (Popper)
**What:** A genuine hypothesis must specify what would refute it. "X correlates
with Y" is not a hypothesis — it is an observation. A hypothesis requires a
mechanistic claim, a quantified prediction, and a falsification criterion that
is logically prior to the experiment.

**Affects:** `hypotheses/generator.py`

**Implementation:**
- Strengthened system prompt: explicit falsification criteria + null hypothesis formalism
- `Hypothesis.falsifiable` scoring rubric tied to operationalization quality
- Added: "What effect size would definitively refute this?" to experiment design
- `bridge_type` → causal hypotheses get higher rigor requirements in prompt

**Status:** ✅ Implemented (Session 4)

---

### 5. Causal inference (DAGs / do-calculus)
**What:** Semantic proximity ≠ causal mechanism. The gap finder currently surfaces
pairs that are *semantically near* but *structurally distant* — but the highest
value gaps are missing *causal links*, not just conceptual proximity. Pearl's
do-calculus gives a formal framework for what "causal gap" means.

**Affects:** `gaps/finder.py`, `gaps/models.py`

**Implementation:**
- `GapAnalysis.bridge_type` field: "causal" | "methodological" | "semantic" | "integrative"
- Updated system prompt: ask Claude to distinguish correlation-based from  
  mechanism-based gaps; flag directionality evidence
- Weight causal gaps higher in composite scoring

**Status:** ✅ Implemented (Session 4)

---

### 6. Experiment design theory (DOE)
**What:** Power analysis, effect size estimation, and what makes a computational
experiment probative. The `ExperimentDesign` model has the right fields; the
scoring rubric would improve with formal DOE knowledge baked into the prompt.

**Affects:** `hypotheses/generator.py`, `hypotheses/models.py`

**Implementation ideas:**
- Add `minimum_detectable_effect` field to ExperimentDesign
- Add `statistical_power_notes` field
- Prompt asks Claude to specify a concrete statistical test + alpha threshold

**Status:** 🔲 Planned

---

## Tier 3 — Domain-agnostic scientific method

### 7. Embedding geometry (isotropy / hubness)
**What:** Sentence-transformer cosine similarities compress toward 1 as
dimensionality grows — the "hubness" problem. All-MiniLM-L6-v2's 384-D space
is not isotropic; the mean vector is far from the origin. Mean-centering before
cosine similarity removes the dominant global direction and restores
discriminability.

**Affects:** `embed/embedder.py`, `gaps/dedup.py`

**Implementation:**
- `Embedder.embed()` now returns mean-centered vectors by default
- `Embedder.similarity_matrix()` mean-centers before cosine sim
- `deduplicate_gaps()` uses centered vectors automatically

**Status:** ✅ Implemented (Session 4)

---

### 8. Meta-science & replication
**What:** Most novel findings fail to replicate. Base-rate problem: if P(H is true)
is low (novel domain), even a significant p-value has low PPV. This informs
which hypotheses are worth pursuing and what error bars matter.

**Affects:** `hypotheses/generator.py`, scoring rubric

**Implementation ideas:**
- Add `replication_risk` field to Hypothesis (low/medium/high)
- Prompt Claude to flag single-study dependencies
- Penalize hypotheses that require hard-to-replicate wet-lab conditions

**Status:** 🔲 Planned

---

### 9. Analogical reasoning / cross-domain transfer
**What:** The structural reason "domain-agnostic" is meaningful: a pattern in  
information theory (e.g. channel capacity ↔ noise) may be isomorphic to one in
aging biology (e.g. epigenetic fidelity ↔ damage accumulation). This is the
basis for cross-domain hypothesis seeding.

**Affects:** Stage 4 (future), `domains/` config

**Implementation ideas:**
- Cross-domain gap comparison at Stage 2: flag gaps already "solved" in another domain
- Structural graph isomorphism search across domain ConceptGraphs
- Cross-domain suggestion engine as stage 2.5

**Status:** 🔲 Future (Stage 4) — **partially done (Session 5):** `athanor cross-domain --domain-a X --domain-b Y` CLI command implemented (cosine bridge detection → GapFinder → JSON report). Graph-isomorphism search remains future work.

---

## Tier 4 — Computational substrate

### 10. Graph neural networks
**Predictive** gap detection rather than structural detection — train on
known-gap / not-gap labels to predict where the *next* gap is likely. Natural
Stage 4 extension once labeled data accumulates from human review.

**Status:** 🔲 Future (Stage 5)

---

### 11. LLM prompt engineering for structured extraction
**What:** The concept/edge extraction is only as good as the prompts. Best
practices: few-shot scientific examples, explicit negative examples (what not
to extract), schema validation in the loop, retry on malformed JSON.

**Affects:** `graph/extractor.py`, `gaps/finder.py`, `hypotheses/generator.py`

**Implementation ideas:**
- Add 2-3 few-shot examples to `graph/extractor.py` system prompt
- Add retry loop with error feedback on JSON parse failure
- Add confidence field to extracted concepts

**Status:** 🔲 Planned

---

## Domain: Longevity Biology

### 12. Hallmarks of aging (Lopez-Otin 2013 + 2023 update)
**Original 9 (2013):** genomic instability, telomere attrition, epigenetic
alterations, loss of proteostasis, deregulated nutrient sensing, mitochondrial
dysfunction, cellular senescence, stem cell exhaustion, altered intercellular
communication.

**New 3 (2023):** disabled macroautophagy, chronic inflammation (inflammaging),
dysbiosis.

**Affects:** `domains/longevity_biology.yaml`

**Status:** ✅ Implemented (Session 4)

---

### 13. Network medicine / systems biology
**What:** Biological networks differ from citation networks — scale-free topology,
disease modules, essential hubs (lethal if removed). Bridge genes between disease
modules are drug targets. Maps directly onto the graph architecture.

**Affects:** `domains/longevity_biology.yaml`, `graph/extractor.py` (domain context)

**Status:** 🔲 Planned

---

## Tier 5 — Human-in-the-loop review

### 14. Hypothesis approval loop
**What:** Researchers need an interactive workflow to curate generated hypotheses
before investing lab/compute resources. Flag approved/rejected hypotheses so
downstream tools can filter to human-vetted candidates.

**Affects:** `cli.py`, `hypotheses/models.py`

**Implementation:**
- `Hypothesis.approved: Optional[bool]` field (was already in models, now wired up)
- `HypothesisReport.pending_review` property: hypotheses with `approved is None`
- `athanor approve --domain X` CLI command: interactive y/n/s/q loop,
  renders statement / mechanism / falsification criteria / experiment type,
  writes decision back to `hypothesis_report.json`

**Status:** ✅ Implemented (Session 5)

---

### 15. Domain-scoped graph paths
**What:** All domains were overwriting the same `outputs/graphs/concept_graph.json`.
Fixed `_out()` in `cli.py` to scope graphs per-domain: `outputs/graphs/<domain>/`.

**Status:** ✅ Fixed (Session 5) — prerequisite for cross-domain command.

---

## Priority matrix

| Item | Impact | Effort | Status |
|------|--------|--------|--------|
| Structural holes (Burt) | High | Low | ✅ |
| Falsification prompts | High | Low | ✅ |
| Causal bridge_type | High | Low | ✅ |
| Isotropy correction | Medium | Low | ✅ |
| Longevity hallmarks YAML | High | Low | ✅ |
| Bibliometrics / co-citation | High | Medium | ✅ (Session 5) |
| Cross-domain gap finder | High | High | ✅ (Session 5) |
| Hypothesis approval loop | High | Low | ✅ (Session 5) |
| Domain-scoped graph paths | High | Low | ✅ (Session 5) |
| Richer edge types | Medium | Medium | 🔲 |
| DOE fields in experiments | Medium | Low | 🔲 |
| Replication risk scoring | Medium | Low | 🔲 |
| Few-shot prompt examples | High | Medium | 🔲 |
| GNN gap prediction | High | High | 🔲 |
| `athanor report` markdown export | Medium | Low | 🔲 |
| Hypothesis aging / tracking | Medium | Medium | 🔲 |

---

*Last updated: 2025-07-11 (Session 5)*
