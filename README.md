# athanor

> *A crucible for automated science.*

Domain-agnostic research pipeline factory. Point it at any scientific field — get a concept graph, gap analysis, falsifiable hypotheses, and independent critic scores. Works across physics, biology, computer science, or any domain you define.

---

## Pipeline stages

```
Stage 1 — Literature Mapper     Fetch papers (arXiv + Semantic Scholar), extract concepts,
                                 build an embedding-weighted concept graph with Burt structural holes.

Stage 2 — Gap Finder            Identify structural gaps (high embedding similarity,
                                 high graph distance), score each for novelty / tractability / impact.

Stage 3 — Hypothesis Generator  Convert top gaps into Popperian falsifiable hypotheses
                                 with concrete experiment designs (computational preferred).

Stage 3.5 — Critic Pass         Independent re-scoring of each hypothesis; blended final score
                                 to reduce single-model anchoring bias.

Approve → Seed Loop             Manually approve/reject hypotheses; approved keywords feed back
                                 into the next Stage 1 query for iterative refinement.

Cross-Domain                    Compare concept graphs across domains to find bridging opportunities.
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/sethc5/athanor.git && cd athanor
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Credentials
cp .env.example .env          # add ANTHROPIC_API_KEY

# 3. Select a workspace
export ATHANOR_WORKSPACE=sandbox     # or string_physics, longevity, etc.

# 4. Run the pipeline
athanor run -d information_theory              # full pipeline (Stages 1-3)
athanor run -d information_theory -s 1         # Stage 1 only
athanor run -d information_theory --critique   # Stages 1-3 + critic pass
```

---

## Workspaces & domains

Each workspace is an isolated directory under `workspaces/`:

```
workspaces/
├── sandbox/
│   └── domains/                # YAML domain configs
│       ├── information_theory.yaml
│       └── quantum_computing.yaml
├── string_physics/
│   └── domains/
│       ├── string_landscape.yaml
│       ├── moduli_periods.yaml
│       └── ...
└── longevity/
    └── domains/
        └── longevity_biology.yaml
```

Set the active workspace:

```bash
export ATHANOR_WORKSPACE=string_physics
# or
athanor -w string_physics run -d string_landscape
```

### Domain YAML

```yaml
name: string_landscape
display: "String Landscape & Calabi-Yau Compactifications"
description: >
  The search for Standard Model physics from string/F-theory
  compactifications on Calabi-Yau 3-folds. ...

arxiv_query: >
  Calabi-Yau compactification Standard Model string theory F-theory ...
s2_query: >
  Calabi-Yau three-fold Standard Model string compactification ...

max_papers: 20
sources:
  - arxiv
  - semantic_scholar

max_gaps: 20
sparse_sim_threshold: 0.38
domain_context: >
  This is a mathematical physics domain combining algebraic geometry
  and string theory. Precision is essential. ...
```

---

## CLI commands

| Command | Description |
|---------|-------------|
| `athanor run -d DOMAIN` | Run pipeline (Stages 1-3, optionally `--critique` for 3.5) |
| `athanor list-domains` | List available domain configs in the active workspace |
| `athanor status` | Show pipeline output status for all domains |
| `athanor critique -d DOMAIN` | Run independent critic pass on existing hypotheses |
| `athanor approve -d DOMAIN` | Interactively approve/reject hypotheses |
| `athanor report -d DOMAIN` | Render a Markdown report of hypotheses |
| `athanor search QUERY` | Search hypotheses across all domains by keyword/score/filter |
| `athanor cross-domain -a D1 -b D2` | Find bridging gaps between two domains |

Key flags for `run`:

```
--stages 1,2,3    # Pick which stages to execute
--max-papers N    # Override paper count from domain config
--max-gaps N      # Override gap count for Stage 2
--s2              # Also fetch from Semantic Scholar
--cocite          # Add bibliographic coupling edges (requires --s2)
--pdf             # Download full PDF text
--workers N       # Parallel Claude workers (default 8)
--critique        # Run Stage 3.5 after Stage 3
--no-cache        # Ignore cached outputs
```

---

## Project structure

```
athanor/
├── athanor/
│   ├── cli.py              # Click CLI (thin wrapper)
│   ├── pipeline.py         # Core pipeline execution logic
│   ├── config.py           # Central config — reads .env
│   ├── llm_utils.py        # Claude JSON calling, retry, fence-stripping
│   ├── domains/            # Domain loader (YAML → dict)
│   ├── ingest/
│   │   ├── arxiv_client.py # Fetch & cache papers from arXiv
│   │   ├── semantic_scholar.py  # Semantic Scholar API client
│   │   ├── parser.py       # Text normalisation (preserves LaTeX math)
│   │   └── pdf.py          # Full-text PDF extraction
│   ├── embed/
│   │   └── embedder.py     # Local sentence-transformer embeddings + cache
│   ├── graph/
│   │   ├── models.py       # Concept, Edge, ConceptGraph (Pydantic)
│   │   ├── extractor.py    # Claude → JSON concept/edge extraction
│   │   └── builder.py      # Merge extractions, centrality, Burt constraint
│   ├── gaps/
│   │   ├── models.py       # CandidateGap, GapAnalysis, GapReport
│   │   └── finder.py       # Claude → scored research questions
│   ├── hypotheses/
│   │   ├── models.py       # Hypothesis, ExperimentDesign, HypothesisReport
│   │   ├── generator.py    # Claude → falsifiable hypotheses + experiments
│   │   └── critic.py       # Independent re-scoring (Stage 3.5)
│   └── viz/
│       └── visualizer.py   # pyvis + plotly renderers
├── tests/                  # pytest suite (69 tests)
├── workspaces/             # Isolated workspace directories
├── domains/                # Legacy domain configs (use workspaces/)
└── pyproject.toml
```

---

## Stack

| Component | Library |
|-----------|---------|
| LLM | `anthropic` (Claude) |
| Paper fetch | `arxiv` + `requests` (Semantic Scholar) |
| Embeddings | `sentence-transformers` (local, offline) |
| Graph analysis | `networkx` (Burt structural holes) |
| Data models | `pydantic` v2 |
| CLI | `click` + `rich` |
| Visualization | `pyvis` + `plotly` |
| Testing | `pytest` |

---

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

---

## Philosophy

Automated science infrastructure is the highest-leverage entry point because it compounds — every domain gets faster once the factory exists.
