# athanor

> *A crucible for automated science.*

Domain-agnostic research pipeline factory. Point it at any field, get a concept graph, gap analysis, and tractable hypotheses.

**Current status:** Stage 1 — Literature Mapper (complete)

---

## The vision

```
Ingest → Map → Gap-find → Hypothesize → Design experiment → Test computationally → Surface → Repeat
```

Four pillars:
1. **Literature ingestion & gap finding** ← Stage 1 lives here
2. **Hypothesis generation**
3. **Ontology building**
4. **Experiment design automation**

Test domain: information theory (mature enough to verify, open enough to have real gaps).
True target: longevity biology, quantum gravity, anything.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/sethc5/athanor.git
cd athanor

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Credentials
cp .env.example .env
# edit .env — add your ANTHROPIC_API_KEY

# 4. Run
jupyter notebook notebooks/stage1_literature_mapper.ipynb
```

---

## Structure

```
athanor/
├── athanor/
│   ├── config.py           # Central config — reads .env
│   ├── ingest/
│   │   ├── arxiv_client.py # Fetch & cache papers from arXiv
│   │   └── parser.py       # Normalise paper text
│   ├── embed/
│   │   └── embedder.py     # Local sentence-transformer embeddings
│   ├── graph/
│   │   ├── models.py       # Concept, Edge, ConceptGraph (Pydantic)
│   │   ├── extractor.py    # Claude → JSON concept/edge extraction
│   │   └── builder.py      # Merge extractions, compute centrality
│   └── viz/
│       └── visualizer.py   # pyvis + plotly renderers
│   └── gaps/
│       ├── models.py       # CandidateGap, GapAnalysis, GapReport (Pydantic)
│       └── finder.py       # Claude → scored research questions
├── notebooks/
│   ├── stage1_literature_mapper.ipynb
│   └── stage2_gap_finder.ipynb
├── data/
│   └── raw/                # Cached arXiv JSON + Claude extractions
└── outputs/
    └── graphs/             # HTML visualizations + graph JSON
```

---

## Changing domain

Edit the `CONFIG` block in notebook §2:

```python
CONFIG = {
    "domain": "longevity biology",
    "arxiv_query": "aging senescence epigenetic clock longevity",
    "max_papers": 15,
    ...
}
```

Delete `data/raw/` caches and re-run. Everything else is automatic.

---

## Stack

| Component | Library |
|-----------|---------|
| Paper fetch | `arxiv` |
| Concept extraction | `anthropic` (claude-opus-4-5) |
| Embeddings | `sentence-transformers` (local, offline) |
| Graph | `networkx` |
| Visualization | `pyvis` + `plotly` |
| Data models | `pydantic` v2 |

---

## Philosophy

> We are code and we don't understand it.  
> Aging is correct code running in an obsolete context.  
> The project is to fork the repo and change the objectives.

Automated science infrastructure is the highest-leverage entry point because it compounds — every domain gets faster once the factory exists.

---

## Roadmap

- [x] Stage 1 — Literature mapper
- [x] Stage 2 — Gap finder (reads Stage 1 output)
- [ ] Stage 3 — Hypothesis generator
- [ ] Semantic Scholar API as alternative/supplement to arXiv
- [ ] Full PDF text extraction (GROBID / pypdf)
- [ ] Domain-specific ontology layer
- [ ] Experiment design automation
