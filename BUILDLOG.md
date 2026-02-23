# Athanor — Build Log

---

## Session 3 — 2026-02-22 | commit `4ff79a7`

### Completed

| Item | Module / File | Notes |
|------|--------------|-------|
| Stage 1 — Literature Mapper | `notebooks/stage1_literature_mapper.ipynb` | arXiv fetch → embed → concept graph → viz |
| Stage 2 — Gap Finder | `notebooks/stage2_gap_finder.ipynb` | sparse graph → Claude gap analysis → ranked report |
| Stage 3 — Hypothesis Generator | `notebooks/stage3_hypothesis_generator.ipynb` | human review → Claude hypothesis + experiment design → report |
| `athanor.hypotheses` module | `athanor/hypotheses/` | `Hypothesis`, `ExperimentDesign`, `HypothesisReport`, `HypothesisGenerator` |
| Semantic Scholar client | `athanor/ingest/semantic_scholar.py` | S2 Graph API, caching, rate limiting, `S2_API_KEY` env var |
| PDF full-text extraction | `athanor/ingest/pdf.py` | pypdf, arXiv PDF URL builder, graceful fallback to abstract |
| Gap deduplication | `athanor/gaps/dedup.py` | greedy cosine similarity clustering (threshold=0.85) |
| Human-in-loop approval | `athanor/gaps/models.py`, `athanor/hypotheses/models.py` | `approved: Optional[bool]` field on `GapAnalysis` + `Hypothesis` |
| Domain YAML configs | `domains/information_theory.yaml`, `domains/longevity_biology.yaml` | query, max_papers, seed_concepts, domain_context |
| Domain loader | `athanor/domains/__init__.py` | `load_domain(name_or_path)` resolves from `domains/` dir |
| CLI | `athanor/cli.py` | `athanor run --domain X --stages 1,2,3 [--pdf] [--s2] [--no-cache]` |
| Package entry point | `pyproject.toml` | `athanor = "athanor.cli:cli"`, `pip install -e .` |
| Updated deps | `requirements.txt`, `.env.example` | Added: pypdf, requests, pyyaml, click, scikit-learn, S2_API_KEY |

### Pending (next session)

- [ ] **Install deps** — `.venv-1` is empty; run `pip install -r requirements.txt` before notebooks will execute
- [ ] **Pylance errors** — all 175 are unresolved imports caused by missing venv packages; will clear after install
- [ ] **Smoke test** — `python -c "import anthropic, arxiv, pypdf, click, yaml, sklearn; print('OK')"`
- [ ] **Stage 4 ideas** — experiment prioritization, resource estimation, collaboration matching

### Pipeline state (end of session)

```
Ingest (arXiv + S2 + PDF)
  → embed (sentence-transformers)
  → concept graph (networkx + Claude)
  → gap finder (sparse similarity + Claude)
  → dedup gaps (cosine clustering)
  → human-in-loop approval
  → hypothesis generator (Claude)
  → experiment design (Claude)
  → hypothesis report (JSON)
```

### Architecture snapshot

```
athanor/
├── __init__.py
├── config.py               # cfg singleton, env vars
├── cli.py                  # Click CLI entry point
├── domains/                # YAML domain loader
├── ingest/
│   ├── arxiv_client.py
│   ├── parser.py
│   ├── semantic_scholar.py # NEW
│   └── pdf.py              # NEW
├── embed/embedder.py
├── graph/
│   ├── models.py
│   ├── extractor.py
│   └── builder.py
├── gaps/
│   ├── models.py           # GapAnalysis.approved field added
│   ├── finder.py
│   └── dedup.py            # NEW
├── hypotheses/             # NEW module
│   ├── models.py
│   └── generator.py
└── viz/visualizer.py

domains/
├── information_theory.yaml
└── longevity_biology.yaml

notebooks/
├── stage1_literature_mapper.ipynb
├── stage2_gap_finder.ipynb
└── stage3_hypothesis_generator.ipynb
```

---

## Session 2 — earlier | commit `b557e88`

- Built `athanor.gaps` module: `CandidateGap`, `GapAnalysis`, `GapReport`, `GapFinder`
- Built `stage2_gap_finder.ipynb` (8 sections)

---

## Session 1 — earlier | commit `a71d7ef`

- Initialized project, GitHub repo `sethc5/athanor`
- Built `athanor` package: `config`, `ingest`, `embed`, `graph`, `viz`
- Built `stage1_literature_mapper.ipynb` (9 sections)
- Created `.env.example`, `.gitignore`, `requirements.txt`, `README.md`
