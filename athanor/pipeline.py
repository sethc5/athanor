"""
athanor.pipeline — core pipeline execution logic.

All heavy logic extracted from cli.py so it can be called from tests,
notebooks, or other tools without pulling in Click / Rich / console I/O.

Functions here raise on errors instead of sys.exit-ing.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from athanor.config import cfg, project_root as _root, workspace_root
from athanor.domains import load_domain
from athanor.embed import Embedder
from athanor.gaps import GapFinder, deduplicate_gaps
from athanor.gaps.models import CandidateGap, GapReport
from athanor.graph import GraphBuilder, compute_candidate_gaps
from athanor.graph.models import ConceptGraph
from athanor.hypotheses import HypothesisGenerator
from athanor.hypotheses.critic import HypothesisCritic
from athanor.hypotheses.models import HypothesisReport
from athanor.ingest import (
    ArxivClient,
    SemanticScholarClient,
    parse_papers,
    enrich_papers_with_fulltext,
)
from athanor.ingest.cocitation import compute_biblio_coupling

log = logging.getLogger("athanor.pipeline")


def _get_api_key() -> str:
    """Return the Anthropic API key, raising a clear error if unset."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return key


def _extend_unique(target: list, new_papers: list) -> None:
    """Append *new_papers* to *target*, skipping titles already present."""
    existing = {p.title.lower() for p in target}
    for p in new_papers:
        if p.title.lower() not in existing:
            target.append(p)
            existing.add(p.title.lower())


# ── workspace helpers (testable, no Rich) ────────────────────────────────────

def output_paths(domain: str) -> Dict[str, Path]:
    """Return ``{graphs, gaps, hyps}`` output directories for *domain*."""
    root = workspace_root() / "outputs"
    return {
        "graphs": root / "graphs" / domain,
        "gaps":   root / "gaps" / domain,
        "hyps":   root / "hypotheses" / domain,
    }


# ── feedback loop helpers ────────────────────────────────────────────────────

def _load_approved_keywords(hyp_path: Path) -> List[str]:
    """Extract unique keywords from approved hypotheses (for query enrichment)."""
    if not hyp_path.exists():
        return []
    prior = HypothesisReport.model_validate_json(hyp_path.read_text())
    kw = [
        kw
        for h in prior.hypotheses
        if h.approved is True
        for kw in h.keywords
    ]
    return list(dict.fromkeys(kw))  # preserve order, deduplicate


def _load_approved_statements(hyp_path: Path) -> List[str]:
    """Extract statements from approved hypotheses (for gap exclusion)."""
    if not hyp_path.exists():
        return []
    prior = HypothesisReport.model_validate_json(hyp_path.read_text())
    return [h.statement for h in prior.hypotheses if h.approved is True]


# ── Stage 1 — Literature Mapper ─────────────────────────────────────────────

def run_stage_1(
    dom: Dict[str, Any],
    out: Dict[str, Path],
    *,
    no_cache: bool = False,
    pdf: bool = False,
    s2: bool = False,
    cocite: bool = False,
    workers: int = 8,
    hyp_path_feedback: Optional[Path] = None,
) -> Tuple[ConceptGraph, List[Dict[str, Any]], int]:
    """Run Stage 1: fetch papers → concept graph → candidate gaps.

    Returns ``(concept_graph, candidate_gaps, n_papers)``.
    """
    domain_name = dom["name"]
    graph_path = out["graphs"] / "concept_graph.json"
    gaps_path = out["graphs"] / "candidate_gaps.json"

    papers: list = []

    # arXiv
    arxiv_client = ArxivClient(cache_dir=cfg.data_raw)
    _query_list: List[str] = dom.get("queries") or []
    if not _query_list:
        _single = dom.get("arxiv_query", dom.get("s2_query", ""))
        if _single:
            _query_list = [_single]

    # Approved-hypothesis feedback loop
    if hyp_path_feedback:
        _dedup_kw = _load_approved_keywords(hyp_path_feedback)
        if _dedup_kw:
            _extra_q = " ".join(_dedup_kw[:12])
            if _extra_q not in _query_list:
                _query_list = _query_list + [_extra_q]
                log.info(
                    "Feedback loop: added query from approved hypotheses"
                )

    for _q in _query_list:
        _fetched = arxiv_client.fetch(
            _q, max_results=dom["max_papers"], use_cache=not no_cache,
        )
        _extend_unique(papers, _fetched)

    # Semantic Scholar (optional)
    if s2 or "semantic_scholar" in dom.get("sources", []):
        s2_client = SemanticScholarClient(cache_dir=cfg.data_raw)
        _s2_query_list: List[str] = dom.get("queries") or []
        if not _s2_query_list:
            _s2_single = dom.get("s2_query", dom.get("arxiv_query", ""))
            if _s2_single:
                _s2_query_list = [_s2_single]
        s2_papers: list = []
        for _q in _s2_query_list:
            _s2_fetched = s2_client.fetch(
                _q, max_results=dom["max_papers"], use_cache=not no_cache,
            )
            _extend_unique(s2_papers, _s2_fetched)
        _extend_unique(papers, s2_papers)

    # Seed papers
    seed_ids = [
        s.removeprefix("arxiv:").strip()
        for s in dom.get("seed_papers", [])
        if str(s).startswith("arxiv:")
    ]
    if seed_ids:
        seed = arxiv_client.fetch_by_ids(seed_ids)
        _extend_unique(papers, seed)
        log.info("Added seed paper(s)")

    log.info("Fetched %d papers total", len(papers))

    # Bibliographic coupling — save to disk for downstream use
    if cocite and s2:
        s2_client_cc = SemanticScholarClient(cache_dir=cfg.data_raw)
        s2_ids = [
            p.url.rstrip("/").split("/")[-1]
            for p in papers
            if "semanticscholar.org/paper/" in p.url
        ]
        if s2_ids:
            ref_map = s2_client_cc.fetch_references_batch(s2_ids)
            coupling = compute_biblio_coupling(ref_map)
            coupling_path = out["graphs"] / "biblio_coupling.json"
            coupling_serializable = {
                f"{a}||{b}": score for (a, b), score in coupling.items()
            }
            coupling_path.write_text(json.dumps(coupling_serializable, indent=2))
            log.info(
                "Bibliographic coupling: %d coupled pairs → %s",
                len(coupling), coupling_path,
            )

    # PDF enrichment
    if pdf:
        enrich_papers_with_fulltext(papers, max_papers=dom["max_papers"])

    parsed = parse_papers(papers)

    # Explicit extractor so Stage 1 uses the domain's claude_model and a
    # validated API key from the environment.
    from athanor.graph.extractor import ConceptExtractor

    extractor = ConceptExtractor(
        model=dom.get("claude_model", cfg.model),
        api_key=_get_api_key(),
    )
    builder = GraphBuilder(extractor=extractor)
    _primary_query = (dom.get("queries") or [dom.get("arxiv_query", "")])[0]
    concept_graph = builder.build(
        parsed,
        domain=domain_name,
        query=_primary_query,
        save_path=graph_path,
        max_workers=workers,
        domain_context=dom.get("domain_context", ""),
    )

    # Compute and save candidate gaps
    threshold = dom.get("sparse_sim_threshold", 0.45)
    candidate_gaps = compute_candidate_gaps(concept_graph, threshold)
    gaps_path.write_text(json.dumps(candidate_gaps, indent=2))

    log.info(
        "Stage 1 complete — %d concepts, %d gap candidates",
        len(concept_graph.concepts),
        len(candidate_gaps),
    )
    return concept_graph, candidate_gaps, len(papers)


# ── Stage 2 — Gap Finder ────────────────────────────────────────────────────

def run_stage_2(
    dom: Dict[str, Any],
    out: Dict[str, Path],
    *,
    workers: int = 8,
    hyp_path_feedback: Optional[Path] = None,
) -> GapReport:
    """Run Stage 2: candidate gaps → gap analyses.

    Returns a ``GapReport``.
    """
    domain_name = dom["name"]
    graph_path = out["graphs"] / "concept_graph.json"
    gaps_path = out["graphs"] / "candidate_gaps.json"
    report_path = out["gaps"] / "gap_report.json"

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Missing concept_graph.json — run Stage 1 first"
        )

    concept_graph = ConceptGraph.model_validate_json(graph_path.read_text())
    concept_map = {c.label: c for c in concept_graph.concepts}

    raw_gaps = json.loads(gaps_path.read_text())
    enriched = []
    for g in raw_gaps:
        ca = concept_map.get(g["concept_a"])
        cb = concept_map.get(g["concept_b"])
        enriched.append(
            CandidateGap(
                **g,
                description_a=ca.description if ca else "",
                description_b=cb.description if cb else "",
                papers_a=list(ca.source_papers[:4]) if ca else [],
                papers_b=list(cb.source_papers[:4]) if cb else [],
            )
        )

    log.info("Deduplicating %d candidates…", len(enriched))
    deduped, _ = deduplicate_gaps(enriched)
    log.info("→ %d after deduplication", len(deduped))

    # Feedback loop: avoid regenerating already-approved areas
    _prior_approved_statements: List[str] = []
    if hyp_path_feedback:
        _prior_approved_statements = _load_approved_statements(hyp_path_feedback)
        if _prior_approved_statements:
            log.info(
                "Feedback loop: gap finder will avoid %d approved hypothesis areas",
                len(_prior_approved_statements),
            )

    finder = GapFinder(
        domain=domain_name,
        model=dom.get("claude_model", cfg.model),
        api_key=_get_api_key(),
        max_gaps=dom.get("max_gaps", 15),
        max_workers=workers,
        domain_context=dom.get("domain_context", ""),
        prior_approved=_prior_approved_statements,
    )
    _primary_query = (
        dom.get("queries") or [dom.get("arxiv_query", "")]
    )[0]
    gap_report = finder.analyse(deduped, query=_primary_query)
    report_path.write_text(gap_report.model_dump_json(indent=2))

    log.info(
        "Stage 2 complete — %d gaps analysed → %s",
        len(gap_report.analyses),
        report_path,
    )
    return gap_report


# ── Stage 3 — Hypothesis Generator ──────────────────────────────────────────

def run_stage_3(
    dom: Dict[str, Any],
    out: Dict[str, Path],
    *,
    workers: int = 8,
) -> HypothesisReport:
    """Run Stage 3: gap analyses → hypotheses.

    Returns a ``HypothesisReport``.
    """
    domain_name = dom["name"]
    report_path = out["gaps"] / "gap_report.json"
    hyp_path = out["hyps"] / "hypothesis_report.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing gap_report.json — run Stage 2 first"
        )

    gap_report = GapReport.model_validate_json(report_path.read_text())
    _s3_workers = max(1, workers // 2)

    generator = HypothesisGenerator(
        domain=domain_name,
        model=dom.get("claude_model", cfg.model),
        api_key=_get_api_key(),
        max_tokens=dom.get("max_tokens_hypothesis", 4096),
        max_workers=_s3_workers,
        domain_context=dom.get("domain_context", ""),
    )
    hyp_report = generator.generate(gap_report.ranked)
    hyp_path.write_text(hyp_report.model_dump_json(indent=2))

    log.info(
        "Stage 3 complete — %d hypotheses → %s",
        len(hyp_report.hypotheses),
        hyp_path,
    )
    return hyp_report


# ── Stage 3.5 — Critic Pass ─────────────────────────────────────────────────

def run_critic(
    dom: Dict[str, Any],
    out: Dict[str, Path],
    *,
    workers: int = 4,
) -> HypothesisReport:
    """Run the independent critic pass on existing hypotheses.

    Returns the updated ``HypothesisReport``.
    """
    hyp_path = out["hyps"] / "hypothesis_report.json"
    if not hyp_path.exists():
        raise FileNotFoundError(
            f"No hypothesis report. Run Stage 3 first."
        )

    report = HypothesisReport.model_validate_json(hyp_path.read_text())
    if not report.hypotheses:
        log.warning("No hypotheses to critique.")
        return report

    _s3_workers = max(1, workers)
    critic = HypothesisCritic(
        model=dom.get("claude_model", cfg.model),
        api_key=_get_api_key(),
        max_workers=_s3_workers,
    )
    report = critic.critique(report)
    hyp_path.write_text(report.model_dump_json(indent=2))

    _critiqued = sum(
        1 for h in report.hypotheses if h.critic_novelty is not None
    )
    log.info("Critic pass complete — %d hypotheses re-scored → %s", _critiqued, hyp_path)
    return report


# ── Cross-domain gap analysis ───────────────────────────────────────────────

def run_cross_domain(
    domain_a: str,
    domain_b: str,
    *,
    top: int = 10,
    threshold: float = 0.45,
) -> GapReport:
    """Find structural gaps between two domains.

    Returns the cross-domain ``GapReport``.
    """
    dom_a = load_domain(domain_a)
    dom_b = load_domain(domain_b)
    name_a, name_b = dom_a["name"], dom_b["name"]
    cross_name = f"{name_a}__{name_b}"

    graph_a = output_paths(name_a)["graphs"] / "concept_graph.json"
    graph_b = output_paths(name_b)["graphs"] / "concept_graph.json"
    for p, d in [(graph_a, name_a), (graph_b, name_b)]:
        if not p.exists():
            raise FileNotFoundError(
                f"No concept graph for '{d}' — run Stage 1 first."
            )

    cg_a = ConceptGraph.model_validate_json(graph_a.read_text())
    cg_b = ConceptGraph.model_validate_json(graph_b.read_text())

    embedder = Embedder()
    texts_a = [f"{c.label}. {c.description}" for c in cg_a.concepts]
    texts_b = [f"{c.label}. {c.description}" for c in cg_b.concepts]
    embs_a = embedder.embed(texts_a, center=False)
    embs_b = embedder.embed(texts_b, center=False)

    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embs_a, embs_b)

    bridges: list[tuple[float, Any, Any]] = []
    for i, ca in enumerate(cg_a.concepts):
        for j, cb in enumerate(cg_b.concepts):
            s = float(sim[i, j])
            if s >= threshold:
                bridges.append((s, ca, cb))
    bridges.sort(key=lambda x: x[0], reverse=True)
    bridges = bridges[: top * 4]

    if not bridges:
        log.warning("No bridges found. Try lowering threshold.")
        return GapReport(
            domain=cross_name,
            query=f"cross-domain: {name_a} ↔ {name_b}",
        )

    candidates = [
        CandidateGap(
            concept_a=ca.label,
            concept_b=cb.label,
            similarity=s,
            graph_distance=999,
            structural_hole_score=0.5,
            description_a=ca.description,
            description_b=cb.description,
            papers_a=list(ca.source_papers[:4]),
            papers_b=list(cb.source_papers[:4]),
        )
        for s, ca, cb in bridges
    ]

    combined_context = (
        f"Cross-domain bridge between:\n"
        f"DOMAIN A ({name_a}): {dom_a.get('description', '')}\n"
        f"DOMAIN B ({name_b}): {dom_b.get('description', '')}\n"
        f"Focus on mechanisms that could translate concepts or methods between these fields."
    )
    combined_display = f"{dom_a['display']} ↔ {dom_b['display']}"

    # Prefer the more capable model when domains disagree.
    # claude-sonnet < claude-opus ordering (alphabetical happens to work for
    # the Anthropic naming scheme: "opus" > "sonnet" > "haiku").
    model_a = dom_a.get("claude_model", cfg.model)
    model_b = dom_b.get("claude_model", cfg.model)
    cross_model = max(model_a, model_b)

    finder = GapFinder(
        domain=combined_display,
        model=cross_model,
        api_key=_get_api_key(),
        max_gaps=top,
        max_workers=4,
        domain_context=combined_context,
    )
    gap_report = finder.analyse(candidates[:top], query=f"cross-domain: {name_a} ↔ {name_b}")

    out_path = workspace_root() / "outputs" / "gaps" / cross_name
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "gap_report.json"
    report_path.write_text(gap_report.model_dump_json(indent=2))

    log.info(
        "Cross-domain analysis complete — %d bridges → %s",
        len(gap_report.analyses),
        report_path,
    )
    return gap_report
