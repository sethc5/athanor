"""
athanor CLI — run the full pipeline headlessly from the command line.

Usage:
    athanor run --domain information_theory
    athanor run --domain longevity_biology --max-papers 20 --stages 1,2,3
    athanor run --domain information_theory --stages 1      # Stage 1 only
    athanor list-domains
    athanor status                                           # show outputs for a domain

Pipeline stages:
    1 = Literature mapper  (arXiv fetch → concept graph)
    2 = Gap finder         (concept graph → ranked research questions)
    3 = Hypothesis gen     (research questions → experiments)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Make sure athanor package is importable when run from project root
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

console = Console()
log = logging.getLogger("athanor.cli")


# ── helpers ──────────────────────────────────────────────────────────────────

def _out(domain: str) -> dict:
    """Return output path dict for a domain."""
    root = _root / "outputs"
    return {
        "graphs": root / "graphs",
        "gaps":   root / "gaps" / domain,
        "hyps":   root / "hypotheses" / domain,
    }


# ── CLI root ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Athanor — domain-agnostic automated science infrastructure."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s — %(message)s")


# ── list-domains ──────────────────────────────────────────────────────────────

@cli.command("list-domains")
def list_domains_cmd() -> None:
    """List all available domain configurations."""
    from athanor.domains import list_domains
    domains = list_domains()
    if not domains:
        console.print("[yellow]No domain configs found in domains/[/]")
        return
    console.print("[bold green]Available domains:[/]")
    for d in domains:
        console.print(f"  • {d}")


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", "-d", required=True, help="Domain name (e.g. information_theory)")
def status(domain: str) -> None:
    """Show pipeline output status for a domain."""
    paths = _out(domain)
    g = paths["graphs"] / "concept_graph.json"
    r = paths["gaps"] / "gap_report.json"
    h = paths["hyps"] / "hypothesis_report.json"

    def check(label, path):
        exists = path.exists()
        color = "green" if exists else "red"
        size = f"({path.stat().st_size // 1024}KB)" if exists else "(missing)"
        console.print(f"  [{color}]{'✓' if exists else '✗'}[/] {label}: {path} {size}")

    console.print(f"[bold]Pipeline status — {domain}[/]")
    check("Stage 1 concept graph", g)
    check("Stage 2 gap report", r)
    check("Stage 3 hypothesis report", h)


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain",      "-d", required=True, help="Domain name or path to YAML config")
@click.option("--stages",      "-s", default="1,2,3", show_default=True,
              help="Comma-separated stages to run, e.g. '1,2' or '3'")
@click.option("--max-papers",  type=int, default=None, help="Override max_papers from domain config")
@click.option("--max-gaps",    type=int, default=None, help="Override max_gaps for Stage 2")
@click.option("--no-cache",    is_flag=True, help="Ignore all cached outputs and re-fetch/re-extract")
@click.option("--pdf",         is_flag=True, help="Download full PDF text (Stage 1 only)")
@click.option("--s2",          is_flag=True, help="Also fetch from Semantic Scholar")
def run(
    domain: str,
    stages: str,
    max_papers: int,
    max_gaps: int,
    no_cache: bool,
    pdf: bool,
    s2: bool,
) -> None:
    """Run the full athanor pipeline for a domain."""
    from athanor.domains import load_domain
    from athanor.config import cfg

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/] ANTHROPIC_API_KEY not set. Copy .env.example → .env")
        raise SystemExit(1)

    dom = load_domain(domain)
    domain_name = dom["name"]
    stage_list = [int(s.strip()) for s in stages.split(",")]

    if max_papers:
        dom["max_papers"] = max_papers
    if max_gaps:
        dom["max_gaps"] = max_gaps

    out = _out(domain_name)
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]{dom['display']}[/]\n"
        f"Stages: {stage_list} | Papers: {dom['max_papers']} | "
        f"Gaps: {dom.get('max_gaps',15)} | PDF: {pdf} | S2: {s2}",
        title="Athanor Pipeline",
        border_style="green",
    ))

    graph_path = out["graphs"] / "concept_graph.json"
    gaps_path  = out["graphs"] / "candidate_gaps.json"
    report_path = out["gaps"] / "gap_report.json"
    hyp_path   = out["hyps"] / "hypothesis_report.json"

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if 1 in stage_list:
        console.rule("[bold]Stage 1 — Literature Mapper[/]")
        from athanor.ingest import ArxivClient, SemanticScholarClient, parse_papers, enrich_papers_with_fulltext
        from athanor.graph import GraphBuilder

        papers = []

        # arXiv
        arxiv_client = ArxivClient(cache_dir=cfg.data_raw)
        papers.extend(arxiv_client.fetch(
            dom.get("arxiv_query", dom.get("s2_query", "")),
            max_results=dom["max_papers"],
            use_cache=not no_cache,
        ))

        # Semantic Scholar (optional)
        if s2 or "semantic_scholar" in dom.get("sources", []):
            s2_client = SemanticScholarClient(cache_dir=cfg.data_raw)
            s2_papers = s2_client.fetch(
                dom.get("s2_query", dom.get("arxiv_query", "")),
                max_results=dom["max_papers"],
                use_cache=not no_cache,
            )
            # Deduplicate by title
            existing_titles = {p.title.lower() for p in papers}
            papers.extend(p for p in s2_papers if p.title.lower() not in existing_titles)

        console.print(f"[green]✓ Fetched {len(papers)} papers total[/]")

        # PDF enrichment
        if pdf:
            console.print("[yellow]Downloading full-text PDFs…[/]")
            enrich_papers_with_fulltext(papers, max_papers=dom["max_papers"])

        parsed = parse_papers(papers)
        builder = GraphBuilder()
        concept_graph = builder.build(
            parsed,
            domain=domain_name,
            query=dom.get("arxiv_query", ""),
            save_path=graph_path,
        )

        # Also save candidate gaps
        from athanor.graph.models import ConceptGraph
        from athanor.embed import Embedder
        from sklearn.metrics.pairwise import cosine_similarity
        G = concept_graph.to_networkx()
        import networkx as nx
        embedder = Embedder()
        concept_texts = [c.label + ". " + c.description for c in concept_graph.concepts]
        concept_embs = embedder.embed(concept_texts)
        csim = cosine_similarity(concept_embs)
        labels = [c.label for c in concept_graph.concepts]
        threshold = dom.get("sparse_sim_threshold", 0.45)
        candidate_gaps = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if csim[i, j] < threshold:
                    continue
                try:
                    dist = nx.shortest_path_length(G, labels[i], labels[j])
                except nx.NetworkXNoPath:
                    dist = 999
                if dist > 2:
                    candidate_gaps.append({"concept_a": labels[i], "concept_b": labels[j],
                                           "similarity": float(csim[i, j]), "graph_distance": dist})
        candidate_gaps.sort(key=lambda x: x["similarity"], reverse=True)
        gaps_path.write_text(json.dumps(candidate_gaps, indent=2))
        console.print(f"[green]✓ Stage 1 complete[/] — {len(concept_graph.concepts)} concepts, {len(candidate_gaps)} gap candidates")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if 2 in stage_list:
        console.rule("[bold]Stage 2 — Gap Finder[/]")
        if not graph_path.exists():
            console.print("[red]Missing concept_graph.json — run Stage 1 first[/]"); raise SystemExit(1)

        from athanor.graph.models import ConceptGraph
        from athanor.gaps import GapFinder, CandidateGap, deduplicate_gaps

        concept_graph = ConceptGraph.model_validate_json(graph_path.read_text())
        concept_map = {c.label: c for c in concept_graph.concepts}

        raw_gaps = json.loads(gaps_path.read_text())
        enriched = []
        for g in raw_gaps:
            ca = concept_map.get(g["concept_a"])
            cb = concept_map.get(g["concept_b"])
            from athanor.gaps.models import CandidateGap
            enriched.append(CandidateGap(
                **g,
                description_a=ca.description if ca else "",
                description_b=cb.description if cb else "",
                papers_a=list(ca.source_papers[:4]) if ca else [],
                papers_b=list(cb.source_papers[:4]) if cb else [],
            ))

        console.print(f"Deduplicating {len(enriched)} candidates…")
        deduped, _ = deduplicate_gaps(enriched)
        console.print(f"[green]→ {len(deduped)} after deduplication[/]")

        finder = GapFinder(
            domain=domain_name,
            model=dom.get("claude_model", cfg.model),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_gaps=dom.get("max_gaps", 15),
        )
        gap_report = finder.analyse(deduped, query=dom.get("arxiv_query", ""))
        report_path.write_text(gap_report.model_dump_json(indent=2))
        console.print(f"[green]✓ Stage 2 complete[/] — {len(gap_report.analyses)} gaps analysed, saved → {report_path}")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if 3 in stage_list:
        console.rule("[bold]Stage 3 — Hypothesis Generator[/]")
        if not report_path.exists():
            console.print("[red]Missing gap_report.json — run Stage 2 first[/]"); raise SystemExit(1)

        from athanor.gaps.models import GapReport
        from athanor.hypotheses import HypothesisGenerator

        gap_report = GapReport.model_validate_json(report_path.read_text())
        generator = HypothesisGenerator(
            domain=domain_name,
            model=dom.get("claude_model", cfg.model),
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        hyp_report = generator.generate(gap_report.ranked)
        hyp_path.write_text(hyp_report.model_dump_json(indent=2))
        console.print(f"[green]✓ Stage 3 complete[/] — {len(hyp_report.hypotheses)} hypotheses, saved → {hyp_path}")

        # Quick summary
        top = hyp_report.top
        if top:
            best = top[0]
            console.print(f"\n[bold green]Top hypothesis:[/] {best.statement}")
            if best.experiment:
                comp = "Computational ✓" if best.experiment.computational else "Wet lab required"
                console.print(f"[dim]{comp} | Effort: {best.experiment.estimated_effort}[/]")

    console.print("\n[bold green]Pipeline complete.[/]")
