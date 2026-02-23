"""
athanor CLI — run the full pipeline headlessly from the command line.

Usage:
    athanor run --domain information_theory
    athanor run --domain longevity_biology --max-papers 20 --stages 1,2,3
    athanor run --domain information_theory --stages 1      # Stage 1 only
    athanor list-domains
    athanor status --domain longevity_biology
    athanor approve --domain longevity_biology              # review hypotheses
    athanor cross-domain --domain-a longevity_biology --domain-b information_theory

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
    """Return output path dict for a domain (all paths are domain-scoped)."""
    root = _root / "outputs"
    return {
        "graphs": root / "graphs" / domain,
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
@click.option("--cocite",      is_flag=True, help="Add bibliographic coupling edges (requires --s2)")
@click.option("--workers",     "-w", type=int, default=2, show_default=True,
              help="Parallel Claude workers for Stages 1+2 (Stage 3 uses max(1, workers//2) due to larger token budget)")
def run(
    domain: str,
    stages: str,
    max_papers: int,
    max_gaps: int,
    no_cache: bool,
    pdf: bool,
    s2: bool,
    cocite: bool,
    workers: int,
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
    # Domain YAML can override CLI --workers (useful for token-heavy domains)
    workers = dom.get("max_workers", workers)

    out = _out(domain_name)
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]{dom['display']}[/]\n"
        f"Stages: {stage_list} | Papers: {dom['max_papers']} | "
        f"Gaps: {dom.get('max_gaps',15)} | Workers: {workers} | PDF: {pdf} | S2: {s2} | co-cite: {cocite}",
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

        # Seed papers (specific arXiv IDs pinned in domain YAML)
        seed_ids = [
            s.removeprefix("arxiv:").strip()
            for s in dom.get("seed_papers", [])
            if str(s).startswith("arxiv:")
        ]
        if seed_ids:
            console.print(f"[yellow]Fetching {len(seed_ids)} seed paper(s) by arXiv ID…[/]")
            seed = arxiv_client.fetch_by_ids(seed_ids)
            existing_titles = {p.title.lower() for p in papers}
            added = [p for p in seed if p.title.lower() not in existing_titles]
            papers.extend(added)
            console.print(f"[green]✓ Added {len(added)} seed paper(s)[/]")

        console.print(f"[green]✓ Fetched {len(papers)} papers total[/]")

        # Bibliographic coupling (optional co-citation seeding)
        if cocite and s2:
            from athanor.ingest.cocitation import compute_biblio_coupling
            console.print("[yellow]Fetching reference lists for bibliographic coupling…[/]")
            s2_client_cc = SemanticScholarClient(cache_dir=cfg.data_raw)
            s2_ids = [
                p.url.rstrip("/").split("/")[-1]
                for p in papers
                if "semanticscholar.org/paper/" in p.url
            ]
            if s2_ids:
                ref_map = s2_client_cc.fetch_references_batch(s2_ids)
                coupling = compute_biblio_coupling(ref_map)
                console.print(f"[green]✓ Bibliographic coupling: {len(coupling)} coupled pairs[/]")
            else:
                console.print("[dim]No Semantic Scholar IDs in fetched papers — skipping co-citation[/]")

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
            max_workers=workers,
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
        concept_list = concept_graph.concepts
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if csim[i, j] < threshold:
                    continue
                try:
                    dist = nx.shortest_path_length(G, labels[i], labels[j])
                except nx.NetworkXNoPath:
                    dist = 999
                if dist > 2:
                    # Structural hole score: high when both endpoints are brokers
                    # (low Burt constraint) straddling disconnected clusters
                    sh_a = 1.0 - concept_list[i].burt_constraint
                    sh_b = 1.0 - concept_list[j].burt_constraint
                    sh_score = round((sh_a + sh_b) / 2, 4)
                    candidate_gaps.append({
                        "concept_a": labels[i],
                        "concept_b": labels[j],
                        "similarity": float(csim[i, j]),
                        "graph_distance": dist,
                        "structural_hole_score": sh_score,
                    })
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
            max_workers=workers,
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
        # Stage 3 uses 4096 tokens/request — cap workers to stay under 10K OPM Haiku limit
        _s3_workers = max(1, workers // 2)
        generator = HypothesisGenerator(
            domain=domain_name,
            model=dom.get("claude_model", cfg.model),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=dom.get("max_tokens_hypothesis", 4096),
            max_workers=_s3_workers,
        )
        hyp_report = generator.generate(gap_report.ranked)
        hyp_path.write_text(hyp_report.model_dump_json(indent=2))
        console.print(f"[green]✓ Stage 3 complete[/] — {len(hyp_report.hypotheses)} hypotheses, saved → {hyp_path}")

        # Quick summary
        top = hyp_report.top(3)
        if top:
            best = top[0]
            console.print(f"\n[bold green]Top hypothesis:[/] {best.statement}")
            if best.experiment:
                comp = "Computational ✓" if best.experiment.computational else "Wet lab required"
                console.print(f"[dim]{comp} | Effort: {best.experiment.estimated_effort}[/]")

    console.print("\n[bold green]Pipeline complete.[/]")

# ── report command ──────────────────────────────────────────────────────────
@cli.command()
@click.option("--domain", "-d", required=True, help="Domain name")
@click.option("--top", "-n", default=10, show_default=True, help="Max hypotheses to include")
@click.option("--approved-only", is_flag=True, help="Only include approved hypotheses")
@click.option("--out", "-o", default=None, help="Output .md file path (default: print to stdout)")
def report(domain: str, top: int, approved_only: bool, out: str) -> None:
    """Render a Markdown report of hypotheses for a domain."""
    from datetime import datetime
    from athanor.hypotheses.models import HypothesisReport as HR

    hyp_path = _out(domain)["hyps"] / "hypothesis_report.json"
    if not hyp_path.exists():
        console.print(f"[red]No hypotheses for '{domain}'. Run Stage 3 first.[/]")
        raise SystemExit(1)

    rep = HR.model_validate_json(hyp_path.read_text())
    candidates = [
        h for h in rep.ranked
        if not approved_only or h.approved is True
    ][:top]

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    title = domain.replace("_", " ").title()

    lines = [
        f"# Athanor — {title} Hypotheses",
        f"*{date_str} | `{domain}` | "
        f"{len(rep.hypotheses)} total | {len(candidates)} shown"
        + (" (approved only)" if approved_only else "") + "*",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| # | Gap | Score | N | R | I | Rep. Risk | Compute |",
        "|---|-----|-------|---|---|---|-----------|---------|" ,
    ]
    for i, h in enumerate(candidates, 1):
        comp = "\u2713" if h.experiment and h.experiment.computational else "\u2717"
        label = {True: " \u2705", False: " \u274c", None: ""}.get(h.approved, "")
        risk_icon = {"low": "\U0001f7e2", "high": "\U0001f534"}.get(getattr(h, "replication_risk", "medium"), "\U0001f7e1")
        lines.append(
            f"| {i} | {h.gap_concept_a} \u2194 {h.gap_concept_b}{label} | "
            f"{h.composite_score:.1f} | {h.novelty} | {h.rigor} | {h.impact} | "
            f"{risk_icon} {getattr(h, 'replication_risk', 'medium')} | {comp} |"
        )

    lines += ["", "---", ""]

    for i, h in enumerate(candidates, 1):
        approved_tag = {True: " \u2705 Approved", False: " \u274c Rejected", None: ""}.get(h.approved, "")
        comp_label = "Computational" if h.experiment and h.experiment.computational else "Requires wet-lab"
        lines += [
            f"## {i}. {h.gap_concept_a} \u2194 {h.gap_concept_b}{approved_tag}",
            f"**Score:** {h.composite_score:.1f}  "
            f"(N\u202f{h.novelty} \u00b7 R\u202f{h.rigor} \u00b7 I\u202f{h.impact})  |  "
            f"{comp_label}  |  Replication risk: **{getattr(h, 'replication_risk', 'medium')}**",
            "",
            f"**Hypothesis:** {h.statement}",
            "",
            f"**Mechanism:** {h.mechanism}",
            "",
            f"**Prediction:** {h.prediction}",
            "",
            f"**Falsification criterion:** {h.falsification_criteria}",
            "",
        ]
        if h.minimum_effect_size:
            lines += [f"**Min. effect size:** {h.minimum_effect_size}", ""]
        if h.experiment:
            e = h.experiment
            lines += ["### Experiment", "", f"*{e.approach}*", ""]
            for step in e.steps:
                lines.append(f"- {step}")
            if e.steps:
                lines.append("")
            if e.tools:
                lines += [f"**Tools:** {', '.join(e.tools)}", ""]
            if e.statistical_test:
                lines += [f"**Statistical test:** {e.statistical_test}", ""]
            if e.minimum_detectable_effect:
                lines += [f"**Min. detectable effect:** {e.minimum_detectable_effect}", ""]
            if e.statistical_power_notes:
                lines += [f"**Power / sample size:** {e.statistical_power_notes}", ""]
            if e.estimated_effort:
                lines += [f"**Effort:** {e.estimated_effort}", ""]
            if e.requires_followup:
                lines += [f"**Wet-lab follow-up:** {e.requires_followup}", ""]
        lines += ["---", ""]

    md = "\n".join(lines)
    if out:
        from pathlib import Path as _P
        _P(out).write_text(md)
        console.print(f"[green]Report written \u2192 {out}[/]")
    else:
        console.print(md)


# ── search command ──────────────────────────────────────────────────────────
@cli.command()
@click.argument("query", default="")
@click.option("--domain", "-d", default=None, help="Restrict to one domain")
@click.option("--min-score", default=0.0, show_default=True, help="Min composite score")
@click.option("--approved-only", is_flag=True, help="Only approved hypotheses")
@click.option("--risk", default=None, type=click.Choice(["low", "medium", "high"]), help="Filter by replication risk")
@click.option("--compute", is_flag=True, help="Only computational experiments")
def search(query: str, domain: str, min_score: float, approved_only: bool, risk: str, compute: bool) -> None:
    """Search hypotheses across all domains by keyword, score, or filter."""
    from athanor.hypotheses.models import HypothesisReport

    hyp_root = _root / "outputs" / "hypotheses"
    if not hyp_root.exists():
        console.print("[red]No hypothesis outputs found. Run Stage 3 for at least one domain.[/]")
        return

    domain_dirs = (
        [hyp_root / domain] if domain else sorted(hyp_root.iterdir())
    )

    results: list[tuple[str, object]] = []  # (domain_name, Hypothesis)
    for d in domain_dirs:
        report_file = d / "hypothesis_report.json"
        if not report_file.exists():
            continue
        rep = HypothesisReport.model_validate_json(report_file.read_text())
        for h in rep.ranked:
            if h.composite_score < min_score:
                continue
            if approved_only and h.approved is not True:
                continue
            if risk and getattr(h, "replication_risk", "medium") != risk:
                continue
            if compute and not (h.experiment and h.experiment.computational):
                continue
            if query:
                needle = query.lower()
                haystack = " ".join([
                    h.statement, h.mechanism, h.gap_concept_a,
                    h.gap_concept_b, " ".join(h.keywords),
                ]).lower()
                if needle not in haystack:
                    continue
            results.append((d.name, h))

    if not results:
        console.print(f"[yellow]No hypotheses match the search.[/]")
        return

    console.print(f"[bold]{len(results)} result(s)[/]\n")
    for dom_name, h in results:
        risk_icon = {"low": "\U0001f7e2", "high": "\U0001f534"}.get(getattr(h, "replication_risk", "medium"), "\U0001f7e1")
        comp_icon = "\u2713" if h.experiment and h.experiment.computational else "\u2717"
        appr_icon = {True: " \u2705", False: " \u274c", None: ""}.get(h.approved, "")
        console.print(
            f"[dim]{dom_name}[/]  [cyan]{h.gap_concept_a} \u2194 {h.gap_concept_b}[/]{appr_icon}"
            f"  score=[bold]{h.composite_score:.1f}[/]  {risk_icon}  compute={comp_icon}"
        )
        console.print(f"  {h.statement[:120]}\u2026" if len(h.statement) > 120 else f"  {h.statement}")
        console.print()


# ── approve command ──────────────────────────────────────────────────────────
@cli.command()
@click.option("--domain", "-d", required=True, help="Domain name")
@click.option("--all", "show_all", is_flag=True, help="Re-review already-reviewed hypotheses")
def approve(domain: str, show_all: bool) -> None:
    """Interactively approve or reject hypotheses for a domain."""
    from athanor.hypotheses.models import HypothesisReport
    hyp_path = _out(domain)["hyps"] / "hypothesis_report.json"
    if not hyp_path.exists():
        console.print(f"[red]No hypothesis report found for '{domain}'. Run Stage 3 first.[/]")
        raise SystemExit(1)
    report = HypothesisReport.model_validate_json(hyp_path.read_text())
    candidates = list(report.hypotheses if show_all else report.pending_review)
    if not candidates:
        console.print("[green]All hypotheses already reviewed. Use --all to re-review.[/]")
        return
    candidates.sort(key=lambda h: h.composite_score, reverse=True)
    console.print(
        f"[bold]Reviewing {len(candidates)} hypothesis(es) for [cyan]{domain}[/][/] "
        "([bold]y[/]=approve  [bold]n[/]=reject  [bold]s[/]=skip  [bold]q[/]=quit)\n"
    )
    approved_n = rejected_n = skipped_n = 0
    for i, hyp in enumerate(candidates, 1):
        console.rule(
            f"[bold]#{i}/{len(candidates)}  Score {hyp.composite_score:.1f}  "
            f"N:{hyp.novelty} R:{hyp.rigor} I:{hyp.impact}[/]"
        )
        console.print(f"[cyan bold]{hyp.gap_concept_a} \u2194 {hyp.gap_concept_b}[/]")
        console.print(f"\n[bold]Hypothesis:[/] {hyp.statement}\n")
        console.print(f"[bold]Mechanism:[/]  {hyp.mechanism}\n")
        console.print(f"[bold]Falsify if:[/] {hyp.falsification_criteria}\n")
        if hyp.experiment:
            tag = "Computational \u2713" if hyp.experiment.computational else "Wet-lab"
            risk = getattr(hyp, "replication_risk", "medium")
            console.print(f"[dim]{tag} | Replication risk: {risk} | Effort: {hyp.experiment.estimated_effort}[/]\n")
        choice = click.prompt(
            "  Decision", default="s",
            type=click.Choice(["y", "n", "s", "q"], case_sensitive=False),
        )
        if choice == "q":
            break
        elif choice == "y":
            hyp.approved = True
            approved_n += 1
        elif choice == "n":
            hyp.approved = False
            rejected_n += 1
        else:
            skipped_n += 1
    hyp_path.write_text(report.model_dump_json(indent=2))
    console.print(
        f"\n[green]Saved.[/] Approved: {approved_n}  Rejected: {rejected_n}  Skipped: {skipped_n}"
    )


# ── cross-domain command ──────────────────────────────────────────────────────
@cli.command("cross-domain")
@click.option("--domain-a", "-a", required=True, help="First domain (Stage 1 must be complete)")
@click.option("--domain-b", "-b", required=True, help="Second domain (Stage 1 must be complete)")
@click.option("--top", "-n", default=10, show_default=True, help="Max bridges to analyse")
@click.option("--threshold", default=0.45, show_default=True, help="Min cross-domain cosine similarity")
def cross_domain_cmd(domain_a: str, domain_b: str, top: int, threshold: float) -> None:
    """Find structural gaps BETWEEN two domains \u2014 cross-pollination opportunities."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from athanor.domains import load_domain
    from athanor.graph.models import ConceptGraph
    from athanor.embed import Embedder
    from athanor.gaps import GapFinder
    from athanor.gaps.models import CandidateGap

    dom_a = load_domain(domain_a)
    dom_b = load_domain(domain_b)
    name_a, name_b = dom_a["name"], dom_b["name"]
    cross_name = f"{name_a}__{name_b}"

    graph_a = _out(name_a)["graphs"] / "concept_graph.json"
    graph_b = _out(name_b)["graphs"] / "concept_graph.json"
    for p, d in [(graph_a, name_a), (graph_b, name_b)]:
        if not p.exists():
            console.print(f"[red]No concept graph for '{d}' \u2014 run Stage 1 first.[/]")
            raise SystemExit(1)

    cg_a = ConceptGraph.model_validate_json(graph_a.read_text())
    cg_b = ConceptGraph.model_validate_json(graph_b.read_text())
    console.print(
        f"[green]Loaded:[/] {name_a} ({len(cg_a.concepts)} concepts)  +  "
        f"{name_b} ({len(cg_b.concepts)} concepts)"
    )

    # Embed all concepts (no centering — preserve inter-domain distances)
    embedder = Embedder()
    texts_a = [f"{c.label}. {c.description}" for c in cg_a.concepts]
    texts_b = [f"{c.label}. {c.description}" for c in cg_b.concepts]
    embs_a = embedder.embed(texts_a, center=False)
    embs_b = embedder.embed(texts_b, center=False)

    sim = cosine_similarity(embs_a, embs_b)  # shape (|A|, |B|)

    bridges: list[tuple[float, object, object]] = []
    for i, ca in enumerate(cg_a.concepts):
        for j, cb in enumerate(cg_b.concepts):
            s = float(sim[i, j])
            if s >= threshold:
                bridges.append((s, ca, cb))
    bridges.sort(key=lambda x: x[0], reverse=True)
    bridges = bridges[: top * 4]  # oversample
    console.print(f"Found {len(bridges)} cross-domain bridges (threshold={threshold})")

    if not bridges:
        console.print("[yellow]No bridges found. Try lowering --threshold.[/]")
        return

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
    combined_display = f"{dom_a['display']} \u2194 {dom_b['display']}"

    finder = GapFinder(
        domain=combined_display,
        model=dom_a.get("claude_model", cfg.model),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_gaps=top,
        max_workers=4,  # 1024 tokens/request × 4 = 4K OPM burst, safe for Haiku
    )
    gap_report = finder.analyse(candidates[:top], query=combined_context)

    out_path = _root / "outputs" / "gaps" / cross_name
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "gap_report.json"
    report_path.write_text(gap_report.model_dump_json(indent=2))
    console.print(
        f"[green]\u2713 Cross-domain analysis complete[/] \u2014 "
        f"{len(gap_report.analyses)} bridges analysed \u2192 {report_path}"
    )
    for a in gap_report.ranked[:5]:
        console.print(
            f"  \u2022 [cyan]{a.concept_a}[/] ({name_a}) \u2194 [cyan]{a.concept_b}[/] ({name_b}): "
            f"{a.research_question[:100]}\u2026"
        )


if __name__ == '__main__':
    cli()
