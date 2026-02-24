"""
athanor CLI — thin Click wrapper over :mod:`athanor.pipeline`.

All heavy lifting lives in ``pipeline.py``; this module handles
Click decorators, Rich console output, and user-facing formatting.

Usage:
    athanor run --domain information_theory
    athanor run --domain longevity_biology --max-papers 20 --stages 1,2,3
    athanor run --domain information_theory --stages 1      # Stage 1 only
    athanor list-domains
    athanor status --domain longevity_biology
    athanor approve --domain longevity_biology              # review hypotheses
    athanor cross-domain --domain-a longevity_biology --domain-b information_theory
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from athanor import pipeline
from athanor.config import cfg
from athanor.domains import list_domains, load_domain
from athanor.hypotheses.models import HypothesisReport

console = Console()
log = logging.getLogger("athanor.cli")


# ── workspace helpers (delegate to pipeline) ─────────────────────────────────

def _ws_root() -> Path:
    """Return the active workspace root, printing Rich errors on failure."""
    try:
        return pipeline.workspace_root()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1)


def _out(domain: str) -> dict:
    """Return output path dict for a domain."""
    return pipeline.output_paths(domain)


# ── CLI root ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option(
    "--workspace", "-w",
    default=lambda: os.environ.get("ATHANOR_WORKSPACE", ""),
    metavar="NAME",
    help="Active workspace under workspaces/ (or set ATHANOR_WORKSPACE env var).",
)
def cli(verbose: bool, workspace: str) -> None:
    """Athanor — domain-agnostic automated science infrastructure."""
    if workspace:
        os.environ["ATHANOR_WORKSPACE"] = workspace
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s — %(message)s")


# ── list-domains ──────────────────────────────────────────────────────────────

@cli.command("list-domains")
def list_domains_cmd() -> None:
    """List all available domain configurations."""
    domains = list_domains()
    if not domains:
        console.print("[yellow]No domain configs found in domains/[/]")
        return
    console.print("[bold green]Available domains:[/]")
    for d in domains:
        console.print(f"  • {d}")


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", "-d", default=None, help="Domain name — omit to show all domains")
def status(domain: str) -> None:
    """Show pipeline output status for one or all domains."""

    def _stage_info(path):
        if not path.exists():
            return "[red]✗[/]", "missing"
        sz = path.stat().st_size
        try:
            data = json.loads(path.read_text())
        except Exception:
            return "[yellow]?[/]", f"{sz//1024}KB"
        count = ""
        if "concepts" in data:
            count = f"{len(data['concepts'])} concepts, {len(data.get('edges',[]))} edges"
        elif "analyses" in data:
            count = f"{len(data['analyses'])} gaps"
        elif "hypotheses" in data:
            count = f"{len(data['hypotheses'])} hypotheses"
        return "[green]✓[/]", count or f"{sz//1024}KB"

    domains_to_show = [domain] if domain else list_domains()

    if len(domains_to_show) == 1:
        d = domains_to_show[0]
        paths = _out(d)
        console.print(f"\n[bold]Pipeline status — {d}[/]")
        for label, key, fname in [
            ("Stage 1 — concept graph",   "graphs", "concept_graph.json"),
            ("Stage 2 — gap report",      "gaps",   "gap_report.json"),
            ("Stage 3 — hypothesis report","hyps",  "hypothesis_report.json"),
        ]:
            p = paths[key] / fname
            icon, info = _stage_info(p)
            console.print(f"  {icon} {label}: {info}")
    else:
        table = Table(title="Athanor Pipeline Status", show_lines=True)
        table.add_column("Domain", style="bold cyan", no_wrap=True)
        table.add_column("Stage 1\nConcept Graph", justify="center")
        table.add_column("Stage 2\nGap Report", justify="center")
        table.add_column("Stage 3\nHypotheses", justify="center")
        table.add_column("Top Score", justify="right")

        for d in sorted(domains_to_show):
            paths = _out(d)
            s1_icon, s1_info = _stage_info(paths["graphs"] / "concept_graph.json")
            s2_icon, s2_info = _stage_info(paths["gaps"] / "gap_report.json")
            s3_icon, s3_info = _stage_info(paths["hyps"] / "hypothesis_report.json")

            best_score = ""
            hyp_path = paths["hyps"] / "hypothesis_report.json"
            if hyp_path.exists():
                try:
                    hr = HypothesisReport.model_validate_json(hyp_path.read_text())
                    top = hr.top(1)
                    if top:
                        best_score = f"{top[0].final_score:.2f}"
                except Exception:
                    pass

            table.add_row(
                d, f"{s1_icon} {s1_info}", f"{s2_icon} {s2_info}",
                f"{s3_icon} {s3_info}", best_score,
            )
        console.print(table)


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
@click.option("--workers",     "-w", type=int, default=8, show_default=True,
              help="Parallel Claude workers for Stages 1+2")
@click.option("--critique",    is_flag=True, help="Run independent critic pass after Stage 3 (Stage 3.5)")
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
    critique: bool,
) -> None:
    """Run the full athanor pipeline for a domain."""
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

    hyp_path = out["hyps"] / "hypothesis_report.json"

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if 1 in stage_list:
        console.rule("[bold]Stage 1 — Literature Mapper[/]")
        concept_graph, candidate_gaps, n_papers = pipeline.run_stage_1(
            dom, out,
            no_cache=no_cache, pdf=pdf, s2=s2, cocite=cocite,
            workers=workers, hyp_path_feedback=hyp_path,
        )
        console.print(
            f"[green]✓ Stage 1 complete[/] — "
            f"{len(concept_graph.concepts)} concepts, {len(candidate_gaps)} gap candidates"
        )

    # ── Stage 2 ──────────────────────────────────────────────────────────
    if 2 in stage_list:
        console.rule("[bold]Stage 2 — Gap Finder[/]")
        gap_report = pipeline.run_stage_2(
            dom, out, workers=workers, hyp_path_feedback=hyp_path,
        )
        console.print(
            f"[green]✓ Stage 2 complete[/] — "
            f"{len(gap_report.analyses)} gaps analysed"
        )

    # ── Stage 3 ──────────────────────────────────────────────────────────
    if 3 in stage_list:
        console.rule("[bold]Stage 3 — Hypothesis Generator[/]")
        hyp_report = pipeline.run_stage_3(dom, out, workers=workers)
        console.print(
            f"[green]✓ Stage 3 complete[/] — "
            f"{len(hyp_report.hypotheses)} hypotheses"
        )

        # Stage 3.5 — optional critic pass
        if critique:
            console.rule("[bold]Stage 3.5 — Critic Pass[/]")
            hyp_report = pipeline.run_critic(dom, out, workers=max(1, workers // 2))
            _critiqued = sum(1 for h in hyp_report.hypotheses if h.critic_novelty is not None)
            console.print(f"[green]✓ Critic pass complete[/] — {_critiqued} hypotheses re-scored")

        # Quick summary
        top = hyp_report.top(3)
        if top:
            best = top[0]
            console.print(f"\n[bold green]Top hypothesis:[/] {best.statement}")
            console.print(f"[dim]Score: {best.composite_score:.2f} gen", end="")
            if best.critic_novelty is not None:
                console.print(f" → {best.final_score:.2f} blended[/]")
            else:
                console.print("[/]")
            if best.experiment:
                comp = "Computational ✓" if best.experiment.computational else "Wet lab required"
                console.print(f"[dim]{comp} | Effort: {best.experiment.estimated_effort}[/]")

    console.print("\n[bold green]Pipeline complete.[/]")


# ── critique command ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", "-d", required=True, help="Domain whose hypothesis_report.json to critique")
@click.option("--workers", "-w", type=int, default=4, show_default=True, help="Parallel critic workers")
def critique(domain: str, workers: int) -> None:
    """Run an independent critic pass (Stage 3.5) on an existing hypothesis report."""
    dom = load_domain(domain)
    out = _out(dom["name"])
    hyp_path = out["hyps"] / "hypothesis_report.json"

    if not hyp_path.exists():
        console.print(f"[red]No hypothesis report for '{domain}'. Run Stage 3 first.[/]")
        raise SystemExit(1)

    report = HypothesisReport.model_validate_json(hyp_path.read_text())
    n = len(report.hypotheses)
    if n == 0:
        console.print("[yellow]No hypotheses to critique.[/]")
        return

    console.rule(f"[bold]Critic Pass — {domain}[/]")
    console.print(f"Re-scoring {n} hypotheses independently (workers={workers})…")

    report_updated = pipeline.run_critic(dom, out, workers=workers)

    console.print(f"\n[green]✓ Critic pass complete[/] — {n} hypotheses updated → {hyp_path}\n")

    # Show delta table

    tbl = Table(title=f"Score delta — {domain}", show_header=True)
    tbl.add_column("Gap", style="dim", max_width=40)
    tbl.add_column("Gen", justify="center")
    tbl.add_column("Crit N/R/I", justify="center")
    tbl.add_column("Blended", justify="center", style="bold")
    tbl.add_column("Δ", justify="center")
    tbl.add_column("Critic note", max_width=40, style="dim")

    for h in sorted(report_updated.hypotheses, key=lambda x: x.final_score, reverse=True):
        gen_s = h.composite_score
        fin_s = h.final_score
        delta = fin_s - gen_s
        delta_str = (
            f"[red]{delta:+.2f}[/]" if delta < -0.2
            else (f"[green]{delta:+.2f}[/]" if delta > 0.2 else f"{delta:+.2f}")
        )
        crit_str = (
            f"{h.critic_novelty}/{h.critic_rigor}/{h.critic_impact}"
            if h.critic_novelty else "—"
        )
        tbl.add_row(
            f"{h.gap_concept_a} ⇔ {h.gap_concept_b}",
            f"{gen_s:.2f}", crit_str, f"{fin_s:.2f}", delta_str,
            h.critic_note[:60] if h.critic_note else "",
        )
    console.print(tbl)


# ── report command ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", "-d", default=None, help="Domain name (omit for all-domain digest)")
@click.option("--top", "-n", default=10, show_default=True, help="Max hypotheses to include")
@click.option("--approved-only", is_flag=True, help="Only include approved hypotheses")
@click.option("--out", "-o", default=None, help="Output .md file path (default: print to stdout)")
def report(domain: str, top: int, approved_only: bool, out: str) -> None:
    """Render a Markdown report of hypotheses for one or all domains."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if domain:
        domain_list = [domain]
    else:
        domain_list = [
            d for d in sorted(list_domains())
            if (_out(d)["hyps"] / "hypothesis_report.json").exists()
        ]
        if not domain_list:
            console.print("[red]No hypothesis reports found. Run Stage 3 first.[/]")
            raise SystemExit(1)

    all_hypotheses: list[tuple[str, object]] = []
    for d in domain_list:
        hyp_path = _out(d)["hyps"] / "hypothesis_report.json"
        if not hyp_path.exists():
            if len(domain_list) == 1:
                console.print(f"[red]No hypotheses for '{d}'. Run Stage 3 first.[/]")
                raise SystemExit(1)
            continue
        rep = HypothesisReport.model_validate_json(hyp_path.read_text())
        for h in rep.ranked:
            if approved_only and h.approved is not True:
                continue
            all_hypotheses.append((d, h))

    all_hypotheses.sort(key=lambda x: x[1].final_score, reverse=True)
    candidates = all_hypotheses[:top]

    multi = len(domain_list) > 1
    if multi:
        title = "Cross-Domain Research Digest"
        subtitle = f"{', '.join(domain_list)}"
    else:
        title_domain = domain_list[0].replace("_", " ").title()
        title = f"{title_domain} Hypotheses"
        subtitle = domain_list[0]

    lines = [
        f"# Athanor — {title}",
        f"*{date_str} | {subtitle} | "
        f"{len(all_hypotheses)} total | {len(candidates)} shown"
        + (" (approved only)" if approved_only else "") + "*",
        "", "---", "",
        "## Summary", "",
    ]

    header_cols = (
        ["#", "Domain", "Gap", "Score", "N", "R", "I", "Rep. Risk", "Compute"]
        if multi else
        ["#", "Gap", "Score", "N", "R", "I", "Rep. Risk", "Compute"]
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for i, (dom_name, h) in enumerate(candidates, 1):
        comp = "✓" if h.experiment and h.experiment.computational else "✗"
        label = {True: " ✅", False: " ❌", None: ""}.get(h.approved, "")
        risk_icon = {"low": "🟢", "high": "🔴"}.get(
            getattr(h, "replication_risk", "medium"), "🟡"
        )
        row_vals = [str(i)]
        if multi:
            row_vals.append(dom_name)
        row_vals += [
            f"{h.gap_concept_a} ↔ {h.gap_concept_b}{label}",
            f"{h.final_score:.1f}",
            str(h.novelty), str(h.rigor), str(h.impact),
            f"{risk_icon} {getattr(h, 'replication_risk', 'medium')}",
            comp,
        ]
        lines.append("| " + " | ".join(row_vals) + " |")

    lines += ["", "---", ""]

    for i, (dom_name, h) in enumerate(candidates, 1):
        approved_tag = {True: " ✅ Approved", False: " ❌ Rejected", None: ""}.get(h.approved, "")
        comp_label = "Computational" if h.experiment and h.experiment.computational else "Requires wet-lab"
        domain_tag = f" *(domain: {dom_name})*" if multi else ""
        lines += [
            f"## {i}. {h.gap_concept_a} ↔ {h.gap_concept_b}{approved_tag}{domain_tag}",
            f"**Score:** {h.final_score:.1f}  "
            f"(N\u202f{h.novelty} · R\u202f{h.rigor} · I\u202f{h.impact})  |  "
            f"{comp_label}  |  Replication risk: **{getattr(h, 'replication_risk', 'medium')}**",
            "",
            f"**Hypothesis:** {h.statement}", "",
            f"**Mechanism:** {h.mechanism}", "",
            f"**Prediction:** {h.prediction}", "",
            f"**Falsification criterion:** {h.falsification_criteria}", "",
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
        Path(out).write_text(md)
        console.print(f"[green]Report written → {out}[/]")
    else:
        console.print(md)


# ── search command ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("query", default="")
@click.option("--domain", "-d", default=None, help="Restrict to one domain")
@click.option("--min-score", default=0.0, show_default=True, help="Min composite score")
@click.option("--approved-only", is_flag=True, help="Only approved hypotheses")
@click.option("--risk", default=None, type=click.Choice(["low", "medium", "high"]),
              help="Filter by replication risk")
@click.option("--compute", is_flag=True, help="Only computational experiments")
def search(query: str, domain: str, min_score: float, approved_only: bool,
           risk: str, compute: bool) -> None:
    """Search hypotheses across all domains by keyword, score, or filter."""
    hyp_root = _ws_root() / "outputs" / "hypotheses"
    if not hyp_root.exists():
        console.print("[red]No hypothesis outputs found. Run Stage 3 for at least one domain.[/]")
        return

    domain_dirs = [hyp_root / domain] if domain else sorted(hyp_root.iterdir())

    results: list[tuple[str, object]] = []
    for d in domain_dirs:
        report_file = d / "hypothesis_report.json"
        if not report_file.exists():
            continue
        rep = HypothesisReport.model_validate_json(report_file.read_text())
        for h in rep.ranked:
            if h.final_score < min_score:
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
        console.print("[yellow]No hypotheses match the search.[/]")
        return

    console.print(f"[bold]{len(results)} result(s)[/]\n")
    for dom_name, h in results:
        risk_icon = {"low": "🟢", "high": "🔴"}.get(
            getattr(h, "replication_risk", "medium"), "🟡"
        )
        comp_icon = "✓" if h.experiment and h.experiment.computational else "✗"
        appr_icon = {True: " ✅", False: " ❌", None: ""}.get(h.approved, "")
        console.print(
            f"[dim]{dom_name}[/]  [cyan]{h.gap_concept_a} ↔ {h.gap_concept_b}[/]{appr_icon}"
            f"  score=[bold]{h.final_score:.1f}[/]  {risk_icon}  compute={comp_icon}"
        )
        console.print(
            f"  {h.statement[:120]}…" if len(h.statement) > 120 else f"  {h.statement}"
        )
        console.print()


# ── approve command ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", "-d", required=True, help="Domain name")
@click.option("--all", "show_all", is_flag=True, help="Re-review already-reviewed hypotheses")
def approve(domain: str, show_all: bool) -> None:
    """Interactively approve or reject hypotheses for a domain."""
    hyp_path = _out(domain)["hyps"] / "hypothesis_report.json"
    if not hyp_path.exists():
        console.print(f"[red]No hypothesis report found for '{domain}'. Run Stage 3 first.[/]")
        raise SystemExit(1)

    report = HypothesisReport.model_validate_json(hyp_path.read_text())
    candidates = list(report.hypotheses if show_all else report.pending_review)
    if not candidates:
        console.print("[green]All hypotheses already reviewed. Use --all to re-review.[/]")
        return

    candidates.sort(key=lambda h: h.final_score, reverse=True)
    console.print(
        f"[bold]Reviewing {len(candidates)} hypothesis(es) for [cyan]{domain}[/][/] "
        "([bold]y[/]=approve  [bold]n[/]=reject  [bold]s[/]=skip  [bold]q[/]=quit)\n"
    )

    approved_n = rejected_n = skipped_n = 0
    for i, hyp in enumerate(candidates, 1):
        console.rule(
            f"[bold]#{i}/{len(candidates)}  Score {hyp.final_score:.1f}  "
            f"N:{hyp.novelty} R:{hyp.rigor} I:{hyp.impact}[/]"
        )
        console.print(f"[cyan bold]{hyp.gap_concept_a} ↔ {hyp.gap_concept_b}[/]")
        console.print(f"\n[bold]Hypothesis:[/] {hyp.statement}\n")
        console.print(f"[bold]Mechanism:[/]  {hyp.mechanism}\n")
        console.print(f"[bold]Falsify if:[/] {hyp.falsification_criteria}\n")
        if hyp.experiment:
            tag = "Computational ✓" if hyp.experiment.computational else "Wet-lab"
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
@click.option("--domain-a", "-a", default=None, help="First domain (Stage 1 must be complete)")
@click.option("--domain-b", "-b", default=None, help="Second domain (Stage 1 must be complete)")
@click.option("--all", "all_pairs", is_flag=True, help="Run all domain-pair combinations automatically")
@click.option("--top", "-n", default=10, show_default=True, help="Max bridges to analyse per pair")
@click.option("--threshold", default=0.45, show_default=True, help="Min cross-domain cosine similarity")
def cross_domain_cmd(domain_a: str, domain_b: str, all_pairs: bool,
                     top: int, threshold: float) -> None:
    """Find structural gaps BETWEEN two domains — cross-pollination opportunities."""

    if all_pairs:
        available = [
            d for d in sorted(list_domains())
            if (_out(d)["graphs"] / "concept_graph.json").exists()
        ]
        if len(available) < 2:
            console.print("[red]Need at least 2 domains with concept graphs. Run Stage 1 first.[/]")
            raise SystemExit(1)
        pairs = [
            (available[i], available[j])
            for i in range(len(available))
            for j in range(i + 1, len(available))
        ]
        console.print(f"[bold]Running {len(pairs)} cross-domain pairs across {len(available)} domains[/]")
        for da, db in pairs:
            console.rule(f"[cyan]{da} ↔ {db}[/]")
            gap_report = pipeline.run_cross_domain(da, db, top=top, threshold=threshold)
            for a in gap_report.ranked[:5]:
                console.print(
                    f"  • [cyan]{a.concept_a}[/] ↔ [cyan]{a.concept_b}[/]: "
                    f"{a.research_question[:100]}…"
                )
        return

    if not domain_a or not domain_b:
        console.print("[red]Provide --domain-a and --domain-b, or use --all[/]")
        raise SystemExit(1)

    gap_report = pipeline.run_cross_domain(domain_a, domain_b, top=top, threshold=threshold)
    for a in gap_report.ranked[:5]:
        console.print(
            f"  • [cyan]{a.concept_a}[/] ↔ [cyan]{a.concept_b}[/]: "
            f"{a.research_question[:100]}…"
        )


if __name__ == '__main__':
    cli()
