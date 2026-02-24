"""
athanor.reporting — render hypothesis reports to Markdown.

Extracted from cli.py so the same renderer is available in notebooks,
scripts, and tests — not only behind the Click ``report`` command.
"""
from __future__ import annotations

from collections.abc import Sequence

from athanor.hypotheses.models import Hypothesis


def render_hypothesis_markdown(
    candidates: Sequence[tuple[str, Hypothesis]],
    *,
    title: str,
    subtitle: str,
    date_str: str,
    total: int,
    multi_domain: bool = False,
    approved_only: bool = False,
) -> str:
    """Render a list of ``(domain_name, Hypothesis)`` pairs to Markdown.

    Args:
        candidates:    scored hypotheses to include (already sorted & sliced)
        title:         report title (e.g. "Information Theory Hypotheses")
        subtitle:      short subtitle line
        date_str:      ISO date string for the header
        total:         total hypothesis count before slicing
        multi_domain:  if True, add a "Domain" column to the summary table
        approved_only: annotate header when only approved hypotheses shown

    Returns:
        A single Markdown string ready for writing to disk or display.
    """
    lines: list[str] = [
        f"# Athanor — {title}",
        f"*{date_str} | {subtitle} | "
        f"{total} total | {len(candidates)} shown"
        + (" (approved only)" if approved_only else "") + "*",
        "", "---", "",
        "## Summary", "",
    ]

    # ── summary table ────────────────────────────────────────────────────
    header_cols = (
        ["#", "Domain", "Gap", "Score", "N", "R", "I", "Rep. Risk", "Compute"]
        if multi_domain else
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
        if multi_domain:
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

    # ── detailed sections ────────────────────────────────────────────────
    for i, (dom_name, h) in enumerate(candidates, 1):
        approved_tag = {True: " ✅ Approved", False: " ❌ Rejected", None: ""}.get(
            h.approved, ""
        )
        comp_label = (
            "Computational"
            if h.experiment and h.experiment.computational
            else "Requires wet-lab"
        )
        domain_tag = f" *(domain: {dom_name})*" if multi_domain else ""
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

    return "\n".join(lines)
