"""CLI smoke tests + pipeline module tests via Click CliRunner."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from athanor.cli import cli
from athanor import pipeline


# ────────────────────────── workspace fixture ────────────────────────────────

@pytest.fixture
def workspace_with_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a fake workspace with domain config + minimal stage outputs.

    Patches _root in pipeline, cli, and domains modules.
    """
    ws_name = "test_ws"
    ws = tmp_path / "workspaces" / ws_name
    dom_dir = ws / "domains"
    dom_dir.mkdir(parents=True)
    # Write a minimal domain YAML
    (dom_dir / "test_domain.yaml").write_text(
        "name: test_domain\n"
        "display: Test Domain\n"
        "max_papers: 5\n"
        "max_gaps: 5\n"
        "queries:\n  - 'test query'\n"
    )

    # Stage 1 output
    graphs_dir = ws / "outputs" / "graphs" / "test_domain"
    graphs_dir.mkdir(parents=True)
    (graphs_dir / "concept_graph.json").write_text(json.dumps({
        "domain": "test_domain",
        "query": "test query",
        "concepts": [
            {"label": "Concept A", "description": "Desc A"},
            {"label": "Concept B", "description": "Desc B"},
        ],
        "edges": [
            {"source": "Concept A", "target": "Concept B", "relation": "relates to"},
        ],
    }))
    (graphs_dir / "candidate_gaps.json").write_text(json.dumps([
        {"concept_a": "Concept A", "concept_b": "Concept B",
         "similarity": 0.7, "graph_distance": 5, "structural_hole_score": 0.3},
    ]))

    # Stage 2 output
    gaps_dir = ws / "outputs" / "gaps" / "test_domain"
    gaps_dir.mkdir(parents=True)
    (gaps_dir / "gap_report.json").write_text(json.dumps({
        "domain": "test_domain",
        "query": "test query",
        "n_candidates": 1,
        "n_analyzed": 1,
        "analyses": [{
            "concept_a": "Concept A", "concept_b": "Concept B",
            "research_question": "How A relates to B?",
            "why_unexplored": "Not studied",
            "intersection_opportunity": "Great",
            "methodology": "Simulation",
            "computational": True,
            "novelty": 4, "tractability": 3, "impact": 5,
        }],
    }))

    # Stage 3 output
    hyps_dir = ws / "outputs" / "hypotheses" / "test_domain"
    hyps_dir.mkdir(parents=True)
    (hyps_dir / "hypothesis_report.json").write_text(json.dumps({
        "domain": "test_domain",
        "query": "test query",
        "n_gaps_considered": 1,
        "hypotheses": [{
            "gap_concept_a": "Concept A",
            "gap_concept_b": "Concept B",
            "source_question": "How A relates to B?",
            "statement": "We hypothesize A causes B",
            "mechanism": "Through Z",
            "prediction": "B increases by 50%",
            "falsification_criteria": "If B doesn't increase",
            "novelty": 4, "rigor": 3, "impact": 5,
            "keywords": ["A", "B"],
            "experiment": {
                "approach": "Run simulation", "steps": ["Step 1"],
                "tools": ["Python"], "computational": True,
                "estimated_effort": "1 week", "data_requirements": "None",
                "expected_positive": "B increases",
                "expected_negative": "B stays same",
                "null_hypothesis": "No effect",
            },
        }],
    }))

    # Patch the single source of truth and all re-exported references
    import athanor.config as _cfg_mod
    monkeypatch.setattr(_cfg_mod, "project_root", tmp_path)
    monkeypatch.setattr(pipeline, "_root", tmp_path)
    import athanor.domains as _dom_mod
    monkeypatch.setattr(_dom_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("ATHANOR_WORKSPACE", ws_name)

    return ws


# ────────────────────────── CLI help tests ───────────────────────────────────

class TestHelp:
    """Every command should have valid --help output."""

    @pytest.fixture(autouse=True)
    def runner(self):
        self.runner = CliRunner()

    @pytest.mark.parametrize("cmd", [
        [], ["run", "--help"], ["status", "--help"],
        ["approve", "--help"], ["critique", "--help"],
        ["cross-domain", "--help"], ["report", "--help"],
        ["search", "--help"], ["list-domains", "--help"],
    ])
    def test_help(self, cmd):
        args = cmd if cmd else ["--help"]
        result = self.runner.invoke(cli, args)
        assert result.exit_code == 0
        assert "Usage" in result.output or "Options" in result.output


# ────────────────────────── list-domains ─────────────────────────────────────

class TestListDomains:
    def test_lists_domains(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["list-domains"])
        assert result.exit_code == 0
        assert "test_domain" in result.output


# ────────────────────────── status ───────────────────────────────────────────

class TestStatus:
    def test_all_domains(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "test_domain" in result.output

    def test_single_domain(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--domain", "test_domain"])
        assert result.exit_code == 0
        assert "Stage 1" in result.output


# ────────────────────────── search ───────────────────────────────────────────

class TestSearch:
    def test_keyword_search(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "hypothesize"])
        assert result.exit_code == 0
        assert "1 result" in result.output

    def test_no_results(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "nonexistent_xyz_12345"])
        assert result.exit_code == 0
        assert "No hypotheses" in result.output


# ────────────────────────── report ───────────────────────────────────────────

class TestReport:
    def test_single_domain_report(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--domain", "test_domain"])
        assert result.exit_code == 0
        assert "Concept A" in result.output

    def test_report_to_file(self, workspace_with_data, tmp_path):
        runner = CliRunner()
        out_file = str(tmp_path / "report.md")
        result = runner.invoke(cli, ["report", "--domain", "test_domain", "--out", out_file])
        assert result.exit_code == 0
        assert Path(out_file).exists()
        content = Path(out_file).read_text()
        assert "Concept A" in content


# ────────────────────────── run guards ───────────────────────────────────────

class TestRunGuards:
    def test_run_requires_api_key(self, workspace_with_data):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--domain", "test_domain"])
        assert result.exit_code != 0
        assert "ANTHROPIC_API_KEY" in result.output


# ────────────────────────── pipeline module ──────────────────────────────────

class TestPipelineModule:
    def test_workspace_root(self, workspace_with_data):
        root = pipeline.workspace_root()
        assert root.exists()
        assert "test_ws" in str(root)

    def test_output_paths(self, workspace_with_data):
        out = pipeline.output_paths("test_domain")
        assert "graphs" in out
        assert "gaps" in out
        assert "hyps" in out
        assert "test_domain" in str(out["graphs"])

    def test_workspace_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "_root", tmp_path)
        monkeypatch.setenv("ATHANOR_WORKSPACE", "nonexistent_ws_xyz")
        with pytest.raises(FileNotFoundError):
            pipeline.workspace_root()
