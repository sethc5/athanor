"""Tests for compute_candidate_gaps, Paper.from_dict, pipeline helpers, and builder utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from athanor.ingest.arxiv_client import Paper
from athanor.graph.models import Concept, ConceptGraph
from athanor.graph.builder import GraphBuilder, normalize_label
from athanor import pipeline


# ── Paper.from_dict backward compat ──────────────────────────────────────────

class TestPaperFromDict:
    def test_basic_roundtrip(self):
        p = Paper(
            arxiv_id="2401.00001", title="T", abstract="A",
            authors=["Auth"], categories=["cs.AI"],
            published="2024-01-01", url="http://x",
        )
        d = p.to_dict()
        p2 = Paper.from_dict(d)
        assert p2.arxiv_id == p.arxiv_id
        assert p2.title == p.title

    def test_url_in_full_text_migrated(self):
        """Old caches stored PDF URL in full_text — should migrate to pdf_url."""
        d = {
            "arxiv_id": "1234", "title": "T", "abstract": "A",
            "authors": [], "categories": [], "published": "2024",
            "url": "http://x", "full_text": "http://example.com/paper.pdf",
        }
        p = Paper.from_dict(d)
        assert p.pdf_url == "http://example.com/paper.pdf"
        assert p.full_text is None

    def test_normal_full_text_preserved(self):
        d = {
            "arxiv_id": "1234", "title": "T", "abstract": "A",
            "authors": [], "categories": [], "published": "2024",
            "url": "http://x", "full_text": "This is actual text content",
        }
        p = Paper.from_dict(d)
        assert p.full_text == "This is actual text content"
        assert p.pdf_url is None

    def test_unknown_keys_ignored(self):
        d = {
            "arxiv_id": "1234", "title": "T", "abstract": "A",
            "authors": [], "categories": [], "published": "2024",
            "url": "http://x", "future_field": "should not crash",
        }
        p = Paper.from_dict(d)
        assert p.arxiv_id == "1234"


# ── normalize_label ──────────────────────────────────────────────────────────

class TestNormalizeLabelImported:
    """Verify the real normalize_label from builder matches expectations."""

    def test_hyphen_and_space(self):
        assert normalize_label("Calabi-Yau three-fold") == "calabiyauthreefold"

    def test_underscore(self):
        assert normalize_label("some_concept_name") == "someconceptname"

    def test_idempotent(self):
        assert normalize_label("abc") == "abc"


# ── _get_api_key ─────────────────────────────────────────────────────────────

class TestGetApiKey:
    def test_returns_key_when_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        assert pipeline._get_api_key() == "sk-test-123"

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            pipeline._get_api_key()


# ── build_from_raw robustness ────────────────────────────────────────────────

class TestBuildFromRaw:
    def test_unknown_edge_fields_ignored(self):
        raw = {
            "concepts": [
                {"label": "A", "description": "desc A"},
                {"label": "B", "description": "desc B"},
            ],
            "edges": [
                {"source": "A", "target": "B", "extra_unknown": "ignored"},
            ],
        }
        g = GraphBuilder().build_from_raw(raw)
        assert len(g.edges) == 1
        assert g.edges[0].relation == "related_to"

    def test_explicit_relation_preserved(self):
        raw = {
            "concepts": [
                {"label": "A", "description": "a"},
                {"label": "B", "description": "b"},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "causes"},
            ],
        }
        g = GraphBuilder().build_from_raw(raw)
        assert g.edges[0].relation == "causes"


# ── dedup empty list ─────────────────────────────────────────────────────────

class TestDedupEdgeCases:
    def test_empty_list_cluster_map(self):
        from athanor.gaps.dedup import deduplicate_gaps
        result, clusters = deduplicate_gaps([])
        assert result == []
        assert clusters == {}


# ── GapReport.top() parameterised ────────────────────────────────────────────

class TestGapReportTop:
    def test_top_is_callable_with_n(self):
        from athanor.gaps.models import GapReport, GapAnalysis
        analyses = [
            GapAnalysis(
                concept_a="A", concept_b="B",
                research_question="Q?", why_unexplored="W",
                intersection_opportunity="O", methodology="M",
                computational=True, novelty=i, tractability=3, impact=3,
            )
            for i in range(1, 6)
        ]
        gr = GapReport(domain="t", query="q", analyses=analyses)
        assert len(gr.top(2)) == 2
        assert len(gr.top(10)) == 5  # capped at available
        assert gr.top(1)[0].composite_score >= gr.top(2)[1].composite_score


# ── _EDGE_FIELDS module-level constant ───────────────────────────────────────

class TestEdgeFieldsConstant:
    def test_edge_fields_is_frozenset(self):
        from athanor.graph.builder import _EDGE_FIELDS
        assert isinstance(_EDGE_FIELDS, frozenset)
        assert "source" in _EDGE_FIELDS
        assert "target" in _EDGE_FIELDS
        assert "relation" in _EDGE_FIELDS


# ── edge_type preserved through merge ────────────────────────────────────────

class TestEdgeTypePreserved:
    def test_build_from_raw_preserves_edge_type(self):
        raw = {
            "concepts": [
                {"label": "A", "description": "a"},
                {"label": "B", "description": "b"},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "causes",
                 "edge_type": "causal"},
            ],
        }
        g = GraphBuilder().build_from_raw(raw)
        assert g.edges[0].edge_type == "causal"


# ── _extend_unique helper ────────────────────────────────────────────────────

class TestExtendUnique:
    def test_no_duplicates(self):
        from athanor.pipeline import _extend_unique
        p1 = Paper(arxiv_id="1", title="Alpha", abstract="A",
                   authors=[], categories=[], published="2024", url="http://a")
        p2 = Paper(arxiv_id="2", title="Beta", abstract="B",
                   authors=[], categories=[], published="2024", url="http://b")
        target: list = [p1]
        _extend_unique(target, [p2])
        assert len(target) == 2

    def test_deduplicates_by_title(self):
        from athanor.pipeline import _extend_unique
        p1 = Paper(arxiv_id="1", title="Same Title", abstract="A",
                   authors=[], categories=[], published="2024", url="http://a")
        p2 = Paper(arxiv_id="2", title="same title", abstract="B",
                   authors=[], categories=[], published="2024", url="http://b")
        target: list = [p1]
        _extend_unique(target, [p2])
        assert len(target) == 1

    def test_empty_new_papers(self):
        from athanor.pipeline import _extend_unique
        p1 = Paper(arxiv_id="1", title="T", abstract="A",
                   authors=[], categories=[], published="2024", url="http://a")
        target: list = [p1]
        _extend_unique(target, [])
        assert len(target) == 1


# ── workspace_root in config ─────────────────────────────────────────────────

class TestWorkspaceRootConfig:
    def test_no_workspace_returns_project_root(self, tmp_path, monkeypatch):
        import athanor.config as _cfg
        monkeypatch.setattr(_cfg, "project_root", tmp_path)
        monkeypatch.setenv("ATHANOR_WORKSPACE", "")
        from athanor.config import workspace_root
        assert workspace_root() == tmp_path

    def test_workspace_returns_subdir(self, tmp_path, monkeypatch):
        import athanor.config as _cfg
        ws = tmp_path / "workspaces" / "test_ws"
        ws.mkdir(parents=True)
        monkeypatch.setattr(_cfg, "project_root", tmp_path)
        monkeypatch.setenv("ATHANOR_WORKSPACE", "test_ws")
        from athanor.config import workspace_root
        assert workspace_root() == ws

    def test_missing_workspace_raises(self, tmp_path, monkeypatch):
        import athanor.config as _cfg
        monkeypatch.setattr(_cfg, "project_root", tmp_path)
        monkeypatch.setenv("ATHANOR_WORKSPACE", "nonexistent")
        from athanor.config import workspace_root
        with pytest.raises(FileNotFoundError):
            workspace_root()
