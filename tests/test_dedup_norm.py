"""Tests for _norm() canonicalization and deduplicate_gaps() clustering."""
from __future__ import annotations

import numpy as np
import pytest

from athanor.gaps.models import CandidateGap
from athanor.gaps.dedup import deduplicate_gaps
from athanor.graph.builder import normalize_label


# ── normalize_label() tests ──────────────────────────────────────────────────

class TestNormalizeLabel:
    def test_lowercase(self):
        assert normalize_label("CalabiYau") == "calabiyau"

    def test_hyphen_collapse(self):
        assert normalize_label("Calabi-Yau") == "calabiyau"

    def test_underscore_collapse(self):
        assert normalize_label("Calabi_Yau") == "calabiyau"

    def test_mixed_separators(self):
        assert normalize_label("Calabi-Yau three_fold") == "calabiyauthreefold"

    def test_empty_string(self):
        assert normalize_label("") == ""

    def test_already_normalized(self):
        assert normalize_label("calabiyau") == "calabiyau"

    def test_multiple_spaces(self):
        assert normalize_label("a   b   c") == normalize_label("abc")


# ── deduplicate_gaps() tests ─────────────────────────────────────────────────

class _FakeEmbedder:
    """Return pre-set vectors so tests don't depend on real model / isotropy."""

    def __init__(self, vectors):
        self._vectors = vectors

    def embed(self, texts):
        return np.array(self._vectors[: len(texts)], dtype=np.float32)


class TestDeduplicateGaps:
    """Test gap deduplication with mock embeddings."""

    @staticmethod
    def _make_gap(a: str, b: str, sim: float = 0.5) -> CandidateGap:
        return CandidateGap(
            concept_a=a,
            concept_b=b,
            similarity=sim,
            graph_distance=5,
            description_a=f"Description of {a}",
            description_b=f"Description of {b}",
        )

    def test_single_gap_passthrough(self):
        gap = self._make_gap("A", "B")
        result, clusters = deduplicate_gaps([gap])
        assert len(result) == 1
        assert result[0] is gap

    def test_empty_list(self):
        result, clusters = deduplicate_gaps([])
        assert isinstance(result, list)

    def test_identical_gaps_merged(self):
        """Two gaps with identical embeddings should be merged."""
        g1 = self._make_gap("Neural Networks", "Hodge Numbers")
        g2 = self._make_gap("Neural Networks", "Hodge Numbers")
        # Identical vectors → cosine_sim == 1.0 → merged at any threshold
        emb = _FakeEmbedder([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result, clusters = deduplicate_gaps([g1, g2], threshold=0.85, embedder=emb)
        assert len(result) == 1

    def test_different_gaps_preserved(self):
        """Orthogonal embeddings should survive deduplication."""
        g1 = self._make_gap("Neural Networks", "Hodge Numbers")
        g2 = self._make_gap("Dark Energy", "Cosmic Inflation")
        emb = _FakeEmbedder([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result, clusters = deduplicate_gaps([g1, g2], threshold=0.85, embedder=emb)
        assert len(result) == 2

    def test_threshold_sensitivity(self):
        """Lower threshold should merge more aggressively."""
        g1 = self._make_gap("Machine Learning", "Deep Learning")
        g2 = self._make_gap("ML Methods", "Neural Networks")
        # Vectors with ~0.5 cosine similarity
        emb_strict = _FakeEmbedder([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        emb_loose = _FakeEmbedder([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        result_strict, _ = deduplicate_gaps([g1, g2], threshold=0.99, embedder=emb_strict)
        result_loose, _ = deduplicate_gaps([g1, g2], threshold=0.3, embedder=emb_loose)
        assert len(result_strict) >= len(result_loose)

    def test_cluster_map_validity(self):
        """Cluster map should have correct structure."""
        gaps = [
            self._make_gap("A", "B"),
            self._make_gap("C", "D"),
            self._make_gap("E", "F"),
        ]
        # All orthogonal → 3 clusters
        emb = _FakeEmbedder([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result, clusters = deduplicate_gaps(gaps, threshold=0.85, embedder=emb)
        for rep_idx in clusters:
            assert 0 <= rep_idx < len(gaps)
        for members in clusters.values():
            for m in members:
                assert 0 <= m < len(gaps)
