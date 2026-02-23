"""Tests for athanor.ingest.parser — clean_text() and paper_to_text()."""
from __future__ import annotations

import pytest

from athanor.ingest.parser import clean_text, paper_to_text
from athanor.ingest.arxiv_client import Paper


# ── clean_text ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("a   b\n\nc") == "a b c"

    def test_strips_latex_commands_outside_math(self):
        result = clean_text(r"Important \textbf{bold} text")
        assert "textbf" not in result
        assert "Important" in result
        assert "text" in result

    def test_preserves_math_commands_inside_dollars(self):
        result = clean_text(r"The relation $\frac{a}{b} = c$ holds")
        assert r"\frac{a}{b}" in result

    def test_preserves_simple_math(self):
        result = clean_text(r"Entropy $H(X) = -\sum p \log p$ is key")
        assert "$" in result
        assert "H(X)" in result

    def test_strips_outside_preserves_inside(self):
        result = clean_text(r"\textbf{bold} and $\sqrt{x}$ math")
        assert "textbf" not in result
        assert r"\sqrt{x}" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_no_latex(self):
        assert clean_text("plain text") == "plain text"


# ── paper_to_text ────────────────────────────────────────────────────────────

def _paper(**kw):
    defaults = dict(
        arxiv_id="1", title="Title", abstract="Abstract body",
        authors=[], categories=[], published="2024-01-01", url="http://x",
    )
    defaults.update(kw)
    return Paper(**defaults)


class TestPaperToText:
    def test_abstract_only(self):
        p = _paper()
        result = paper_to_text(p)
        assert "Abstract:" in result
        assert "Title: Title" in result
        assert "Full text:" not in result

    def test_full_text_label(self):
        p = _paper(full_text="Long body text with many words " * 10)
        result = paper_to_text(p)
        assert "Full text:" in result
        assert "Abstract:" not in result
        assert "Title: Title" in result

    def test_title_always_present(self):
        p = _paper(title="My Paper", full_text="body")
        result = paper_to_text(p)
        assert "Title: My Paper" in result
