"""Shared pytest fixtures for the Athanor test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ATHANOR_WORKSPACE to a temp dir so tests never touch real data.

    Also removes ANTHROPIC_API_KEY to prevent accidental API calls.
    """
    monkeypatch.setenv("ATHANOR_WORKSPACE", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Patch the single source of truth for project root
    import athanor.config as _cfg
    monkeypatch.setattr(_cfg, "project_root", tmp_path)
    # Also patch the re-exported reference in pipeline
    import athanor.pipeline as _pl
    monkeypatch.setattr(_pl, "_root", tmp_path)
