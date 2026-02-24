"""
athanor.config — centralised configuration via environment variables.

All external knobs live here. Import ``cfg`` everywhere else.
``project_root`` is the single source of truth for the repository root path.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env from the project root (two levels up from this file)
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env", override=False)


class Config(BaseSettings):
    """Application settings populated from environment variables.

    pydantic-settings reads env vars automatically:  field ``foo_bar`` maps to
    env var ``FOO_BAR`` (uppercased).  The ``model`` field uses a
    ``validation_alias`` because the env var is ``ANTHROPIC_MODEL``, not
    ``MODEL``.
    """

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    model: str = Field(default="claude-opus-4-5", validation_alias="ANTHROPIC_MODEL")

    # ── arXiv ────────────────────────────────────────────────────────────────
    arxiv_max_results: int = 20
    arxiv_sort_by: str = "relevance"

    # ── Semantic Scholar ─────────────────────────────────────────────────────
    s2_api_key: str = ""

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Paths (derived from project_root, not env vars) ──────────────────────
    @property
    def project_root(self) -> Path:               # noqa: D401
        return project_root

    @property
    def data_raw(self) -> Path:                    # noqa: D401
        return project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:              # noqa: D401
        return project_root / "data" / "processed"


cfg = Config()
# Create cache directories eagerly
for _p in (cfg.data_raw, cfg.data_processed):
    _p.mkdir(parents=True, exist_ok=True)


def workspace_root() -> Path:
    """Return the active workspace root (workspaces/<name>/ or repo root).

    Reads ``ATHANOR_WORKSPACE`` env var.  Raises ``FileNotFoundError`` if the
    workspace directory doesn't exist.
    """
    ws = os.environ.get("ATHANOR_WORKSPACE", "").strip()
    if ws:
        p = project_root / "workspaces" / ws
        if not p.exists():
            raise FileNotFoundError(f"Workspace '{ws}' not found at {p}")
        return p
    return project_root
