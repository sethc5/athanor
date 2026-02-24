"""
athanor.config — centralised configuration via environment variables.

All external knobs live here. Import `cfg` everywhere else.
`project_root` is the single source of truth for the repository root path.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env", override=False)


class Config:
    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    model: str = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5")

    # ── arXiv ────────────────────────────────────────────────────────────────
    arxiv_max_results: int = int(os.environ.get("ARXIV_MAX_RESULTS", "20"))
    arxiv_sort_by: str = os.environ.get("ARXIV_SORT_BY", "relevance")

    # ── Semantic Scholar ─────────────────────────────────────────────────────
    s2_api_key: str = os.environ.get("S2_API_KEY", "")  # optional; raises rate limit

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    project_root: Path = project_root
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


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
