"""
athanor.domains — domain configuration loader.

Domain configs live in:
  workspaces/<name>/domains/*.yaml  (when ATHANOR_WORKSPACE is set)
  domains/*.yaml                    (fallback / default workspace)

Load one with: athanor.domains.load_domain("information_theory")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _domains_dir() -> Path:
    """Return the domains directory for the active workspace."""
    ws = os.environ.get("ATHANOR_WORKSPACE", "").strip()
    if ws:
        ws_path = _REPO_ROOT / "workspaces" / ws
        ws_domains = ws_path / "domains"
        if ws_domains.exists():
            return ws_domains
        # Workspace exists but no domains/ subdir yet
        raise FileNotFoundError(
            f"Workspace '{ws}' has no domains/ folder: {ws_domains}\n"
            f"Create it or check ATHANOR_WORKSPACE."
        )
    return _REPO_ROOT / "domains"


def load_domain(name_or_path: str) -> Dict[str, Any]:
    """Load a domain config by name (e.g. 'information_theory') or file path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = _domains_dir() / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {path}\n"
            f"Available: {list_domains()}"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("embedding_model", "all-MiniLM-L6-v2")
    cfg.setdefault("claude_model", "claude-opus-4-5")
    cfg.setdefault("max_papers", 15)
    cfg.setdefault("max_gaps", 15)
    cfg.setdefault("sources", ["arxiv"])
    return cfg


def list_domains() -> list[str]:
    """Return names of all available domain configs in the active workspace."""
    return [p.stem for p in sorted(_domains_dir().glob("*.yaml"))]
