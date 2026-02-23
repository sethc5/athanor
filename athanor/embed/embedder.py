"""
athanor.embed.embedder — sentence embedding with local models.

Uses sentence-transformers so no external API calls are required for
embedding.  The model is downloaded once and cached by HuggingFace.

Disk cache: raw (uncentered) embeddings are stored under
  ~/.cache/athanor/embeds/<model_slug>/<sha256_of_text>.npy
so repeated runs skip re-encoding already-seen texts.  Centering is
applied to the assembled batch matrix on every call so it always
reflects the global mean of the current batch.

Domain-agnostic: swap embedding_model in .env to change behaviour.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

import numpy as np

from athanor.config import cfg

log = logging.getLogger(__name__)

_CACHE_ROOT = Path.home() / ".cache" / "athanor" / "embeds"


class Embedder:
    """Lazy-loading sentence embedder with a per-text disk cache.

    The model is only loaded on first call to .embed() so importing
    this module doesn't pay the model-load cost at import time.
    Raw (uncentered) embeddings are cached to disk; centering is
    re-applied over the live batch so the global mean stays correct.
    """

    def __init__(self, model_name: str = cfg.embedding_model) -> None:
        self._model_name = model_name
        self._model = None
        slug = model_name.replace("/", "__")
        self._cache_dir = _CACHE_ROOT / slug
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public ───────────────────────────────────────────────────────────────

    def embed(self, texts: List[str], center: bool = True) -> np.ndarray:
        """Return a (N, D) float32 array of embeddings for *texts*.

        Args:
            texts:  list of strings to embed
            center: if True (default), mean-center the embedding matrix to
                    correct for isotropy problems in sentence-transformer spaces.
                    Mean-centering removes the dominant global direction that
                    compresses all pairwise cosine similarities toward 1,
                    restoring discriminability in high-dimensional space.
        """
        keys = [self._cache_key(t) for t in texts]
        cached: dict[int, np.ndarray] = {}
        missing_idx: list[int] = []

        for i, key in enumerate(keys):
            vec = self._load_cached(key)
            if vec is not None:
                cached[i] = vec
            else:
                missing_idx.append(i)

        if missing_idx:
            log.info(
                "Embedding %d/%d texts (cache hit: %d) with %s",
                len(missing_idx), len(texts), len(cached), self._model_name,
            )
            model = self._load_model()
            new_texts = [texts[i] for i in missing_idx]
            new_vecs = model.encode(new_texts, show_progress_bar=False, batch_size=32)
            new_vecs = np.array(new_vecs, dtype=np.float32)
            for j, i in enumerate(missing_idx):
                self._save_cached(keys[i], new_vecs[j])
                cached[i] = new_vecs[j]
        else:
            log.info("Embedding: all %d texts served from cache", len(texts))

        vecs = np.stack([cached[i] for i in range(len(texts))], axis=0)
        if center and len(vecs) > 1:
            vecs = vecs - vecs.mean(axis=0, keepdims=True)
        return vecs

    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Return an (N, N) cosine similarity matrix (with isotropy correction)."""
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = self.embed(texts, center=True)  # mean-centered
        return cosine_similarity(vecs)

    # ── private ──────────────────────────────────────────────────────────────
    def _cache_key(self, text: str) -> str:
        """SHA-256 hex digest of (model_name + text) for a stable cache key."""
        payload = f"{self._model_name}\x00{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npy"

    def _load_cached(self, key: str) -> np.ndarray | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                return np.load(str(p))
            except Exception:  # noqa: BLE001
                p.unlink(missing_ok=True)
        return None

    def _save_cached(self, key: str, vec: np.ndarray) -> None:
        try:
            np.save(str(self._cache_path(key)), vec)
        except OSError:
            pass  # non-fatal — just skip caching
    def _load_model(self):
        if self._model is None:
            log.info("Loading sentence-transformer model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model
