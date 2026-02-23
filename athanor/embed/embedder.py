"""
athanor.embed.embedder — sentence embedding with local models.

Uses sentence-transformers so no external API calls are required for
embedding.  The model is downloaded once and cached by HuggingFace.

Domain-agnostic: swap embedding_model in .env to change behaviour.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from athanor.config import cfg

log = logging.getLogger(__name__)


class Embedder:
    """Lazy-loading sentence embedder.

    The model is only loaded on first call to .embed() so importing
    this module doesn't pay the model-load cost at import time.
    """

    def __init__(self, model_name: str = cfg.embedding_model) -> None:
        self._model_name = model_name
        self._model = None

    # ── public ───────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a (N, D) float32 array of embeddings for *texts*."""
        model = self._load_model()
        log.info("Embedding %d texts with %s", len(texts), self._model_name)
        vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
        return np.array(vectors, dtype=np.float32)

    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Return an (N, N) cosine similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = self.embed(texts)
        return cosine_similarity(vecs)

    # ── private ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            log.info("Loading sentence-transformer model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model
