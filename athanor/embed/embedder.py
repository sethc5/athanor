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
        model = self._load_model()
        log.info("Embedding %d texts with %s", len(texts), self._model_name)
        vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
        vecs = np.array(vectors, dtype=np.float32)
        if center and len(vecs) > 1:
            vecs = vecs - vecs.mean(axis=0, keepdims=True)
        return vecs

    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Return an (N, N) cosine similarity matrix (with isotropy correction)."""
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = self.embed(texts, center=True)  # mean-centered
        return cosine_similarity(vecs)

    # ── private ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            log.info("Loading sentence-transformer model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model
