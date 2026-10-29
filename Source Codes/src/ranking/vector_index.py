"""
vector_index.py  —  High-Performance Vector Index for Candidate Retrieval
-------------------------------------------------------------------------
Implements Stage 1 of the two-stage recommender architecture:
  1. Stage 1 (Retrieval): Sub-millisecond ANN search using FAISS (or vectorized NumPy fallback)
     to retrieve top-M (~100) relevant article candidates.
  2. Stage 2 (Reranking): Responsible scoring, persuasion risk penalty, provider cap,
     and plain-English explanation generation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class VectorIndex:
    """
    Candidate retrieval index supporting FAISS Inner Product (Cosine) search
    with transparent NumPy fallback.
    """

    def __init__(self, emb_dim: int = 128, use_faiss: bool = True):
        self.emb_dim = emb_dim
        self.news_ids: list[str] = []
        self._id_to_pos: dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._has_faiss = False

        if use_faiss:
            try:
                import faiss
                self._faiss = faiss
                self._faiss_index = faiss.IndexFlatIP(emb_dim)
                self._has_faiss = True
                log.info("VectorIndex: FAISS IndexFlatIP initialized (dim=%d).", emb_dim)
            except Exception as exc:
                log.info("VectorIndex: FAISS not available (%s) — using fast NumPy fallback.", exc)

    def build(self, news_ids: list[str], embeddings: np.ndarray):
        """
        Build or replace the index with given embeddings.
        Embeddings are automatically L2-normalized so Inner Product == Cosine Similarity.
        """
        assert len(news_ids) == len(embeddings), "Count mismatch between IDs and embeddings"
        self.news_ids = list(news_ids)
        self._id_to_pos = {nid: i for i, nid in enumerate(self.news_ids)}

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normed_emb = (embeddings / norms).astype(np.float32)
        self.embeddings = normed_emb

        if self._has_faiss and self._faiss_index is not None:
            self._faiss_index.reset()
            self._faiss_index.add(normed_emb)

        log.info("VectorIndex built with %d items (dim=%d).", len(self.news_ids), self.emb_dim)

    def search(self, query_vec: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Retrieve top_k candidate articles by cosine similarity.

        Returns
        -------
        list of (news_id, similarity_score)
        """
        if self.embeddings is None or len(self.news_ids) == 0:
            return []

        # Normalize query
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        q_norm = q_norm.reshape(1, -1).astype(np.float32)

        k = min(top_k, len(self.news_ids))

        if self._has_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_norm, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.news_ids):
                    results.append((self.news_ids[idx], float(score)))
            return results

        # Vectorized NumPy fallback
        scores = np.dot(self.embeddings, q_norm[0])
        top_indices = np.argpartition(scores, -k)[-k:]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self.news_ids[i], float(scores[i])) for i in sorted_indices]
