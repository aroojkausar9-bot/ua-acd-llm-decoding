"""Hybrid BM25 + Dense retrieval with entity-aware filtering."""
from typing import List, Tuple, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def build_indices(
    all_passages: List[str],
    embed_model: SentenceTransformer,
) -> Tuple[faiss.Index, BM25Okapi]:
    """
    Build FAISS dense index and BM25 sparse index from a passage corpus.

    Args:
        all_passages: flat list of passage strings
        embed_model:  SentenceTransformer for dense encoding

    Returns:
        faiss_index: FAISS FlatIP index
        bm25_index:  BM25Okapi index
    """
    # Dense index
    embeddings = embed_model.encode(
        all_passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    )
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    # BM25 index (tokenise on whitespace for speed)
    tokenised = [p.lower().split() for p in all_passages]
    bm25_index = BM25Okapi(tokenised)

    return faiss_index, bm25_index


class HybridRetriever:
    """
    Hybrid BM25 + Dense retrieval with optional entity-aware filtering.

    Scores are combined as:
        hybrid = bm25_weight * bm25_normalised + dense_weight * dense_score
    """

    def __init__(
        self,
        embed_model: SentenceTransformer,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        all_passages: List[str],
        passage_to_entity: List[str],
        bm25_weight: float = 0.5,
        top_k: int = 5,
    ):
        self.embed_model       = embed_model
        self.faiss_index       = faiss_index
        self.bm25_index        = bm25_index
        self.all_passages      = all_passages
        self.passage_to_entity = passage_to_entity
        self.bm25_weight       = bm25_weight
        self.dense_weight      = 1.0 - bm25_weight
        self.top_k             = top_k

    def retrieve(
        self,
        query: str,
        entity: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k passages for a query.

        Args:
            query:  query string
            entity: optional entity name for entity-aware filtering (lowercased match)
            top_k:  override default top_k

        Returns:
            list of (passage, score) tuples sorted descending by score
        """
        k = top_k or self.top_k
        n = len(self.all_passages)

        # Dense scores
        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        raw_scores, indices = self.faiss_index.search(q_emb, min(n, k * 10))
        dense_scores = np.zeros(n)
        for idx, score in zip(indices[0], raw_scores[0]):
            if 0 <= idx < n:
                dense_scores[idx] = float(score)

        # BM25 scores (normalised)
        bm25_raw = np.array(self.bm25_index.get_scores(query.lower().split()))
        bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
        bm25_scores = bm25_raw / bm25_max

        # Hybrid combination
        hybrid = self.bm25_weight * bm25_scores + self.dense_weight * dense_scores

        # Entity filter: zero out passages from other entities
        if entity:
            ent_lower = entity.lower()
            for i, ent in enumerate(self.passage_to_entity):
                if ent != ent_lower:
                    hybrid[i] = 0.0

        top_indices = np.argsort(hybrid)[::-1][:k]
        return [
            (self.all_passages[i], float(hybrid[i]))
            for i in top_indices
            if hybrid[i] > 0
        ]
