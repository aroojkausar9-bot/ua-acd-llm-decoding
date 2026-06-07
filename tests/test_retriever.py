"""Minimal tests for HybridRetriever."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.models.retriever import HybridRetriever


def _make_retriever(passages, entity_map):
    embed_model = MagicMock()
    embed_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)

    n   = len(passages)
    dim = 384
    import faiss
    faiss_idx = faiss.IndexFlatIP(dim)
    vecs = np.random.randn(n, dim).astype(np.float32)
    faiss.normalize_L2(vecs)
    faiss_idx.add(vecs)

    from rank_bm25 import BM25Okapi
    tokenised = [p.lower().split() for p in passages]
    bm25_idx  = BM25Okapi(tokenised)

    return HybridRetriever(
        embed_model, faiss_idx, bm25_idx,
        passages, entity_map,
        bm25_weight=0.5, top_k=3,
    )


def test_retrieve_returns_list():
    passages   = ["Marie Curie won the Nobel Prize.", "She was born in Warsaw."]
    entity_map = ["marie curie", "marie curie"]
    r = _make_retriever(passages, entity_map)
    results = r.retrieve("Marie Curie Nobel Prize", entity="marie curie")
    assert isinstance(results, list)


def test_retrieve_entity_filter():
    passages   = ["Marie Curie fact.", "Alan Turing fact."]
    entity_map = ["marie curie", "alan turing"]
    r = _make_retriever(passages, entity_map)
    results = r.retrieve("Marie Curie", entity="marie curie")
    for passage, score in results:
        assert passage != "Alan Turing fact."


def test_retrieve_empty_corpus():
    embed_model = MagicMock()
    embed_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
    import faiss
    from rank_bm25 import BM25Okapi
    faiss_idx = faiss.IndexFlatIP(384)
    bm25_idx  = BM25Okapi([[]])
    r = HybridRetriever(embed_model, faiss_idx, bm25_idx, [], [], top_k=3)
    results = r.retrieve("any query")
    assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
