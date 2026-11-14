"""Hybrid TF-IDF and Day 6 dense retrieval using reciprocal rank fusion."""
from typing import Any

from baseline import TfidfIndex
from dense_retriever import Day6DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        database_url: str | None = None,
        model_name: str | None = None,
        strategy: str = "B",
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.lexical = TfidfIndex(chunks)
        self.dense = Day6DenseRetriever(database_url, model_name, strategy)
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        lexical = self.lexical.search(query, top_k=max(top_k, 10))
        dense = self.dense.search(query, filters=filters, top_k=max(top_k, 10))
        by_id: dict[str, dict[str, Any]] = {}
        fused: dict[str, float] = {}
        for rank, chunk in enumerate(lexical, start=1):
            by_id[chunk["id"]] = dict(chunk)
            fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + 1 / (
                self.rrf_k + rank
            )
        for rank, chunk in enumerate(dense, start=1):
            by_id.setdefault(chunk["id"], dict(chunk))
            fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + 1 / (
                self.rrf_k + rank
            )
        results = []
        for chunk_id, score in fused.items():
            result = dict(by_id[chunk_id])
            result["score"] = score
            result["strategy"] = "hybrid-rrf"
            results.append(result)
        return sorted(results, key=lambda row: (-row["score"], row["id"]))[:top_k]
