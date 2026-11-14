"""Dense retrieval adapter for the Day 6 PostgreSQL/pgvector index."""
import math
import os
from typing import Any

import psycopg
from sentence_transformers import SentenceTransformer
from psycopg.types.json import Jsonb


class Day6DenseRetriever:
    """Query one of the persisted Day 6 chunking strategies."""

    def __init__(
        self,
        database_url: str | None = None,
        model_name: str | None = None,
        strategy: str = "B",
    ) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab",
        )
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.strategy = strategy
        if strategy not in {"A", "B", "C"}:
            raise ValueError("strategy must be A, B, or C")
        self.model = SentenceTransformer(self.model_name)
        self.dimensions = self.model.get_embedding_dimension()
        if self.dimensions != 384:
            raise ValueError(
                f"Expected Day 6 embedding dimension 384, got {self.dimensions}"
            )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        if len(vector) != self.dimensions or not all(
            math.isfinite(float(value)) for value in vector
        ):
            raise ValueError("query embedding is invalid")

        where = ["strategy = %s"]
        parameters: list[Any] = [_vector_literal(vector), self.strategy]
        if filters:
            where.append("metadata @> %s::jsonb")
            parameters.append(Jsonb(filters))
        parameters.extend([_vector_literal(vector), top_k])
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, document_id, source, page, text, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM ingestion_chunks
                    WHERE {' AND '.join(where)}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    parameters,
                )
                return [
                    {
                        "id": row[0],
                        "text": row[4],
                        "source_id": row[1],
                        "source": row[2],
                        "page": row[3],
                        "metadata": row[5] or {},
                        "score": float(row[6]),
                        "strategy": self.strategy,
                    }
                    for row in cursor.fetchall()
                ]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"
