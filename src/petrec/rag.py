"""RAG chunking, multilingual embeddings, and pluggable vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into chunks no larger than 500 chars, preferring word boundaries."""
    if not 300 <= chunk_size <= 500:
        raise ValueError("chunk_size must be between 300 and 500")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be nonnegative and smaller than chunk_size")
    normalized = " ".join(str(text or "").split())
    chunks, start = [], 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


class Embedder:
    """Lazy multilingual-e5-small encoder with E5 query/passage prefixes."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", model: Any = None) -> None:
        self.model_name, self.model = model_name, model

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        prefix = "query: " if query else "passage: "
        vectors = self.model.encode([prefix + text for text in texts], convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != 384:
            raise ValueError("multilingual-e5-small must return 384-dimensional vectors")
        return result


class VectorStore(ABC):
    """Storage interface shared by in-memory and PostgreSQL vector indexes."""

    @abstractmethod
    def add(self, ids: Iterable[str], texts: Iterable[str], embeddings: np.ndarray,
            metadatas: Iterable[dict[str, Any]] | None = None) -> None: ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_m: int = 5) -> list[SearchResult]: ...


class NumpyVectorStore(VectorStore):
    """In-memory cosine index backed by NumPy."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.embeddings = np.empty((0, 384), dtype=np.float32)

    def add(self, ids: Iterable[str], texts: Iterable[str], embeddings: np.ndarray,
            metadatas: Iterable[dict[str, Any]] | None = None) -> None:
        ids, texts = list(map(str, ids)), list(texts)
        vectors = np.asarray(embeddings, dtype=np.float32)
        metadata = list(metadatas) if metadatas is not None else [{} for _ in ids]
        if vectors.shape != (len(ids), 384) or len(texts) != len(ids) or len(metadata) != len(ids):
            raise ValueError("ids, texts, metadata and vector(384) embeddings must match")
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadata)
        self.embeddings = np.vstack((self.embeddings, vectors))

    def search(self, query_embedding: np.ndarray, top_m: int = 5) -> list[SearchResult]:
        if top_m < 1:
            raise ValueError("top_m must be positive")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape != (384,):
            raise ValueError("query embedding must have 384 dimensions")
        if not self.ids:
            return []
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)
        scores = np.divide(self.embeddings @ query, norms, out=np.zeros(len(norms)), where=norms > 0)
        order = np.argsort(-scores, kind="stable")[:top_m]
        return [SearchResult(self.ids[i], self.texts[i], float(scores[i]), self.metadatas[i]) for i in order]


class PgVectorStore(VectorStore):
    """PostgreSQL pgvector implementation; takes a psycopg-compatible connection."""

    def __init__(self, connection: Any, table: str = "petrec_chunks") -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError("table must be a valid identifier")
        self.connection, self.table = connection, table
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{{}}', embedding vector(384) NOT NULL)")
        connection.commit()

    def add(self, ids: Iterable[str], texts: Iterable[str], embeddings: np.ndarray,
            metadatas: Iterable[dict[str, Any]] | None = None) -> None:
        import json
        ids, texts = list(map(str, ids)), list(texts)
        vectors = np.asarray(embeddings, dtype=np.float32)
        metadata = list(metadatas) if metadatas is not None else [{} for _ in ids]
        if vectors.shape != (len(ids), 384) or len(texts) != len(ids) or len(metadata) != len(ids):
            raise ValueError("ids, texts, metadata and vector(384) embeddings must match")
        sql = f"INSERT INTO {self.table} (id,text,metadata,embedding) VALUES (%s,%s,%s::jsonb,%s::vector) ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding"
        with self.connection.cursor() as cursor:
            for id_, text, meta, vector in zip(ids, texts, metadata, vectors):
                encoded = "[" + ",".join(map(str, vector.tolist())) + "]"
                cursor.execute(sql, (id_, text, json.dumps(meta, ensure_ascii=False), encoded))
        self.connection.commit()

    def search(self, query_embedding: np.ndarray, top_m: int = 5) -> list[SearchResult]:
        if top_m < 1:
            raise ValueError("top_m must be positive")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape != (384,):
            raise ValueError("query embedding must have 384 dimensions")
        encoded = "[" + ",".join(map(str, query.tolist())) + "]"
        sql = f"SELECT id,text,metadata,1-(embedding <=> %s::vector) AS score FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s"
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (encoded, encoded, top_m))
            rows = cursor.fetchall()
        return [SearchResult(str(row[0]), row[1], float(row[3]), row[2] or {}) for row in rows]


def search_pg_chunks(connection: Any, query_embedding: np.ndarray, top_m: int = 5,
                     product_id: str | None = None) -> list[SearchResult]:
    """Retrieve Petly ``kb_chunks`` through pgvector cosine distance.

    ``connection`` may be a psycopg connection or SQLAlchemy DBAPI connection. The
    product filter is optional; table/column names are fixed to the data-layer schema.
    """
    if top_m < 1:
        raise ValueError("top_m must be positive")
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if query.shape != (384,):
        raise ValueError("query embedding must have 384 dimensions")
    encoded = "[" + ",".join(map(str, query.tolist())) + "]"
    sql = ("SELECT id, product_id, text, 1-(embedding <=> %s::vector) AS score "
           "FROM kb_chunks WHERE (%s::text IS NULL OR product_id = %s::text) "
           "ORDER BY embedding <=> %s::vector LIMIT %s")
    with connection.cursor() as cursor:
        cursor.execute(sql, (encoded, product_id, product_id, encoded, top_m))
        rows = cursor.fetchall()
    return [SearchResult(str(row[0]), row[2], float(row[3]), {"product_id": row[1]}) for row in rows]


class RAGIndex:
    """Index product descriptions and ingredients as separately identified chunks."""

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None,
                 chunk_size: int = 400, overlap: int = 50) -> None:
        self.store = store or NumpyVectorStore()
        self.embedder = embedder or Embedder()
        self.chunk_size, self.overlap = chunk_size, overlap

    def add_documents(self, documents: Iterable[dict[str, Any]]) -> None:
        ids, texts, metadatas = [], [], []
        for document in documents:
            item_id = document.get("item_id", document.get("id", "doc"))
            for field in ("description", "ingredients_text"):
                for index, chunk in enumerate(chunk_text(document.get(field, ""), self.chunk_size, self.overlap)):
                    ids.append(f"{item_id}:{field}:{index}")
                    texts.append(chunk)
                    metadatas.append({"item_id": item_id, "field": field})
        if texts:
            self.store.add(ids, texts, self.embedder.encode(texts), metadatas)

    def search(self, query: str, top_m: int = 5) -> list[SearchResult]:
        return self.store.search(self.embedder.encode([query], query=True)[0], top_m=top_m)
