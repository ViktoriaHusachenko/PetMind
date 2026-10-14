"""One shared E5 encoder per process and lazy back-fill of product vectors in pgvector.

Loading `multilingual-e5-small` takes seconds; building it for every pipeline made the first
recommendation take 8–50 s (BUG-022). Products without a stored vector are embedded once and saved.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from petrec.profiles import ProductEmbedder
from petrec.rag import Embedder

logger = logging.getLogger("petmind.embeddings")


def product_document_frame(products: Iterable[Any]) -> pd.DataFrame:
    """Columns `ProductEmbedder.embed_products` reads, from ORM products."""
    return pd.DataFrame([{"parent_asin": p.id, "title": p.title, "categories": p.category,
                          "description": p.description, "ingredients_text": p.ingredients_text} for p in products])


class EmbeddingService:
    def __init__(self, product_embedder: ProductEmbedder | None = None) -> None:
        self.product_embedder = product_embedder or ProductEmbedder()
        self._lock = threading.Lock()

    def model(self) -> Any:
        with self._lock:  # SentenceTransformer loading is not re-entrant
            return self.product_embedder._get_model()

    def rag_embedder(self) -> Embedder:
        return Embedder(model=self.model())

    def warm_up(self) -> None:
        started = time.perf_counter()
        try:
            self.model()
            logger.info("step=embedding_model_load latency_ms=%.1f", (time.perf_counter() - started) * 1000)
        except Exception:
            logger.exception("Embedding model could not be loaded; recommendations will be slower")

    def ensure_product_vectors(self, session: Any, products: list[Any]) -> int:
        """Embed and persist vectors for products that have none; returns how many were added."""
        missing = [p for p in products if p.embedding is None]
        if not missing:
            return 0
        self.model()
        vectors = self.product_embedder.embed_products(product_document_frame(missing))
        for product in missing:
            vector = vectors.get(product.id)
            if vector is not None and np.isfinite(vector).all():
                product.embedding = np.asarray(vector, dtype=np.float32).tolist()
        session.commit()
        logger.info("Back-filled %d product embeddings", len(missing))
        return len(missing)
