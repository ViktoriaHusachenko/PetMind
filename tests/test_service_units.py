"""Unit tests for service adapters that need no database: RAG metadata and embedding back-fill."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from petrec.profiles import ProductEmbedder
from petrec.rag import SearchResult

from app.embeddings import EmbeddingService
from app.service import _with_item_id


def test_rag_results_gain_item_id_for_the_core_pipeline():
    result = _with_item_id(SearchResult("c1", "текст", 0.9, {"product_id": "P1"}))
    assert result.metadata == {"product_id": "P1", "item_id": "P1"}
    untouched = _with_item_id(SearchResult("c2", "текст", 0.5, {"item_id": "P2"}))
    assert untouched.metadata == {"item_id": "P2"}


class FakeModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.ones((len(texts), 384), dtype=np.float32)


class FakeSession:
    commits = 0

    def commit(self):
        self.commits += 1


def test_embedding_backfill_only_embeds_missing_products_once():
    model = FakeModel()
    service = EmbeddingService(ProductEmbedder(model=model))
    products = [SimpleNamespace(id="A", title="Корм", category=None, description=None, ingredients_text="salmon",
                                embedding=None),
                SimpleNamespace(id="B", title="Інший", category=None, description=None, ingredients_text=None,
                                embedding=[0.0] * 384)]
    session = FakeSession()
    assert service.ensure_product_vectors(session, products) == 1
    assert len(products[0].embedding) == 384 and model.calls == 1 and session.commits == 1
    assert service.ensure_product_vectors(session, products) == 0 and model.calls == 1
    assert service.rag_embedder().model is model


def test_same_day_purchases_merge_into_one_restock():
    from datetime import date, datetime, timezone
    from app.service import _purchase_days

    rows = [SimpleNamespace(ts=datetime(2026, 9, 1, 9, tzinfo=timezone.utc), quantity=1),
            SimpleNamespace(ts=datetime(2026, 10, 1, 9, tzinfo=timezone.utc), quantity=1),
            SimpleNamespace(ts=datetime(2026, 10, 1, 18, tzinfo=timezone.utc), quantity=2)]
    assert _purchase_days(rows) == ([date(2026, 9, 1), date(2026, 10, 1)], 3)
