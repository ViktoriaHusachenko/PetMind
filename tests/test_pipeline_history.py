"""BUG-025: the preference vector uses only the requesting user's purchases; others feed CF."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import petrec.pipeline as pipeline_module
from petrec.pipeline import RecommendationPipeline


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 384), dtype=np.float32)


def test_preference_vector_ignores_other_users(monkeypatch):
    seen = {}
    original = pipeline_module.build_user_preference_vector

    def capture(history, *args, **kwargs):
        seen["users"] = set(history["user_id"].astype(str))
        return original(history, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "build_user_preference_vector", capture)
    products = pd.DataFrame([{"parent_asin": f"P{i}", "title": f"Корм {i}", "species": "cat",
                              "ingredients_text": "salmon, rice", "price": 100.0} for i in range(3)])
    interactions = pd.DataFrame([
        {"user_id": "me", "parent_asin": "P0", "rating": 5.0, "timestamp": 20000.0},
        {"user_id": "other", "parent_asin": "P1", "rating": 5.0, "timestamp": 20001.0},
        {"user_id": "other", "parent_asin": "P2", "rating": 4.0, "timestamp": 20002.0},
    ])
    profile = pd.DataFrame([{"user_id": "me", "pet_species": "cat", "age_group": "adult", "allergen_groups": [],
                             "lifestyle": None, "diet_restrictions": [], "preference_state": {}}])
    from petrec.profiles import ProductEmbedder

    vectors = {f"P{i}": np.eye(384, dtype=np.float32)[i] for i in range(3)}
    pipeline = RecommendationPipeline(interactions, products, profile, item_embeddings=vectors,
                                      product_embedder=ProductEmbedder(model=FakeModel()), build_rag_index=False)
    pipeline.recommend("me", use_llm=False)
    assert seen["users"] == {"me"}
