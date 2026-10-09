"""Tests for temporal user profiles and E5 embedding behavior."""

import numpy as np
import pandas as pd
import pytest

from petrec.profiles import (
    AnimalProfile,
    ProductEmbedder,
    build_user_preference_vector,
    compute_interaction_weights,
)


class FakeEncoder:
    """Small deterministic encoder for tests without model downloads."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([[len(text), sum(map(ord, text)) % 17] for text in texts], dtype=np.float32)


def test_interaction_weights_sum_and_follow_half_life() -> None:
    """Unit values at one half-life and current time sum to 1.5 and older weighs less."""
    history = pd.DataFrame(
        [
            {"parent_asin": "old", "timestamp": 10.0},
            {"parent_asin": "new", "timestamp": 100.0},
        ]
    )
    weights = compute_interaction_weights(history, half_life_days=90, reference_time=100)
    assert weights.sum() == pytest.approx(1.5)
    assert weights[0] < weights[1]


def test_rated_history_uses_rating_scaled_weight() -> None:
    """A rating of 2 contributes value 0.4 before temporal decay."""
    history = pd.DataFrame([{"parent_asin": "i1", "rating": 2, "timestamp": 5.0}])
    weights = compute_interaction_weights(history, half_life_days=90, reference_time=5)
    assert weights.tolist() == pytest.approx([0.4])


def test_cold_start_uses_query_prefixed_animal_profile() -> None:
    """An empty history embeds profile text with the E5 query prefix."""
    encoder = FakeEncoder()
    embedder = ProductEmbedder(model=encoder)
    profile = AnimalProfile(
        species="cat",
        age_group="adult",
        diet_restrictions=["sterilized"],
        allergens=["poultry"],
    )

    vector = build_user_preference_vector(
        pd.DataFrame(columns=["parent_asin", "timestamp"]),
        {},
        profile=profile,
        embedder=embedder,
    )

    assert vector.shape == (2,)
    assert encoder.calls == [["query: adult cat sterilized without poultry"]]


def test_product_embeddings_use_passage_prefix_and_npy_cache(tmp_path) -> None:
    """Product embeddings use passage prefixes and valid cache reuse avoids re-encoding."""
    products = pd.DataFrame([{"parent_asin": "a1", "title": "dry cat food"}])
    encoder = FakeEncoder()
    cache_path = tmp_path / "items.npy"
    first = ProductEmbedder(cache_path=cache_path, model=encoder).embed_products(products)
    assert encoder.calls == [["passage: dry cat food"]]

    second = ProductEmbedder(cache_path=cache_path, model=FakeEncoder()).embed_products(products)
    np.testing.assert_array_equal(first["a1"], second["a1"])
    assert cache_path.exists()