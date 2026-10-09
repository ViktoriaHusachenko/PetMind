"""Tests for baseline recommenders and the shared evaluation interface."""

import math

import pandas as pd
import pytest

from petrec.baselines import BPRRecommender, ContentBased, ImplicitALS, ItemBasedCF, Popularity
from petrec.evaluate import evaluate


@pytest.fixture
def tiny_interactions() -> pd.DataFrame:
    """Small interactions with simple popularity and item-neighbor relationships."""
    return pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "a", "rating": 1},
            {"user_id": "u1", "parent_asin": "b", "rating": 1},
            {"user_id": "u2", "parent_asin": "a", "rating": 1},
            {"user_id": "u2", "parent_asin": "c", "rating": 1},
            {"user_id": "u3", "parent_asin": "a", "rating": 1},
        ]
    )


def test_popularity_orders_items_and_excludes_seen(tiny_interactions: pd.DataFrame) -> None:
    """Popularity recommends unseen items in descending interaction count."""
    recs = Popularity().fit(tiny_interactions).recommend(["u1"], k=2)
    assert recs["parent_asin"].tolist() == ["c"]
    assert recs["rank"].tolist() == [1]


def test_item_based_cf_returns_unseen_similar_items(tiny_interactions: pd.DataFrame) -> None:
    """An item-based CSR cosine model returns a known unseen neighbor for u1."""
    recommender = ItemBasedCF().fit(tiny_interactions)
    assert recommender.user_item_matrix is not None
    recs = recommender.recommend(["u1"], k=2)
    assert "c" in recs["parent_asin"].tolist()


def test_content_als_and_bpr_baselines_return_unseen_products(tiny_interactions: pd.DataFrame) -> None:
    products = pd.DataFrame([
        {"parent_asin": "a", "title": "chicken cat food", "category": "cat"},
        {"parent_asin": "b", "title": "salmon cat food", "category": "cat"},
        {"parent_asin": "c", "title": "beef cat food", "category": "cat"},
        {"parent_asin": "d", "title": "turkey cat food", "category": "cat"},
    ])
    seen = set(tiny_interactions.loc[tiny_interactions.user_id == "u1", "parent_asin"])
    content = ContentBased().fit(tiny_interactions, products).recommend(["u1"], k=2)
    als = ImplicitALS(factors=4, iterations=2).fit(tiny_interactions).recommend(["u1"], k=2)
    bpr = BPRRecommender(factors=4, epochs=2).fit(tiny_interactions).recommend(["u1"], k=2)
    for result in (content, als, bpr):
        assert not (set(result.parent_asin) & seen)


def test_evaluate_known_metrics_and_csv(tmp_path) -> None:
    """Macro metrics, safety, coverage, diversity, and CSV match a hand calculation."""
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "a", "rank": 1, "is_safe": True, "categories": ["cat"]},
            {"user_id": "u1", "parent_asin": "x", "rank": 2, "is_safe": False, "categories": ["dog"]},
            {"user_id": "u2", "parent_asin": "x", "rank": 1, "is_safe": True, "categories": ["dog"]},
            {"user_id": "u2", "parent_asin": "d", "rank": 2, "is_safe": True, "categories": ["fish"]},
        ]
    )
    test = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "a"},
            {"user_id": "u1", "parent_asin": "b"},
            {"user_id": "u2", "parent_asin": "d"},
        ]
    )
    output_path = tmp_path / "metrics.csv"

    result = evaluate(recs, test, k=2, output_path=output_path)
    row = result.iloc[0]

    assert row["Precision@K"] == pytest.approx(0.5)
    assert row["Recall@K"] == pytest.approx(0.75)
    assert row["NDCG@K"] == pytest.approx(
        (1 / (1 + 1 / math.log2(3)) + 1 / math.log2(3)) / 2
    )
    assert row["HR@K"] == pytest.approx(1.0)
    assert row["Safety@K"] == pytest.approx(0.75)
    assert row["Coverage"] == pytest.approx(2 / 3)
    assert row["Diversity"] == pytest.approx(1.0)
    assert row["latency_ms"] >= 0
    assert pd.read_csv(output_path).shape == (1, len(result.columns))


def test_evaluate_safety_is_nan_without_safety_data(tmp_path) -> None:
    """Safety is explicitly unavailable when allergen or safe-flag data is absent."""
    recs = pd.DataFrame([{"user_id": "u1", "parent_asin": "a"}])
    test = pd.DataFrame([{"user_id": "u1", "parent_asin": "a"}])
    result = evaluate(recs, test, k=1, output_path=tmp_path / "metrics.csv")
    assert pd.isna(result.loc[0, "Safety@K"])
