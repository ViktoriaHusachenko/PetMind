"""Tests for data loading and preparation helpers."""

import pandas as pd
import pytest

from petrec.data_prep import (
    build_pet_profiles,
    extract_ingredients,
    filter_5_core,
    generate_synthetic_data,
    match_open_pet_food_facts,
    print_statistics,
    read_reviews_jsonl,
    save_processed_data,
    temporal_split,
)


def test_synthetic_data_is_deterministic_and_satisfies_5_core() -> None:
    """Synthetic fixtures should be reproducible and usable without downloads."""
    first = generate_synthetic_data()
    second = generate_synthetic_data()
    pd.testing.assert_frame_equal(first[0], second[0])
    reviews, _, _ = first
    assert len(filter_5_core(reviews)) == len(reviews)


def test_temporal_split_uses_last_two_interactions() -> None:
    """Each user's final interactions belong to validation and test in order."""
    reviews, _, _ = generate_synthetic_data(n_users=2, n_items=4, interactions_per_user=4)
    splits = temporal_split(reviews)
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2
    assert splits["val"].groupby("user_id")["timestamp"].max().lt(
        splits["test"].groupby("user_id")["timestamp"].min()
    ).all()


def test_matching_uses_opff_and_description_fallback() -> None:
    """Matched composition wins; unmatched products use the Ingredients marker."""
    products = pd.DataFrame(
        [
            {"parent_asin": "a", "title": "Brand A Cat Food", "description": "Ingredients: rice, chicken."},
            {"parent_asin": "b", "title": "Unknown Product", "description": "Ingredients: oats, fish."},
        ]
    )
    facts = pd.DataFrame(
        [{"product_name": "Cat Food", "brands": "Brand A", "ingredients_text": "turkey"}]
    )
    enriched = match_open_pet_food_facts(products, facts)
    assert enriched.loc[0, "matched_opff"]
    assert enriched.loc[0, "ingredients_text"] == "turkey"
    assert not enriched.loc[1, "matched_opff"]
    assert enriched.loc[1, "ingredients_text"] == "oats, fish"


def test_profiles_do_not_model_allergens_found_in_history() -> None:
    """Modeled allergens are absent from the user's consumed product ingredients."""
    reviews, products, _ = generate_synthetic_data(n_users=10)
    products.loc[products.index[0], "ingredients_text"] = "chicken, rice"
    profiles = build_pet_profiles(reviews, products, random_state=7, profile_rate=1.0)
    history = reviews.merge(products[["parent_asin", "ingredients_text"]], on="parent_asin")
    first_user = profiles.iloc[0]
    if "item_0" in history.loc[history["user_id"] == first_user["user_id"], "parent_asin"].values:
        assert "chicken" not in first_user["allergen_groups"]


def test_extract_ingredients_and_density(capsys: pytest.CaptureFixture[str]) -> None:
    """Description parsing and density reporting expose the expected values."""
    assert extract_ingredients("Food. Ingredients: oats, fish. Feeding guidelines: daily") == "oats, fish"
    reviews, _, _ = generate_synthetic_data(n_users=2, n_items=2, interactions_per_user=2)
    stats = print_statistics(reviews)
    assert stats["density_percent"] == 100.0
    assert "density=100.0000%" in capsys.readouterr().out


def test_jsonl_reader_converts_milliseconds_and_deduplicates(tmp_path) -> None:
    """Chunked Amazon review input converts timestamps to days and drops duplicate rows."""
    source = tmp_path / "reviews.jsonl"
    reviews = pd.DataFrame(
        [
            {"user_id": "u1", "parent_asin": "i1", "rating": 5, "timestamp": 86_400_000},
            {"user_id": "u1", "parent_asin": "i1", "rating": 5, "timestamp": 86_400_000},
            {"user_id": "u2", "parent_asin": "i2", "rating": 4, "timestamp": None},
        ]
    )
    reviews.to_json(source, orient="records", lines=True)

    loaded = read_reviews_jsonl(source, chunksize=1)

    assert len(loaded) == 1
    assert loaded.loc[0, "timestamp"] == 1


def test_processed_tables_are_saved_as_parquet(tmp_path) -> None:
    """Prepared tables and all temporal splits are persisted as Parquet files."""
    reviews, products, _ = generate_synthetic_data(n_users=2, n_items=4, interactions_per_user=4)
    splits = temporal_split(reviews)
    profiles = build_pet_profiles(reviews, products)

    save_processed_data(tmp_path, reviews, products, profiles, splits)

    assert pd.read_parquet(tmp_path / "reviews.parquet").equals(reviews)
    assert {"train.parquet", "val.parquet", "test.parquet"}.issubset(
        {path.name for path in tmp_path.glob("*.parquet")}
    )