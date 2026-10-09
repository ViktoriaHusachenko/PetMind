"""Loading, cleaning, enriching, and splitting pet-product interaction data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

DAY_MS = 86_400_000
REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
PRODUCT_COLUMNS = ["parent_asin", "title", "categories", "description", "price"]
OPFF_COLUMNS = [
    "product_name",
    "brands",
    "ingredients_text",
    "proteins_100g",
    "fat_100g",
    "fiber_100g",
    "energy-kcal_100g",
]
ALLERGEN_GROUPS = {
    "chicken": ("chicken", "poultry"),
    "beef": ("beef",),
    "fish": ("fish", "salmon", "tuna"),
    "dairy": ("milk", "dairy", "cheese", "whey"),
    "wheat": ("wheat", "gluten"),
    "soy": ("soy", "soya"),
    "egg": ("egg",),
    "corn": ("corn", "maize"),
}


def read_reviews_jsonl(path: str | Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Read Amazon review JSONL in chunks and convert millisecond timestamps to days."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_json(path, lines=True, chunksize=chunksize, convert_dates=False):
        missing = set(REVIEW_COLUMNS).difference(chunk.columns)
        if missing:
            raise ValueError(f"Review data is missing required columns: {sorted(missing)}")
        chunk = chunk.loc[:, REVIEW_COLUMNS].copy()
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce") / DAY_MS
        chunk = chunk.dropna(subset=["user_id", "parent_asin", "timestamp"])
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return clean_reviews(pd.concat(chunks, ignore_index=True))


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete or duplicate interactions and normalize timestamps to days."""
    missing = set(REVIEW_COLUMNS).difference(reviews.columns)
    if missing:
        raise ValueError(f"Review data is missing required columns: {sorted(missing)}")

    cleaned = reviews.loc[:, REVIEW_COLUMNS].copy()
    cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["user_id", "parent_asin", "rating", "timestamp"])
    if not cleaned.empty and cleaned["timestamp"].abs().max() > 10_000_000:
        cleaned["timestamp"] = cleaned["timestamp"] / DAY_MS
    return cleaned.drop_duplicates(subset=REVIEW_COLUMNS).reset_index(drop=True)


def filter_5_core(reviews: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Iteratively retain users and products with at least ``k`` interactions."""
    if k < 1:
        raise ValueError("k must be positive")
    filtered = clean_reviews(reviews)
    while not filtered.empty:
        user_counts = filtered["user_id"].value_counts()
        item_counts = filtered["parent_asin"].value_counts()
        keep = filtered["user_id"].isin(user_counts[user_counts >= k].index)
        keep &= filtered["parent_asin"].isin(item_counts[item_counts >= k].index)
        next_filtered = filtered.loc[keep].reset_index(drop=True)
        if len(next_filtered) == len(filtered):
            break
        filtered = next_filtered
    return filtered


def sample_users(
    reviews: pd.DataFrame,
    n_users: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep a reproducible random sample of up to ``n_users`` distinct users."""
    if not 5_000 <= n_users <= 10_000:
        raise ValueError("n_users must be between 5,000 and 10,000")
    users = reviews["user_id"].dropna().drop_duplicates().to_numpy()
    if len(users) <= n_users:
        return reviews.copy().reset_index(drop=True)
    selected = np.random.default_rng(random_state).choice(users, size=n_users, replace=False)
    return reviews.loc[reviews["user_id"].isin(selected)].reset_index(drop=True)


def read_product_metadata(path: str | Path) -> pd.DataFrame:
    """Read Amazon product metadata JSONL and return the requested product fields."""
    products = pd.read_json(path, lines=True)
    if "parent_asin" not in products:
        raise ValueError("Product metadata is missing required column: parent_asin")
    for column in PRODUCT_COLUMNS:
        if column not in products:
            products[column] = None
    return products.loc[:, PRODUCT_COLUMNS].drop_duplicates("parent_asin").reset_index(drop=True)


def read_opff_tsv(path: str | Path) -> pd.DataFrame:
    """Read the relevant Open Pet Food Facts fields from a tab-separated export."""
    products = pd.read_csv(path, sep="\t", low_memory=False)
    for column in OPFF_COLUMNS:
        if column not in products:
            products[column] = None
    return products.loc[:, OPFF_COLUMNS].copy()


def _text(value: Any) -> str:
    """Flatten simple nested metadata values into searchable text."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return str(value)


def extract_ingredients(description: Any) -> str | None:
    """Extract the text after an ``Ingredients:`` marker in a product description."""
    text = _text(description)
    match = re.search(r"\bingredients\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    ingredients = match.group(1).splitlines()[0].strip()
    ingredients = re.split(
        r"\b(?:nutrition facts|guaranteed analysis|feeding guidelines|directions)\b",
        ingredients,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .;\t")
    return ingredients or None


def match_open_pet_food_facts(
    products: pd.DataFrame,
    opff: pd.DataFrame,
    threshold: int = 85,
) -> pd.DataFrame:
    """Fuzzy-match product brand/title to OPFF name/brand, then use description fallback."""
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    result = products.copy().reset_index(drop=True)
    facts = opff.copy().reset_index(drop=True)
    for column in OPFF_COLUMNS:
        if column not in facts:
            facts[column] = None

    choices = [
        " ".join(part for part in (_text(row.brands), _text(row.product_name)) if part)
        for row in facts.itertuples(index=False)
    ]
    matched_rows: list[dict[str, Any]] = []
    for product in result.itertuples(index=False):
        product_values = product._asdict()
        title = _text(product_values.get("title"))
        brand = _text(product_values.get("brand"))
        query = " ".join(part for part in (brand, title) if part)
        match = process.extractOne(query, choices, scorer=fuzz.WRatio) if query and choices else None
        row: dict[str, Any] = {}
        if match is not None and match[1] >= threshold:
            row = facts.iloc[match[2]].to_dict()
            row["matched_opff"] = True
        else:
            row = {column: None for column in OPFF_COLUMNS}
            row["matched_opff"] = False
        ingredients = row.get("ingredients_text")
        if not isinstance(ingredients, str) or not ingredients.strip():
            ingredients = extract_ingredients(product_values.get("description"))
        row["ingredients_text"] = ingredients
        matched_rows.append(row)

    enrichment = pd.DataFrame(matched_rows)
    for column in OPFF_COLUMNS:
        result[column] = enrichment[column] if column in enrichment else None
    result["matched_opff"] = enrichment["matched_opff"] if "matched_opff" in enrichment else False
    return result


def _category_text(value: Any) -> str:
    """Convert category trees and lists to lowercase searchable text."""
    return _text(value).lower()


def _allergen_groups_in(text: str) -> set[str]:
    """Return known allergen groups mentioned in an ingredient string."""
    found: set[str] = set()
    lowered = text.lower()
    for group, terms in ALLERGEN_GROUPS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            found.add(group)
    return found


def build_pet_profiles(
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    random_state: int = 42,
    profile_rate: float = 0.3,
) -> pd.DataFrame:
    """Infer species and age group, and model allergies absent from each user's history."""
    if not 0 <= profile_rate <= 1:
        raise ValueError("profile_rate must be between 0 and 1")
    product_fields = products.copy()
    for column in ("parent_asin", "categories", "title", "ingredients_text"):
        if column not in product_fields:
            product_fields[column] = None
    history = reviews[["user_id", "parent_asin"]].merge(
        product_fields[["parent_asin", "categories", "title", "ingredients_text"]],
        on="parent_asin",
        how="left",
    )
    rng = np.random.default_rng(random_state)
    users = reviews["user_id"].dropna().drop_duplicates().tolist()
    selected_count = int(round(len(users) * profile_rate))
    selected_users = set(rng.choice(users, size=selected_count, replace=False)) if selected_count else set()
    profiles: list[dict[str, Any]] = []

    for user_id in users:
        user_history = history.loc[history["user_id"] == user_id]
        cat_count = sum("cat" in _category_text(value) or "cats" in _category_text(value) for value in user_history["categories"])
        dog_count = sum("dog" in _category_text(value) or "dogs" in _category_text(value) for value in user_history["categories"])
        pet_species = "cats" if cat_count > dog_count else "dogs" if dog_count > cat_count else "unknown"

        history_text = " ".join(
            _text(value) for value in user_history["title"].tolist()
        ).lower()
        if "kitten" in history_text:
            age_group = "kitten"
        elif "puppy" in history_text:
            age_group = "puppy"
        elif "senior" in history_text:
            age_group = "senior"
        else:
            age_group = "adult"

        seen_allergens = _allergen_groups_in(
            " ".join(_text(value) for value in user_history["ingredients_text"].tolist())
        )
        allergen_candidates = sorted(set(ALLERGEN_GROUPS).difference(seen_allergens))
        modeled_allergens: list[str] = []
        if user_id in selected_users and allergen_candidates:
            allergen_count = int(rng.integers(1, len(allergen_candidates) + 1))
            modeled_allergens = sorted(
                rng.choice(allergen_candidates, size=allergen_count, replace=False).tolist()
            )
        profiles.append(
            {
                "user_id": user_id,
                "pet_species": pet_species,
                "age_group": age_group,
                "allergen_groups": modeled_allergens,
            }
        )
    return pd.DataFrame(profiles)


def temporal_split(reviews: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split each user's penultimate interaction to validation and last to test."""
    cleaned = clean_reviews(reviews).sort_values(
        ["user_id", "timestamp", "parent_asin"], kind="stable"
    )
    train_indices: list[Any] = []
    val_indices: list[Any] = []
    test_indices: list[Any] = []
    for _, group in cleaned.groupby("user_id", sort=False):
        indices = group.index.tolist()
        if len(indices) == 1:
            test_indices.extend(indices)
        elif len(indices) == 2:
            val_indices.append(indices[-2])
            test_indices.append(indices[-1])
        else:
            train_indices.extend(indices[:-2])
            val_indices.append(indices[-2])
            test_indices.append(indices[-1])
    return {
        "train": cleaned.loc[train_indices].reset_index(drop=True),
        "val": cleaned.loc[val_indices].reset_index(drop=True),
        "test": cleaned.loc[test_indices].reset_index(drop=True),
    }


def save_processed_data(
    output_dir: str | Path,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    profiles: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    """Write the prepared tables and temporal splits as Parquet files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(output_path / "reviews.parquet", index=False)
    products.to_parquet(output_path / "products.parquet", index=False)
    profiles.to_parquet(output_path / "profiles.parquet", index=False)
    for split_name, split in splits.items():
        split.to_parquet(output_path / f"{split_name}.parquet", index=False)


def print_statistics(reviews: pd.DataFrame, label: str = "dataset") -> dict[str, int | float]:
    """Print interaction counts and matrix density as a percentage."""
    interaction_count = len(reviews)
    user_count = int(reviews["user_id"].nunique()) if "user_id" in reviews else 0
    item_count = int(reviews["parent_asin"].nunique()) if "parent_asin" in reviews else 0
    density = (
        interaction_count / (user_count * item_count) * 100
        if user_count and item_count
        else 0.0
    )
    stats: dict[str, int | float] = {
        "interactions": interaction_count,
        "users": user_count,
        "items": item_count,
        "density_percent": density,
    }
    print(
        f"{label}: interactions={interaction_count:,}, users={user_count:,}, "
        f"items={item_count:,}, density={density:.4f}%"
    )
    return stats


def prepare_data(
    reviews_path: str | Path,
    metadata_path: str | Path,
    opff_path: str | Path,
    output_dir: str | Path,
    n_users: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the complete local-data preparation pipeline and save its Parquet outputs."""
    reviews = read_reviews_jsonl(reviews_path)
    reviews = sample_users(filter_5_core(reviews), n_users=n_users, random_state=random_state)
    products = match_open_pet_food_facts(
        read_product_metadata(metadata_path), read_opff_tsv(opff_path)
    )
    profiles = build_pet_profiles(reviews, products, random_state=random_state)
    splits = temporal_split(reviews)
    save_processed_data(output_dir, reviews, products, profiles, splits)
    print_statistics(reviews, "all")
    for split_name, split in splits.items():
        print_statistics(split, split_name)
    return {"reviews": reviews, "products": products, "profiles": profiles, **splits}


def generate_synthetic_data(
    n_users: int = 20,
    n_items: int = 12,
    interactions_per_user: int = 6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deterministic reviews, metadata, and OPFF-like tables for offline tests."""
    if n_users < 1 or n_items < 1 or interactions_per_user < 1:
        raise ValueError("n_users, n_items, and interactions_per_user must be positive")
    rng = np.random.default_rng(random_state)
    products: list[dict[str, Any]] = []
    opff_rows: list[dict[str, Any]] = []
    for item_index in range(n_items):
        species = "Cats" if item_index % 2 == 0 else "Dogs"
        age_word = "kitten" if species == "Cats" else "puppy"
        title = f"Brand{item_index} {species} {age_word} food {item_index}"
        products.append(
            {
                "parent_asin": f"item_{item_index}",
                "title": title,
                "categories": [species],
                "description": "Ingredients: rice, vegetables, vitamins.",
                "price": 10.0 + item_index,
            }
        )
        opff_rows.append(
            {
                "product_name": title,
                "brands": f"Brand{item_index}",
                "ingredients_text": "rice, vegetables, vitamins",
                "proteins_100g": 20.0,
                "fat_100g": 10.0,
                "fiber_100g": 3.0,
                "energy-kcal_100g": 350.0,
            }
        )

    interactions: list[dict[str, Any]] = []
    for user_index in range(n_users):
        for interaction_index in range(interactions_per_user):
            item_index = (user_index * interactions_per_user + interaction_index) % n_items
            interactions.append(
                {
                    "user_id": f"user_{user_index}",
                    "parent_asin": f"item_{item_index}",
                    "rating": float(rng.integers(1, 6)),
                    "timestamp": float(19_000 + user_index * interactions_per_user + interaction_index),
                }
            )
    return pd.DataFrame(interactions), pd.DataFrame(products), pd.DataFrame(opff_rows)