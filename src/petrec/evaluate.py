"""Unified offline recommendation metrics."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def evaluate(
    recs: pd.DataFrame,
    test: pd.DataFrame,
    k: int,
    output_path: str | Path = "evaluation_results.csv",
) -> pd.DataFrame:
    """Compute macro ranking and safety metrics, catalog coverage, diversity, and latency.

    ``recs`` must contain ``user_id`` and ``parent_asin`` and may include ``rank`` or
    descending ``score``. Safety is computed from a boolean ``is_safe`` recommendation
    column, or from per-user ``allergen_groups`` in ``test`` and recommendation
    ``ingredients_text``. If neither is available, Safety@K is NaN. Latency uses a
    supplied ``latency_ms`` column when present; otherwise it measures this evaluation
    call's wall-clock time. Coverage uses distinct relevant catalog items in ``test``.
    Diversity is average pairwise category Jaccard distance when ``categories`` are
    supplied, and otherwise the fraction of distinct items in the pooled top-K lists.
    The aggregate result is always saved to ``output_path`` as CSV.
    """
    if k < 1:
        raise ValueError("k must be positive")
    for frame_name, frame in (("recs", recs), ("test", test)):
        missing = {"user_id", "parent_asin"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")

    started = time.perf_counter()
    recommendations = recs.copy()
    relevant_by_user = test.groupby("user_id")["parent_asin"].agg(set).to_dict()
    if "rank" in recommendations:
        recommendations = recommendations.sort_values(["user_id", "rank"], kind="stable")
    elif "score" in recommendations:
        recommendations = recommendations.sort_values(
            ["user_id", "score"], ascending=[True, False], kind="stable"
        )
    recommendations = recommendations.drop_duplicates(["user_id", "parent_asin"], keep="first")
    top_by_user = {
        user_id: group.head(k).copy()
        for user_id, group in recommendations.groupby("user_id", sort=False)
    }
    evaluated_users = list(relevant_by_user)
    if not evaluated_users:
        result = pd.DataFrame(
            [{
                "Precision@K": np.nan,
                "Recall@K": np.nan,
                "NDCG@K": np.nan,
                "HR@K": np.nan,
                "Safety@K": np.nan,
                "Coverage": 0.0,
                "Diversity": 0.0,
                "latency_ms": 0.0,
                "users": 0,
            }]
        )
        result.to_csv(output_path, index=False)
        return result

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    hit_rates: list[float] = []
    safety_values: list[float] = []
    all_recommended: list[Any] = []
    category_lists: list[list[set[str]]] = []

    for user_id in evaluated_users:
        relevant = relevant_by_user[user_id]
        user_recs = top_by_user.get(user_id, recommendations.iloc[0:0])
        items = user_recs["parent_asin"].tolist()
        hits = [item in relevant for item in items]
        hit_count = sum(hits)
        precisions.append(hit_count / k)
        recalls.append(hit_count / len(relevant) if relevant else 0.0)
        dcg = sum(1.0 / np.log2(position + 2) for position, is_hit in enumerate(hits) if is_hit)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(position + 2) for position in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        hit_rates.append(float(hit_count > 0))
        all_recommended.extend(items)

        user_categories: list[set[str]] = []
        if "categories" in user_recs:
            user_categories = [_as_category_set(value) for value in user_recs["categories"]]
        if user_categories:
            category_lists.append(user_categories)

        safety = _safety_for_user(user_recs, test, user_id)
        if safety is not None:
            safety_values.append(safety)

    test_catalog = set(test["parent_asin"].dropna())
    coverage = len(set(all_recommended).intersection(test_catalog)) / len(test_catalog) if test_catalog else 0.0
    diversity = _diversity(category_lists) if category_lists else (
        len(set(all_recommended)) / len(all_recommended) if all_recommended else 0.0
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if "latency_ms" in recs and recs["latency_ms"].notna().any():
        latency_ms = float(pd.to_numeric(recs["latency_ms"], errors="coerce").dropna().mean())
    else:
        latency_ms = elapsed_ms

    result = pd.DataFrame(
        [
            {
                "Precision@K": float(np.mean(precisions)),
                "Recall@K": float(np.mean(recalls)),
                "NDCG@K": float(np.mean(ndcgs)),
                "HR@K": float(np.mean(hit_rates)),
                "Safety@K": float(np.mean(safety_values)) if safety_values else np.nan,
                "Coverage": coverage,
                "Diversity": diversity,
                "latency_ms": latency_ms,
                "users": len(evaluated_users),
            }
        ]
    )
    result.to_csv(output_path, index=False)
    return result


def _safety_for_user(
    user_recs: pd.DataFrame,
    test: pd.DataFrame,
    user_id: Any,
) -> float | None:
    """Calculate the safe fraction when recommendation or profile safety data exists."""
    if user_recs.empty:
        return None
    if "is_safe" in user_recs and user_recs["is_safe"].notna().any():
        return float(user_recs["is_safe"].fillna(False).astype(bool).mean())
    profile_rows = test.loc[test["user_id"] == user_id]
    if "allergen_groups" not in profile_rows or "ingredients_text" not in user_recs:
        return None
    allergy_values = profile_rows["allergen_groups"].dropna()
    if allergy_values.empty:
        return None
    allergens = _as_text_set(allergy_values.iloc[0])
    ingredients = user_recs["ingredients_text"].fillna("").astype(str).str.lower()
    if not allergens:
        return 1.0
    safe = [not any(allergen in text for allergen in allergens) for text in ingredients]
    return float(np.mean(safe)) if safe else None


def _as_text_set(value: Any) -> set[str]:
    """Normalize a scalar or collection into lowercase nonempty strings."""
    if value is None or (not isinstance(value, (list, tuple, set, np.ndarray)) and pd.isna(value)):
        return set()
    values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _as_category_set(value: Any) -> set[str]:
    """Normalize category values for pairwise Jaccard diversity."""
    if isinstance(value, dict):
        values: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        values = value
    else:
        values = [value]
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _diversity(category_lists: list[list[set[str]]]) -> float:
    """Return the mean pairwise Jaccard distance across each user's recommendations."""
    distances: list[float] = []
    for categories in category_lists:
        for left_index, left in enumerate(categories):
            for right in categories[left_index + 1 :]:
                union = left | right
                distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return float(np.mean(distances)) if distances else 0.0