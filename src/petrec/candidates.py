"""Hybrid candidate generation with safety and recent-purchase constraints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    low, high = float(np.min(values)), float(np.max(values))
    if high == low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def item_cosine(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return item-item cosine similarity from binary user × item columns."""
    required = {"user_id", "item_id"}
    if not required.issubset(interactions.columns):
        # Existing project interaction tables use parent_asin.
        if not {"user_id", "parent_asin"}.issubset(interactions.columns):
            raise ValueError("interactions must contain user_id and item_id/parent_asin")
        item_column = "parent_asin"
    else:
        item_column = "item_id"
    binary = pd.crosstab(interactions["user_id"], interactions[item_column]).clip(upper=1)
    matrix = cosine_similarity(binary.T) if not binary.empty else np.zeros((0, 0))
    return pd.DataFrame(matrix, index=binary.columns, columns=binary.columns)


def generate_candidates(
    user_id: Any,
    interactions: pd.DataFrame,
    candidate_items: Iterable[Any] | pd.DataFrame,
    item_embeddings: Mapping[Any, np.ndarray],
    user_embedding: np.ndarray,
    *,
    safe: Mapping[Any, bool] | Callable[[Any], bool] | None = None,
    purchases: pd.DataFrame | None = None,
    regular_items: Iterable[Any] = (),
    now: Any | None = None,
    lookback_days: int = 7,
    beta: float = 0.5,
    top_n: int = 50,
    time_decay_half_life_days: float | None = None,
    reference_time: float | None = None,
) -> pd.DataFrame:
    """Generate a user's ranked candidate set.

    ``interactions`` accepts project columns ``parent_asin`` or generic ``item_id``.
    ``safe`` may be an item→bool map or a callable; missing safety verdicts fail closed.
    ``purchases`` requires ``item_id`` (or ``parent_asin``) and ``timestamp``; rows
    within the lookback window are excluded unless that item is marked regular.
    """
    if not 0 <= beta <= 1 or top_n < 1 or lookback_days < 0:
        raise ValueError("beta must be in [0,1], top_n positive, and lookback_days nonnegative")
    if "user_id" not in interactions:
        raise ValueError("interactions must contain user_id")
    column = "item_id" if "item_id" in interactions else "parent_asin"
    if column not in interactions:
        raise ValueError("interactions must contain item_id or parent_asin")
    history = interactions.loc[interactions.user_id == user_id]
    history_ids = history[column].tolist()
    unique_history = list(dict.fromkeys(history_ids))
    candidate_ids = (candidate_items["item_id" if "item_id" in candidate_items else "parent_asin"].tolist()
                     if isinstance(candidate_items, pd.DataFrame) else list(candidate_items))
    candidate_ids = list(dict.fromkeys(candidate_ids))

    if purchases is not None and not purchases.empty:
        purchase_column = "item_id" if "item_id" in purchases else "parent_asin"
        if "timestamp" not in purchases or purchase_column not in purchases:
            raise ValueError("purchases must contain timestamp and item_id/parent_asin")
        recent = purchases.copy()
        if "user_id" in recent:
            recent = recent.loc[recent.user_id == user_id]
        times = pd.to_datetime(recent.timestamp, errors="coerce", utc=True)
        if now is None:
            reference = pd.Timestamp.now(tz="UTC")
        else:
            reference = pd.to_datetime(now, utc=True)
        cutoff = reference - pd.Timedelta(days=lookback_days)
        regular = set(regular_items)
        recent_ids = set(recent.loc[times >= cutoff, purchase_column]) - regular
        candidate_ids = [item for item in candidate_ids if item not in recent_ids]

    item_column = "item_id" if "item_id" in interactions else "parent_asin"
    binary = pd.crosstab(interactions.user_id, interactions[item_column]).clip(upper=1)
    similarity = cosine_similarity(binary.T) if not binary.empty else np.zeros((0, 0))
    positions = {item: index for index, item in enumerate(binary.columns)}
    weights: dict[Any, float] = {}
    if time_decay_half_life_days is not None and time_decay_half_life_days <= 0:
        raise ValueError("time_decay_half_life_days must be positive")
    reference = reference_time
    if reference is None and "timestamp" in history and not history.empty:
        reference = float(pd.to_numeric(history.timestamp, errors="coerce").max())
    for row in history.to_dict("records"):
        value = row.get("weight", row.get("rating", 1.0))
        try:
            weights[row[item_column]] = weights.get(row[item_column], 0.0) + max(0.0, float(value))
        except (TypeError, ValueError):
            weights[row[item_column]] = weights.get(row[item_column], 0.0) + 1.0
        if time_decay_half_life_days is not None and reference is not None and row.get("timestamp") is not None:
            try:
                age = max(0.0, float(reference) - float(row["timestamp"]))
                weights[row[item_column]] *= 0.5 ** (age / time_decay_half_life_days)
            except (TypeError, ValueError):
                pass
    n_user = len(unique_history)
    effective_beta = beta if n_user >= 3 else 0.0
    content_vector = np.asarray(user_embedding, dtype=float).reshape(1, -1)

    rows = []
    for item in candidate_ids:
        is_safe = bool(safe(item) if callable(safe) else safe.get(item, False) if safe is not None else False)
        if not is_safe:
            continue
        vector = item_embeddings.get(item)
        if vector is None:
            content_score = 0.0
        else:
            vector = np.asarray(vector, dtype=float).reshape(1, -1)
            content_score = float(cosine_similarity(vector, content_vector)[0, 0]) if vector.shape[1] == content_vector.shape[1] else 0.0
        cf_numerator = 0.0
        cf_denominator = 0.0
        if item in positions:
            for seen, weight in weights.items():
                if seen in positions:
                    cf_numerator += weight * similarity[positions[item], positions[seen]]
                    cf_denominator += weight
        cf_score = cf_numerator / cf_denominator if cf_denominator > 0 else 0.0
        rows.append({"item_id": item, "s_cf": cf_score, "s_content": content_score, "safe": True})

    result = pd.DataFrame(rows, columns=["item_id", "s_cf", "s_content", "safe"])
    if result.empty:
        result["s_cand"] = pd.Series(dtype=float)
        return result[["item_id", "s_cf", "s_content", "s_cand", "safe"]]
    cf_norm = _minmax(result.s_cf.to_numpy(dtype=float))
    content_norm = _minmax(result.s_content.to_numpy(dtype=float))
    result["s_cand"] = effective_beta * cf_norm + (1.0 - effective_beta) * content_norm
    result = result.sort_values(["s_cand", "item_id"], ascending=[False, True], key=lambda col: col.map(str) if col.name == "item_id" else col, kind="stable")
    return result.head(top_n).reset_index(drop=True)[["item_id", "s_cf", "s_content", "s_cand", "safe"]]


def build_candidates(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for :func:`generate_candidates`."""
    return generate_candidates(*args, **kwargs)
