"""LLM-assisted candidate reranking with strict ID validation and MMR selection."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics.pairwise import cosine_similarity

from .llm import LLMClient

SYSTEM_PROMPT = """[Роль]
Ти помічник з ранжування товарів для домашніх тварин.

[Обмеження]
Оцінюй лише товари зі списку, не вигадуй інгредієнтів, не став діагнозів,
відповідай українською. Оцінюй відповідність описаним потребам, а не лікувальну дію.

[Контекст]
Надається профіль тварини, критерії та список дозволених товарів із контекстом.

[Формат відповіді]
Поверни лише JSON: {\"items\":[{\"id\":\"...\",\"score\":1,\"reason\":\"...\"}]}
score — ціле число від 1 до 5; reason — не більше 200 символів."""


class RankedItem(BaseModel):
    id: str
    score: int = Field(ge=1, le=5)
    reason: str = Field(max_length=200)


class RankedResponse(BaseModel):
    items: list[RankedItem]


class RecommendationItem(BaseModel):
    """Stable response shape consumed by the Petly web interface."""
    item_id: str
    title: str = ""
    brand: str = ""
    price: float | None = None
    image_url: str | None = None
    match_percent: int
    reason: str = Field(max_length=200)
    safe_badge: bool
    composition_summary: list[str]
    llm_used: bool


def _parse_response(raw: str, allowed: set[str]) -> dict[str, RankedItem]:
    value = json.loads(raw)
    parsed = RankedResponse.model_validate(value) if hasattr(RankedResponse, "model_validate") else RankedResponse.parse_obj(value)
    ids = [item.id for item in parsed.items]
    if len(ids) != len(set(ids)):
        raise ValueError("response contains duplicate IDs")
    if not set(ids).issubset(allowed):
        raise ValueError("response contains IDs outside candidate set")
    return {item.id: item for item in parsed.items}


def _fallback_weights(gamma1: float, gamma2: float) -> tuple[float, float]:
    total = gamma1 + gamma2
    return (gamma1 / total, gamma2 / total) if total > 0 else (0.5, 0.5)


def rerank_candidates(
    candidates: pd.DataFrame,
    llm_client: LLMClient | None,
    *,
    context: str = "",
    item_context: Mapping[Any, str] | None = None,
    item_embeddings: Mapping[Any, np.ndarray] | None = None,
    gamma1: float = 0.4,
    gamma2: float = 0.3,
    gamma3: float = 0.3,
    mu: float = 0.8,
    k: int = 10,
    safe_evaluator: Any | None = None,
    filter_unsafe: bool = True,
) -> pd.DataFrame:
    """LLM rerank then select diverse top-K; invalid answers retry once then fall back.

    Fallback removes the unavailable LLM contribution and proportionally normalizes
    gamma1/gamma2 to sum to one. Candidates marked unsafe are removed before prompting.
    """
    if k < 1 or not 0 <= mu <= 1 or min(gamma1, gamma2, gamma3) < 0:
        raise ValueError("invalid k, mu, or gamma weights")
    required = {"item_id", "s_cand"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates must contain {sorted(required)}")
    data = candidates.copy()
    # Safety is a deterministic gate. An LLM response cannot create or override it.
    if safe_evaluator is not None:
        data["safe"] = [bool(safe_evaluator(row)) for row in data.to_dict("records")]
    elif "safe" not in data:
        data["safe"] = False
    data["safe"] = data.safe.fillna(False).astype(bool)
    if filter_unsafe:
        data = data.loc[data.safe].copy()
    if data.empty:
        data["s_LLM"] = pd.Series(dtype=float)
        data["s_final"] = pd.Series(dtype=float)
        data["reason"] = pd.Series(dtype=str)
        return data
    if "S_comp" not in data:
        data["S_comp"] = 0.5
    allowed = {str(value) for value in data.item_id}
    listing = [{"id": str(row.item_id), "s_cand": float(row.s_cand),
                "composition_score": float(row.S_comp),
                "context": (item_context or {}).get(row.item_id, "")}
               for row in data.itertuples(index=False)]
    user_message = json.dumps({"context": context, "candidates": listing}, ensure_ascii=False)
    valid: dict[str, RankedItem] | None = None
    error: Exception | None = None
    if llm_client is not None:
        for attempt in range(2):
            prompt = user_message if attempt == 0 else (
                user_message + "\nПопередня відповідь була невалідною (" + str(error) +
                "). Виправ її та поверни лише JSON із id лише з наданого списку.")
            try:
                valid = _parse_response(llm_client.complete(SYSTEM_PROMPT, prompt), allowed)
                break
            except Exception as exc:
                error = exc
    fallback = valid is None
    g1, g2 = _fallback_weights(gamma1, gamma2) if fallback else (gamma1, gamma2)
    g3 = 0.0 if fallback else gamma3
    if not fallback:
        total = g1 + g2 + g3
        if total <= 0:
            raise ValueError("gamma weights must have a positive sum")
        g1, g2, g3 = g1 / total, g2 / total, g3 / total

    llm_scores, reasons = [], []
    for item in data.item_id:
        result = None if fallback else valid.get(str(item))
        score = float(result.score) if result else 3.0
        llm_scores.append((score - 1.0) / 4.0)
        reasons.append(result.reason if result else None)
    data["s_LLM"] = llm_scores
    data["reason"] = reasons
    data["s_final"] = g1 * data.s_cand.astype(float) + g2 * data.S_comp.astype(float) + g3 * data.s_LLM
    data["llm_fallback"] = fallback
    data["llm_used"] = [reason is not None for reason in reasons]

    # Greedy MMR: relevance minus maximum cosine similarity to already selected items.
    remaining = list(range(len(data)))
    selected: list[int] = []
    ids = data.item_id.tolist()
    while remaining and len(selected) < k:
        best, best_value = None, -float("inf")
        for index in remaining:
            diversity_penalty = 0.0
            if selected and item_embeddings is not None and ids[index] in item_embeddings:
                vector = np.asarray(item_embeddings[ids[index]], dtype=float).reshape(1, -1)
                similarities = []
                for picked in selected:
                    if ids[picked] in item_embeddings:
                        other = np.asarray(item_embeddings[ids[picked]], dtype=float).reshape(1, -1)
                        if vector.shape[1] == other.shape[1]:
                            similarities.append(float(cosine_similarity(vector, other)[0, 0]))
                diversity_penalty = max(similarities, default=0.0)
            value = mu * float(data.iloc[index].s_final) - (1 - mu) * diversity_penalty
            if value > best_value:
                best, best_value = index, value
        selected.append(best)
        remaining.remove(best)
    result = data.iloc[selected].copy().reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def to_recommendation_items(ranked: pd.DataFrame) -> list[RecommendationItem]:
    """Convert ranked rows to the frontend contract, deriving badge only from Safe."""
    recommendations = []
    for row in ranked.to_dict(orient="records"):
        safe_badge = bool(row.get("safe", False))
        ingredients = row.get("ingredients", row.get("composition_summary", []))
        if isinstance(ingredients, str):
            ingredients = [part.strip() for part in ingredients.replace(";", ",").split(",") if part.strip()]
        elif not isinstance(ingredients, (list, tuple)):
            ingredients = []
        ingredients = [str(value.canonical_name if hasattr(value, "canonical_name") else value) for value in ingredients[:3]]
        llm_used = bool(row.get("llm_used", not bool(row.get("llm_fallback", False))))
        score = float(row.get("s_final", 0.0))
        if llm_used:
            reason = str(row.get("reason", ""))[:200]
        else:
            composition = float(row.get("S_comp", 0.5))
            reason = f"Відповідність складу та харчовим показникам: {composition:.0%}."
        recommendations.append(RecommendationItem(
            item_id=str(row.get("item_id", "")), title=str(row.get("title") or ""),
            brand=str(row.get("brand") or ""),
            price=float(row["price"]) if row.get("price") is not None and pd.notna(row.get("price")) else None,
            image_url=str(row["image_url"]) if row.get("image_url") else None,
            match_percent=round(100 * score), reason=reason,
            safe_badge=safe_badge, composition_summary=ingredients,
            llm_used=llm_used,
        ))
    return recommendations


def rerank_recommendations(*args: Any, **kwargs: Any) -> list[RecommendationItem]:
    """Run reranking and return frontend-ready RecommendationItem models."""
    return to_recommendation_items(rerank_candidates(*args, **kwargs))


rerank = rerank_candidates
