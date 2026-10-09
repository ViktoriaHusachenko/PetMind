"""End-to-end personalized recommendation pipeline (candidate → evidence → rerank)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .composition import composition_score
from .llm import LLMClient
from .profiles import AnimalProfile, ProductEmbedder, build_user_preference_vector
from .rag import Embedder, RAGIndex
from .rerank import RecommendationItem, rerank_recommendations

logger = logging.getLogger("petrec.pipeline")


def _log_latency(step: str, started: float) -> None:
    logger.info("step=%s latency_ms=%.1f", step, (time.perf_counter() - started) * 1000)


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class PipelineOptions:
    """Per-run knobs used by experiments without changing the recommendation API."""

    beta: float = 0.5
    rho: float = 0.85
    eta: float = 0.5
    candidate_count: int = 50
    rag_top_m: int = 5
    gamma1: float = 0.4
    gamma2: float = 0.3
    gamma3: float = 0.3
    mmr_mu: float = 0.8
    time_decay_half_life_days: float | None = 90.0
    use_composition: bool = True
    use_safety_filter: bool = True
    use_rag: bool = True
    use_mmr: bool = True


class RecommendationPipeline:
    """Owns read-only recommendation inputs and reuses product/RAG embeddings."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        products: pd.DataFrame,
        profiles: pd.DataFrame | Mapping[Any, Any] | None = None,
        *,
        llm_client: LLMClient | None = None,
        product_embedder: ProductEmbedder | None = None,
        rag_index: RAGIndex | None = None,
        item_embeddings: Mapping[Any, np.ndarray] | None = None,
        build_rag_index: bool = True,
    ) -> None:
        self.interactions = interactions.copy()
        self.products = products.copy()
        self.profiles = profiles
        self.llm_client = llm_client
        self.product_embedder = product_embedder or ProductEmbedder()
        self._item_embeddings = dict(item_embeddings) if item_embeddings is not None else None
        self.rag_index = rag_index
        if self.rag_index is None and build_rag_index:
            self.rag_index = RAGIndex(embedder=Embedder())
            self.rag_index.add_documents(self._rag_documents())

    def _rag_documents(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.products.to_dict(orient="records"):
            rows.append({
                "item_id": row.get("item_id", row.get("parent_asin")),
                "description": row.get("description", ""),
                "ingredients_text": row.get("ingredients_text", ""),
            })
        return rows

    def _profile(self, user_id: Any) -> tuple[Any, list[str], str]:
        profile: Any = None
        if isinstance(self.profiles, pd.DataFrame) and "user_id" in self.profiles:
            matches = self.profiles.loc[self.profiles.user_id == user_id]
            if not matches.empty:
                profile = matches.iloc[0].to_dict()
        elif isinstance(self.profiles, Mapping):
            profile = self.profiles.get(user_id)
        if isinstance(profile, AnimalProfile):
            allergens = list(profile.allergens)
            text = profile.to_embedding_text()
        elif isinstance(profile, Mapping):
            allergens = profile.get("allergens", profile.get("allergen_groups", [])) or []
            if isinstance(allergens, float) and np.isnan(allergens):
                allergens = []
            if isinstance(allergens, str):
                allergens = [allergens]
            if not isinstance(allergens, (list, tuple, set)):
                allergens = []
            text = " ".join(str(profile.get(key, "")) for key in
                             ("species", "pet_species", "age_group", "diet_restrictions", "lifestyle"))
            if allergens:
                text += " без " + ", ".join(map(str, allergens))
            preference_state = profile.get("preference_state") or {}
            if isinstance(preference_state, Mapping):
                preferred = preference_state.get("preferred_categories", [])
                disliked = preference_state.get("disliked_categories", [])
                if preferred:
                    text += " вподобані категорії " + ", ".join(map(str, preferred))
                if disliked:
                    text += " не пропонувати категорії " + ", ".join(map(str, disliked))
        else:
            allergens, text = [], "домашня тварина корм рекомендації"
        return profile, list(allergens), text.strip()

    def _product_embeddings(self) -> dict[Any, np.ndarray]:
        if self._item_embeddings is None:
            if self.rag_index is not None and self.rag_index.embedder.model is not None:
                self.product_embedder._model = self.rag_index.embedder.model
            model = self.product_embedder._get_model()
            if self.rag_index is not None:
                self.rag_index.embedder.model = model
            frame = self.products.copy()
            item_col = "item_id" if "item_id" in frame else "parent_asin"
            if item_col != "parent_asin":
                frame = frame.rename(columns={item_col: "parent_asin"})
            self._item_embeddings = self.product_embedder.embed_products(frame)
        return self._item_embeddings

    def recommend(
        self,
        user_id: Any,
        use_llm: bool = True,
        *,
        options: PipelineOptions | None = None,
    ) -> list[RecommendationItem]:
        """Return up to ten safe personalized recommendations for one user."""
        options = options or PipelineOptions()
        if self.products.empty:
            return []
        products = self.products.copy()
        item_col = "item_id" if "item_id" in products else "parent_asin"
        if item_col != "item_id":
            products = products.rename(columns={item_col: "item_id"})
        if "parent_asin" in self.interactions:
            interactions = self.interactions.rename(columns={"parent_asin": "item_id"})
        else:
            interactions = self.interactions.copy()
        if "timestamp" not in interactions and "timestamp" in self.interactions:
            interactions["timestamp"] = self.interactions.timestamp

        profile, allergens, profile_text = self._profile(user_id)
        species = None
        if isinstance(profile, AnimalProfile):
            species = profile.species
        elif isinstance(profile, Mapping):
            species = profile.get("species", profile.get("pet_species"))
            if species in {"cats", "feline"}:
                species = "cat"
            elif species in {"dogs", "canine"}:
                species = "dog"
        # Never mix catalog rows for other species into a pet's recommendation pool.
        if species in {"cat", "dog"} and "species" in products:
            products = products.loc[products["species"].fillna("").str.lower() == species].copy()
        if products.empty:
            return []
        # Sterilized pets should receive food explicitly labeled for that need.
        if isinstance(profile, Mapping):
            pet_needs = " ".join(str(profile.get(key, "")) for key in ("lifestyle", "diet_restrictions")).lower()
            sterilized_need = any(token in pet_needs for token in (
                "steriliz", "neuter", "\u0441\u0442\u0435\u0440\u0438\u043b",
                "\u0441\u043f\u0435\u0440", "\u043a\u0430\u0441\u0442\u0440",
            ))
            if sterilized_need:
                label_columns = [column for column in ("title", "categories", "description") if column in products]
                labels = products[label_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                need_terms = (
                    "steriliz", "neuter", "spay", "spayed", "\u043a\u0430\u0441\u0442\u0440\u0430\u0446",
                    "\u0441\u0442\u0435\u0440\u0438\u043b", "\u043f\u0456\u0441\u043b\u044f \u0441\u0442\u0435\u0440",
                )
                suitable = labels.apply(lambda text: any(term in text for term in need_terms))
                products = products.loc[suitable].copy()
        if products.empty:
            return []
        stage_started = time.perf_counter()
        content_embeddings = self._product_embeddings()
        user_vector = build_user_preference_vector(
            self.interactions, content_embeddings, profile=profile_text,
            embedder=self.product_embedder,
            half_life_days=options.time_decay_half_life_days or 1e15,
        )
        _log_latency("profile_and_embeddings", stage_started)

        composition_by_id: dict[Any, dict[str, Any]] = {}
        safe_by_id: dict[Any, bool] = {}
        stage_started = time.perf_counter()
        for row in products.to_dict(orient="records"):
            item_id = row["item_id"]
            if options.use_composition:
                composition = composition_score(
                    str(row.get("ingredients_text") or ""), allergens=allergens,
                    nutrients={"protein": _number_or_none(row.get("proteins_100g")),
                               "fat": _number_or_none(row.get("fat_100g"))},
                    moisture=_number_or_none(row.get("moisture_100g")), species=species,
                    eta=options.eta, decay=options.rho,
                )
            else:
                # Ingredient-based hard safety remains active even in composition ablations.
                safety_only = composition_score(
                    str(row.get("ingredients_text") or ""), allergens=allergens,
                    nutrients=None, eta=0.0, decay=options.rho,
                )
                composition = {**safety_only, "S_comp": 0.0}
            composition_by_id[item_id] = composition
            safe_by_id[item_id] = bool(composition["safe"])
        _log_latency("composition_and_safety", stage_started)

        stage_started = time.perf_counter()
        candidates = generate_candidates(
            user_id, interactions, products, content_embeddings, user_vector,
            safe={item_id: True for item_id in safe_by_id},
            beta=options.beta, top_n=options.candidate_count,
            time_decay_half_life_days=options.time_decay_half_life_days,
        )
        _log_latency("candidate_generation", stage_started)
        if candidates.empty:
            return []
        candidates = candidates.merge(products, on="item_id", how="left", suffixes=("", "_product"))
        candidates["safe"] = candidates.item_id.map(safe_by_id).fillna(False).astype(bool)
        candidates["S_comp"] = candidates.item_id.map(
            lambda item: composition_by_id[item]["S_comp"]
        )
        candidates["ingredients"] = candidates.item_id.map(
            lambda item: composition_by_id[item]["ingredients"]
        )

        item_context: dict[Any, str] = {}
        if options.use_rag and self.rag_index is not None:
            stage_started = time.perf_counter()
            if self.rag_index.embedder.model is None:
                self.rag_index.embedder.model = self.product_embedder._get_model()
            retrieved = self.rag_index.search(profile_text, top_m=options.rag_top_m)
            for passage in retrieved:
                item_id = passage.metadata.get("item_id")
                if item_id is not None:
                    item_context[item_id] = (item_context.get(item_id, "") + "\n" + passage.text).strip()
            _log_latency("rag_retrieval", stage_started)

        enabled_llm = self.llm_client if use_llm else None
        mmr_mu = options.mmr_mu if options.use_mmr else 1.0
        stage_started = time.perf_counter()
        recommendations = rerank_recommendations(
            candidates, enabled_llm, context=profile_text, item_context=item_context,
            item_embeddings=content_embeddings, gamma1=options.gamma1,
            gamma2=options.gamma2, gamma3=options.gamma3, mu=mmr_mu, k=10,
            filter_unsafe=options.use_safety_filter,
        )
        _log_latency("llm_rerank_and_mmr", stage_started)
        return recommendations


_default_pipeline: RecommendationPipeline | None = None


def configure_pipeline(
    interactions: pd.DataFrame,
    products: pd.DataFrame,
    profiles: pd.DataFrame | Mapping[Any, Any] | None = None,
    **kwargs: Any,
) -> RecommendationPipeline:
    """Configure the process-local pipeline used by the API convenience function."""
    global _default_pipeline
    _default_pipeline = RecommendationPipeline(interactions, products, profiles, **kwargs)
    return _default_pipeline


def recommend(user_id: Any, use_llm: bool = True) -> list[RecommendationItem]:
    """API entry point: return ten (or fewer, if unavailable) explained items."""
    if _default_pipeline is None:
        raise RuntimeError("Call configure_pipeline() with read-only data before recommend().")
    return _default_pipeline.recommend(user_id, use_llm=use_llm)
