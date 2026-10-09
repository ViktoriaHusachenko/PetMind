"""Animal profiles and embedding-based user preference vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

MODEL_NAME = "intfloat/multilingual-e5-small"


class AnimalProfile(BaseModel):
    """Structured pet profile used for cold-start personalization and safety."""

    species: str
    age_group: str
    weight_kg: float | None = Field(default=None, gt=0)
    allergens: list[str] = Field(default_factory=list)
    diet_restrictions: list[str] = Field(default_factory=list)
    lifestyle: list[str] = Field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Render populated profile attributes as a compact natural-language query."""
        details = [self.age_group, self.species]
        if self.weight_kg is not None:
            details.append(f"weight {self.weight_kg:g} kg")
        if self.lifestyle:
            details.extend(self.lifestyle)
        if self.diet_restrictions:
            details.extend(self.diet_restrictions)
        if self.allergens:
            details.append("without " + ", ".join(self.allergens))
        return " ".join(str(detail).strip() for detail in details if str(detail).strip())


class ProductEmbedder:
    """Create E5 text embeddings and cache product vectors in a NumPy file."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        """Configure the E5 model, optional cache path, and optional injected encoder."""
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._model = model

    def _get_model(self) -> Any:
        """Load SentenceTransformer only when an embedding is first requested."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str, *, query: bool = False) -> np.ndarray:
        """Encode text with the E5 query or passage prefix and return a 1-D vector."""
        prefix = "query: " if query else "passage: "
        encoded = self._get_model().encode(
            [prefix + text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vector = np.asarray(encoded, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("The embedding model returned an empty or non-finite vector")
        return vector

    def embed_products(self, products: pd.DataFrame) -> dict[Any, np.ndarray]:
        """Embed product documents and load/save vectors from an optional ``.npy`` cache."""
        required = {"parent_asin"}
        missing = required.difference(products.columns)
        if missing:
            raise ValueError(f"Product data is missing columns: {sorted(missing)}")

        product_rows: list[tuple[Any, str, str]] = []
        seen_ids: set[Any] = set()
        for row in products.to_dict(orient="records"):
            item_id = row["parent_asin"]
            if pd.isna(item_id) or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            content = " ".join(
                value
                for value in (
                    _as_text(row.get("title")),
                    _as_text(row.get("categories")),
                    _as_text(row.get("description")),
                    _as_text(row.get("ingredients_text")),
                )
                if value
            )
            content = content or str(item_id)
            fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
            product_rows.append((item_id, content, fingerprint))

        item_ids = [row[0] for row in product_rows]
        documents = [row[1] for row in product_rows]
        fingerprints = [row[2] for row in product_rows]
        cached = self._read_cache(item_ids, fingerprints)
        if cached is not None:
            return {item_id: cached[index] for index, item_id in enumerate(item_ids)}

        if not documents:
            return {}
        encoded = self._get_model().encode(
            ["passage: " + document for document in documents],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(item_ids):
            raise ValueError("The embedding model returned an unexpected product matrix shape")
        if not np.isfinite(vectors).all():
            raise ValueError("The embedding model returned non-finite product vectors")
        self._write_cache(item_ids, fingerprints, vectors)
        return {item_id: vectors[index] for index, item_id in enumerate(item_ids)}

    def _read_cache(
        self,
        item_ids: list[Any],
        fingerprints: list[str],
    ) -> np.ndarray | None:
        """Read a cache only when its item order and document fingerprints match."""
        if self.cache_path is None:
            return None
        manifest_path = self.cache_path.with_suffix(".json")
        if not self.cache_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_ids = [str(item_id) for item_id in item_ids]
            if manifest.get("model_name") != self.model_name:
                return None
            if manifest.get("item_ids") != expected_ids or manifest.get("fingerprints") != fingerprints:
                return None
            vectors = np.load(self.cache_path, allow_pickle=False)
            if vectors.ndim != 2 or vectors.shape[0] != len(item_ids):
                return None
            return vectors
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _write_cache(
        self,
        item_ids: list[Any],
        fingerprints: list[str],
        vectors: np.ndarray,
    ) -> None:
        """Persist vectors and a sidecar manifest that detects stale cache entries."""
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, vectors, allow_pickle=False)
        manifest = {
            "model_name": self.model_name,
            "item_ids": [str(item_id) for item_id in item_ids],
            "fingerprints": fingerprints,
        }
        self.cache_path.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )


def compute_interaction_weights(
    history: pd.DataFrame,
    half_life_days: float = 90.0,
    reference_time: float | None = None,
) -> np.ndarray:
    """Compute ``rating/5 * exp(-lambda * age_days)`` weights for user history."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if "timestamp" not in history:
        raise ValueError("history must contain timestamp")
    if history.empty:
        return np.empty(0, dtype=np.float64)

    timestamps = pd.to_numeric(history["timestamp"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(timestamps).all():
        raise ValueError("history timestamps must be finite numeric day values")
    t0 = float(np.max(timestamps)) if reference_time is None else float(reference_time)
    if not np.isfinite(t0) or np.any(timestamps > t0):
        raise ValueError("reference_time must be finite and no earlier than every interaction")

    if "rating" in history:
        ratings = pd.to_numeric(history["rating"], errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(ratings), ratings / 5.0, 1.0)
    else:
        values = np.ones(len(history), dtype=np.float64)
    decay_rate = np.log(2.0) / half_life_days
    return values * np.exp(-decay_rate * (t0 - timestamps))


def build_user_preference_vector(
    history: pd.DataFrame,
    item_embeddings: dict[Any, np.ndarray],
    profile: AnimalProfile | str | None = None,
    embedder: ProductEmbedder | None = None,
    half_life_days: float = 90.0,
    reference_time: float | None = None,
) -> np.ndarray:
    """Build a weighted item-history vector, or a query embedding for cold start."""
    if not history.empty and "parent_asin" not in history:
        raise ValueError("history must contain parent_asin")
    usable_rows = [row for row in history.to_dict(orient="records") if row["parent_asin"] in item_embeddings]
    if usable_rows:
        usable_history = pd.DataFrame(usable_rows)
        weights = compute_interaction_weights(
            usable_history,
            half_life_days=half_life_days,
            reference_time=reference_time,
        )
        vectors = np.stack([np.asarray(item_embeddings[row["parent_asin"]], dtype=np.float64) for row in usable_rows])
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Item embeddings must be finite one-dimensional vectors")
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Interaction weights must have a positive finite sum")
        if np.any(weights < 0):
            raise ValueError("Interaction weights must not be negative")
        return np.average(vectors, axis=0, weights=weights)

    if profile is None:
        raise ValueError("A profile description is required when no item history is available")
    if embedder is None:
        embedder = ProductEmbedder()
    query = profile.to_embedding_text() if isinstance(profile, AnimalProfile) else profile.strip()
    if not query:
        raise ValueError("Cold-start profile description must not be empty")
    return embedder.embed_text(query, query=True)


def _as_text(value: Any) -> str:
    """Convert scalar or nested product metadata into readable document text."""
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        return " ".join(_as_text(part) for part in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_as_text(part) for part in value)
    return str(value).strip()