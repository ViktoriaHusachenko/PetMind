"""Non-LLM recommendation baselines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Popularity:
    """Recommend the most frequently interacted-with items, excluding seen items."""

    def __init__(self) -> None:
        """Initialize an unfitted popularity recommender."""
        self.item_popularity: pd.Series = pd.Series(dtype="float64")
        self.user_items: dict[Any, set[Any]] = {}

    def fit(self, interactions: pd.DataFrame) -> Popularity:
        """Fit item popularity and user histories from interaction rows."""
        _validate_interactions(interactions)
        self.item_popularity = interactions.groupby("parent_asin").size().sort_values(
            ascending=False, kind="stable"
        )
        self.user_items = interactions.groupby("user_id")["parent_asin"].agg(set).to_dict()
        return self

    def recommend(self, users: Iterable[Any], k: int = 10) -> pd.DataFrame:
        """Return up to ``k`` unseen popular items per user as ranked rows."""
        _validate_k(k)
        rows: list[dict[str, Any]] = []
        for user_id in users:
            seen = self.user_items.get(user_id, set())
            rank = 0
            for item_id, score in self.item_popularity.items():
                if item_id in seen:
                    continue
                rank += 1
                rows.append(
                    {"user_id": user_id, "parent_asin": item_id, "score": float(score), "rank": rank}
                )
                if rank == k:
                    break
        return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank"])


class ItemBasedCF:
    """Recommend items using item-item cosine similarity from a CSR matrix."""

    def __init__(self) -> None:
        """Initialize an unfitted item-based collaborative filter."""
        self.user_ids: list[Any] = []
        self.item_ids: list[Any] = []
        self.user_index: dict[Any, int] = {}
        self.item_index: dict[Any, int] = {}
        self.user_item_matrix: csr_matrix | None = None
        self.item_similarity: csr_matrix | None = None

    def fit(self, interactions: pd.DataFrame) -> ItemBasedCF:
        """Build a sparse user-item matrix and sparse item cosine similarities."""
        _validate_interactions(interactions)
        self.user_ids = sorted(interactions["user_id"].unique().tolist(), key=str)
        self.item_ids = sorted(interactions["parent_asin"].unique().tolist(), key=str)
        self.user_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}

        values = pd.to_numeric(interactions.get("rating", 1), errors="coerce")
        if not isinstance(values, pd.Series):
            values = pd.Series(1.0, index=interactions.index)
        values = values.fillna(1.0).astype(float)
        matrix = csr_matrix(
            (
                values.to_numpy(),
                (
                    interactions["user_id"].map(self.user_index).to_numpy(),
                    interactions["parent_asin"].map(self.item_index).to_numpy(),
                ),
            ),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=float,
        )
        matrix.sum_duplicates()
        self.user_item_matrix = matrix
        self.item_similarity = cosine_similarity(matrix.T, dense_output=False).tocsr()
        return self

    def recommend(self, users: Iterable[Any], k: int = 10) -> pd.DataFrame:
        """Rank unseen items by the user's weighted item-neighbor scores."""
        _validate_k(k)
        if self.user_item_matrix is None or self.item_similarity is None:
            raise RuntimeError("Call fit() before recommend().")

        rows: list[dict[str, Any]] = []
        for user_id in users:
            user_position = self.user_index.get(user_id)
            if user_position is None:
                continue
            user_vector = self.user_item_matrix.getrow(user_position)
            scores = (user_vector @ self.item_similarity).toarray().ravel()
            seen_positions = set(user_vector.indices.tolist())
            candidate_positions = [
                position
                for position in range(len(self.item_ids))
                if position not in seen_positions and scores[position] > 0
            ]
            candidate_positions.sort(key=lambda position: (-scores[position], str(self.item_ids[position])))
            for rank, position in enumerate(candidate_positions[:k], start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        "parent_asin": self.item_ids[position],
                        "score": float(scores[position]),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank"])


class ContentBased:
    """TF-IDF content baseline using each user's previously interacted products."""

    def __init__(self) -> None:
        self.item_ids: list[Any] = []
        self.item_matrix = None
        self.user_items: dict[Any, set[Any]] = {}
        self.user_profiles = {}

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame) -> ContentBased:
        _validate_interactions(interactions)
        if "parent_asin" not in products:
            raise ValueError("products must contain parent_asin")
        self.item_ids = products.parent_asin.tolist()
        text_columns = [column for column in ("title", "brand", "category", "description", "ingredients_text") if column in products]
        text = products[text_columns].fillna("").astype(str).agg(" ".join, axis=1) if text_columns else pd.Series([""] * len(products))
        if any(value.strip() for value in text):
            self.item_matrix = TfidfVectorizer(max_features=50000, ngram_range=(1, 2)).fit_transform(text)
        else:
            self.item_matrix = csr_matrix((len(products), 1), dtype=np.float32)
        item_index = {item: position for position, item in enumerate(self.item_ids)}
        self.user_items = interactions.groupby("user_id").parent_asin.agg(set).to_dict()
        self.user_profiles = {}
        for user_id, group in interactions.groupby("user_id"):
            indices = [item_index[item] for item in group.parent_asin if item in item_index]
            if indices:
                self.user_profiles[user_id] = np.asarray(self.item_matrix[indices].mean(axis=0)).ravel()
        return self

    def recommend(self, users: Iterable[Any], k: int = 10) -> pd.DataFrame:
        _validate_k(k)
        if self.item_matrix is None:
            raise RuntimeError("Call fit() before recommend().")
        rows = []
        for user_id in users:
            profile = self.user_profiles.get(user_id)
            if profile is None:
                continue
            scores = np.asarray(self.item_matrix @ profile).ravel()
            seen = self.user_items.get(user_id, set())
            ranked = sorted(((pos, score) for pos, score in enumerate(scores) if self.item_ids[pos] not in seen),
                            key=lambda value: (-value[1], str(self.item_ids[value[0]])))[:k]
            rows.extend({"user_id": user_id, "parent_asin": self.item_ids[pos], "score": float(score), "rank": rank}
                        for rank, (pos, score) in enumerate(ranked, 1))
        return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank"])


class ImplicitALS:
    """Small deterministic implicit-feedback ALS baseline."""

    def __init__(self, factors: int = 32, regularization: float = 0.1, alpha: float = 20.0,
                 iterations: int = 8, seed: int = 17) -> None:
        self.factors, self.regularization, self.alpha = factors, regularization, alpha
        self.iterations, self.seed = iterations, seed
        self.user_ids: list[Any] = []
        self.item_ids: list[Any] = []
        self.matrix: csr_matrix | None = None
        self.user_factors = self.item_factors = None

    def fit(self, interactions: pd.DataFrame) -> ImplicitALS:
        _validate_interactions(interactions)
        self.user_ids = sorted(interactions.user_id.unique().tolist(), key=str)
        self.item_ids = sorted(interactions.parent_asin.unique().tolist(), key=str)
        users = {value: index for index, value in enumerate(self.user_ids)}
        items = {value: index for index, value in enumerate(self.item_ids)}
        self.matrix = csr_matrix((np.ones(len(interactions)),
            (interactions.user_id.map(users), interactions.parent_asin.map(items))),
            shape=(len(users), len(items)), dtype=np.float32)
        self.matrix.sum_duplicates()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (len(users), self.factors))
        self.item_factors = rng.normal(0, 0.01, (len(items), self.factors))
        identity = np.eye(self.factors) * self.regularization
        for _ in range(self.iterations):
            self.user_factors = self._solve(self.matrix, self.item_factors, identity)
            self.item_factors = self._solve(self.matrix.T.tocsr(), self.user_factors, identity)
        return self

    def _solve(self, rows: csr_matrix, fixed: np.ndarray, identity: np.ndarray) -> np.ndarray:
        gram = fixed.T @ fixed
        result = np.zeros((rows.shape[0], self.factors), dtype=np.float32)
        for index in range(rows.shape[0]):
            positive = rows.indices[rows.indptr[index]:rows.indptr[index + 1]]
            if not len(positive):
                continue
            factors = fixed[positive]
            a = gram + self.alpha * (factors.T @ factors) + identity
            b = (1 + self.alpha) * factors.sum(axis=0)
            result[index] = np.linalg.solve(a, b)
        return result

    def recommend(self, users: Iterable[Any], k: int = 10) -> pd.DataFrame:
        _validate_k(k)
        if self.matrix is None:
            raise RuntimeError("Call fit() before recommend().")
        user_index = {value: index for index, value in enumerate(self.user_ids)}
        rows = []
        for user_id in users:
            position = user_index.get(user_id)
            if position is None:
                continue
            scores = self.item_factors @ self.user_factors[position]
            seen = set(self.matrix.getrow(position).indices.tolist())
            ranked = sorted(((idx, score) for idx, score in enumerate(scores) if idx not in seen),
                            key=lambda value: (-value[1], str(self.item_ids[value[0]])))[:k]
            rows.extend({"user_id": user_id, "parent_asin": self.item_ids[idx], "score": float(score), "rank": rank}
                        for rank, (idx, score) in enumerate(ranked, 1))
        return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank"])


class BPRRecommender:
    """Bayesian Personalized Ranking trained with deterministic pairwise SGD."""

    def __init__(self, factors: int = 32, learning_rate: float = 0.03, regularization: float = 0.01,
                 epochs: int = 8, seed: int = 23) -> None:
        self.factors, self.learning_rate, self.regularization = factors, learning_rate, regularization
        self.epochs, self.seed = epochs, seed
        self.user_ids: list[Any] = []
        self.item_ids: list[Any] = []
        self.user_factors = self.item_factors = None
        self.user_items: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> BPRRecommender:
        _validate_interactions(interactions)
        self.user_ids = sorted(interactions.user_id.unique().tolist(), key=str)
        self.item_ids = sorted(interactions.parent_asin.unique().tolist(), key=str)
        users = {value: index for index, value in enumerate(self.user_ids)}
        items = {value: index for index, value in enumerate(self.item_ids)}
        pairs = [(users[user], items[item]) for user, item in interactions[["user_id", "parent_asin"]].itertuples(index=False, name=None)]
        self.user_items = {}
        for user, item in pairs:
            self.user_items.setdefault(user, set()).add(item)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (len(users), self.factors))
        self.item_factors = rng.normal(0, 0.1, (len(items), self.factors))
        trainable = [(user, item) for user, item in pairs if len(self.user_items[user]) < len(items)]
        if not trainable:
            return self
        for _ in range(self.epochs * max(1, len(trainable))):
            user, positive = trainable[int(rng.integers(len(trainable)))]
            negative = int(rng.integers(len(items)))
            while negative in self.user_items[user]:
                negative = int(rng.integers(len(items)))
            u = self.user_factors[user].copy()
            p = self.item_factors[positive].copy()
            n = self.item_factors[negative].copy()
            difference = float(u @ (p - n))
            gradient = 1.0 / (1.0 + np.exp(np.clip(difference, -30, 30)))
            self.user_factors[user] += self.learning_rate * (gradient * (p - n) - self.regularization * u)
            self.item_factors[positive] += self.learning_rate * (gradient * u - self.regularization * p)
            self.item_factors[negative] += self.learning_rate * (-gradient * u - self.regularization * n)
        return self

    def recommend(self, users: Iterable[Any], k: int = 10) -> pd.DataFrame:
        _validate_k(k)
        if self.user_factors is None:
            raise RuntimeError("Call fit() before recommend().")
        user_index = {value: index for index, value in enumerate(self.user_ids)}
        rows = []
        for user_id in users:
            position = user_index.get(user_id)
            if position is None:
                continue
            scores = self.item_factors @ self.user_factors[position]
            seen = self.user_items.get(position, set())
            ranked = sorted(((idx, score) for idx, score in enumerate(scores) if idx not in seen),
                            key=lambda value: (-value[1], str(self.item_ids[value[0]])))[:k]
            rows.extend({"user_id": user_id, "parent_asin": self.item_ids[idx], "score": float(score), "rank": rank}
                        for rank, (idx, score) in enumerate(ranked, 1))
        return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank"])


def _validate_interactions(interactions: pd.DataFrame) -> None:
    """Validate the minimum interaction-table schema."""
    required = {"user_id", "parent_asin"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing interaction columns: {sorted(missing)}")
    if interactions[["user_id", "parent_asin"]].isna().any().any():
        raise ValueError("user_id and parent_asin must not contain missing values")


def _validate_k(k: int) -> None:
    """Reject nonpositive recommendation cutoffs."""
    if k < 1:
        raise ValueError("k must be positive")
