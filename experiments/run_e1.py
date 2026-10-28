"""Experiment E1: proposed method vs. baselines on Amazon Reviews 2023 Pet Supplies (NDCG@10 etc.).

Uses data/processed/*.parquet from prepare_amazon.py. The recommendation pipeline is the
unchanged petrec core; two pure functions are swapped for numerically identical fast
versions so that thousands of users finish in reasonable time:

* candidate generation — the core rebuilds a dense item×item cosine matrix for every user;
  here the same similarities are computed from one sparse matrix (``--self-check`` verifies
  identical output against the core function on synthetic data);
* composition scoring — memoised, because it depends only on the product and the profile.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import petrec.pipeline as core_pipeline  # noqa: E402
from petrec.baselines import BPRRecommender, ContentBased, ImplicitALS, ItemBasedCF, Popularity  # noqa: E402
from petrec.candidates import generate_candidates as core_generate_candidates  # noqa: E402
from petrec.evaluate import evaluate  # noqa: E402
from petrec.pipeline import PipelineOptions, RecommendationPipeline  # noqa: E402
from petrec.profiles import ProductEmbedder  # noqa: E402

METRICS = ("Precision@K", "Recall@K", "NDCG@K", "HR@K", "Safety@K", "Coverage", "Diversity")


class FastCandidates:
    """Same result as petrec.candidates.generate_candidates for a fixed interaction table."""

    def __init__(self) -> None:
        self._key: int | None = None

    def _prepare(self, interactions: pd.DataFrame) -> None:
        column = "item_id" if "item_id" in interactions else "parent_asin"
        users = pd.Index(interactions.user_id.unique())
        items = pd.Index(interactions[column].unique())
        pairs = interactions[["user_id", column]].drop_duplicates()
        matrix = csr_matrix((np.ones(len(pairs)), (users.get_indexer(pairs.user_id), items.get_indexer(pairs[column]))),
                            shape=(len(users), len(items)))
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        self.matrix_t = matrix.T.tocsr()
        self.norms = norms
        self.positions = {item: index for index, item in enumerate(items)}
        self.column = column
        self.histories = {user: group for user, group in interactions.groupby("user_id", sort=False)}
        self._key = id(interactions)

    def _cf(self, candidate_ids: list[Any], weights: dict[Any, float]) -> np.ndarray:
        seen = [(self.positions[item], weight) for item, weight in weights.items() if item in self.positions]
        denominator = sum(weight for _, weight in seen)
        if not seen or denominator <= 0:
            return np.zeros(len(candidate_ids))
        seen_pos = np.array([position for position, _ in seen])
        seen_w = np.array([weight for _, weight in seen])
        co = (self.matrix_t @ self.matrix_t[seen_pos].T).toarray()  # items × seen co-occurrence counts
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = co / np.outer(self.norms, self.norms[seen_pos])
        similarity = np.nan_to_num(similarity)
        scores_all = similarity @ seen_w / denominator
        return np.array([scores_all[self.positions[item]] if item in self.positions else 0.0 for item in candidate_ids])

    def __call__(self, user_id, interactions, candidate_items, item_embeddings, user_embedding, *, safe=None,
                 beta=0.5, top_n=50, time_decay_half_life_days=None, **unused) -> pd.DataFrame:
        if unused.get("purchases") is not None:
            raise NotImplementedError("purchases are not used by the pipeline")
        if self._key != id(interactions):
            self._prepare(interactions)
        history = self.histories.get(user_id, interactions.iloc[0:0])
        candidate_ids = (candidate_items["item_id" if "item_id" in candidate_items else "parent_asin"].tolist()
                         if isinstance(candidate_items, pd.DataFrame) else list(candidate_items))
        candidate_ids = list(dict.fromkeys(candidate_ids))
        candidate_ids = [item for item in candidate_ids
                         if bool(safe(item) if callable(safe) else safe.get(item, False) if safe is not None else False)]
        reference = float(pd.to_numeric(history.timestamp, errors="coerce").max()) if "timestamp" in history and not history.empty else None
        weights: dict[Any, float] = {}
        for row in history.to_dict("records"):  # same accumulation order and decay as the core
            value = row.get("weight", row.get("rating", 1.0))
            try:
                weights[row[self.column]] = weights.get(row[self.column], 0.0) + max(0.0, float(value))
            except (TypeError, ValueError):
                weights[row[self.column]] = weights.get(row[self.column], 0.0) + 1.0
            if time_decay_half_life_days is not None and reference is not None and row.get("timestamp") is not None:
                age = max(0.0, reference - float(row["timestamp"]))
                weights[row[self.column]] *= 0.5 ** (age / time_decay_half_life_days)
        effective_beta = beta if len(set(history[self.column])) >= 3 else 0.0
        if not candidate_ids:
            return pd.DataFrame(columns=["item_id", "s_cf", "s_content", "s_cand", "safe"])
        user = np.asarray(user_embedding, dtype=float).ravel()
        user_norm = np.linalg.norm(user)
        content = np.zeros(len(candidate_ids))
        for index, item in enumerate(candidate_ids):
            vector = item_embeddings.get(item)
            if vector is not None:
                vector = np.asarray(vector, dtype=float).ravel()
                denominator = np.linalg.norm(vector) * user_norm
                if vector.shape == user.shape and denominator > 0:
                    content[index] = float(vector @ user / denominator)
        cf = self._cf(candidate_ids, weights)
        minmax = core_pipeline_minmax
        result = pd.DataFrame({"item_id": candidate_ids, "s_cf": cf, "s_content": content, "safe": True})
        result["s_cand"] = effective_beta * minmax(cf) + (1.0 - effective_beta) * minmax(content)
        result = result.sort_values(["s_cand", "item_id"], ascending=[False, True],
                                    key=lambda col: col.map(str) if col.name == "item_id" else col, kind="stable")
        return result.head(top_n).reset_index(drop=True)[["item_id", "s_cf", "s_content", "s_cand", "safe"]]


from petrec.candidates import _minmax as core_pipeline_minmax  # noqa: E402


def _cached_composition():
    original = core_pipeline.composition_score

    @functools.lru_cache(maxsize=2_000_000)
    def cached(text, allergens, nutrients, moisture, species, eta, decay):
        return original(text, allergens=list(allergens), nutrients=dict(nutrients) if nutrients is not None else None,
                        moisture=moisture, species=species, eta=eta, decay=decay)

    def wrapper(text, *, allergens=(), nutrients=None, moisture=None, species=None, eta=0.5, decay=0.85):
        key_nutrients = tuple(sorted(nutrients.items())) if nutrients is not None else None
        return cached(text, tuple(allergens), key_nutrients, moisture, species, eta, decay)

    return wrapper


def self_check() -> None:
    from petrec.data_prep import generate_synthetic_data
    reviews, _, _ = generate_synthetic_data(n_users=30, n_items=25, interactions_per_user=7)
    interactions = reviews.rename(columns={"parent_asin": "item_id"})
    rng = np.random.default_rng(0)
    items = interactions.item_id.unique().tolist() + ["unseen_item"]
    embeddings = {item: rng.normal(size=8) for item in items[:-3]}
    fast = FastCandidates()
    for user in interactions.user_id.unique()[:10]:
        vector = rng.normal(size=8)
        for half_life in (None, 30.0):
            args = (user, interactions, items, embeddings, vector)
            kwargs = dict(safe={item: True for item in items}, beta=0.5, top_n=12, time_decay_half_life_days=half_life)
            expected, actual = core_generate_candidates(*args, **kwargs), fast(*args, **kwargs)
            assert expected.item_id.tolist() == actual.item_id.tolist(), (user, half_life)
            assert np.allclose(expected[["s_cf", "s_content", "s_cand"]], actual[["s_cf", "s_content", "s_cand"]])
    print("self-check: fast candidate generation matches the core function")


def _pipeline_recs(pipeline: RecommendationPipeline, users: list[Any], options: PipelineOptions | None,
                   use_llm: bool) -> pd.DataFrame:
    rows = []
    for number, user in enumerate(users, 1):
        for rank, item in enumerate(pipeline.recommend(user, use_llm=use_llm, options=options), 1):
            rows.append({"user_id": user, "parent_asin": item.item_id, "score": item.match_percent / 100, "rank": rank,
                         "is_safe": item.safe_badge, "llm_used": item.llm_used})
        if number % 250 == 0:
            print(f"  {number}/{len(users)} users", flush=True)
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank", "is_safe", "llm_used"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--users", type=int, default=2000, help="evaluated test users (fixed random sample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return

    data = Path(args.data)
    train, test = pd.read_parquet(data / "train.parquet"), pd.read_parquet(data / "test.parquet")
    products, profiles = pd.read_parquet(data / "products.parquet"), pd.read_parquet(data / "profiles.parquet")
    products = products.loc[products.parent_asin.isin(set(train.parent_asin) | set(test.parent_asin))].reset_index(drop=True)
    # Parquet returns list columns as numpy arrays; the core expects plain lists.
    products["categories"] = products.categories.map(lambda value: list(value) if value is not None else [])
    profiles["allergen_groups"] = profiles.allergen_groups.map(lambda value: list(value) if value is not None else [])
    users = sorted(set(test.user_id) & set(train.user_id), key=str)
    users = sorted(np.random.default_rng(args.seed).choice(users, size=min(args.users, len(users)), replace=False).tolist(), key=str)
    test_users = test.loc[test.user_id.isin(users)].merge(profiles[["user_id", "allergen_groups"]], on="user_id", how="left")
    print(f"train={len(train):,} test users={len(users):,} products={len(products):,}", flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def measure(name: str, recs: pd.DataFrame) -> dict[str, Any]:
        recs = recs.merge(products[["parent_asin", "ingredients_text", "categories"]], on="parent_asin", how="left")
        result = evaluate(recs, test_users, k=10, output_path=output / "_scratch.csv").iloc[0]
        row = {"Модель": name, **{key: float(result[key]) for key in METRICS}}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    rows = []
    baseline_cache = output / f"E1_baselines_{args.users}_{args.seed}.json"  # same users and data -> same metrics
    cached = json.loads(baseline_cache.read_text(encoding="utf-8")) if baseline_cache.exists() else {}
    for name, model in (("Popularity", Popularity()), ("ItemKNN", ItemBasedCF()), ("ALS", ImplicitALS()),
                        ("BPR", BPRRecommender()), ("Content (TF-IDF)", ContentBased())):
        if name in cached:
            rows.append(cached[name])
            continue
        started = time.perf_counter()
        fitted = model.fit(train, products) if isinstance(model, ContentBased) else model.fit(train)
        rows.append(measure(name, fitted.recommend(users, k=10)))
        cached[name] = rows[-1]
        baseline_cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {name}: {time.perf_counter() - started:.0f}s", flush=True)

    core_pipeline.generate_candidates = FastCandidates()
    core_pipeline.composition_score = _cached_composition()
    started = time.perf_counter()
    # RAG passages only enter the LLM prompt (petrec.rerank), so without the LLM the index does
    # not change any score; skipping it saves hours of CPU encoding. Product vectors are cached.
    embedder = ProductEmbedder(cache_path=data / "product_embeddings.npy")
    pipeline = RecommendationPipeline(train, products, profiles, llm_client=None, product_embedder=embedder,
                                      build_rag_index=False)
    pipeline._product_embeddings()
    print(f"  embeddings and RAG index: {time.perf_counter() - started:.0f}s", flush=True)
    for name, options in (("Запропонований метод (без LLM-переранжування)", None),
                          ("Без складу", PipelineOptions(use_composition=False)),
                          ("Без часового згасання", PipelineOptions(time_decay_half_life_days=None)),
                          ("Без MMR", PipelineOptions(use_mmr=False))):
        started = time.perf_counter()
        rows.append(measure(name, _pipeline_recs(pipeline, users, options, use_llm=False)))
        print(f"  {name}: {time.perf_counter() - started:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "E1_amazon_pet_supplies.csv", index=False, encoding="utf-8-sig")
    (output / "_scratch.csv").unlink(missing_ok=True)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
