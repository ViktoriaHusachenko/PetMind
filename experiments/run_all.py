"""Run recommendation experiments E1–E5 from prepared Parquet data.

The E3 parameter grid must be supplied as a CSV exported from thesis table 2.4;
this repository does not contain that table, so the runner does not invent values.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from petrec.baselines import BPRRecommender, ContentBased, ImplicitALS, ItemBasedCF, Popularity
from petrec.evaluate import evaluate
from petrec.llm import AnthropicClient, OllamaClient, OpenAIClient
from petrec.pipeline import PipelineOptions, RecommendationPipeline
from petrec.reminders import estimate_food_stock_days, smoothed_purchase_interval


GRID_COLUMNS = {
    "T_half": ("T_half", "T½", "t_half", "half_life_days"),
    "rho": ("rho", "ρ"),
    "eta": ("eta", "η"),
    "beta": ("beta", "β"),
    "N": ("N", "top_n", "candidate_count"),
    "M": ("M", "top_m", "rag_top_m"),
    "gamma": ("gamma", "γ", "gamma3"),
    "mu": ("mu", "μ"),
}


def _plot(table: pd.DataFrame, x: str, y: str, title: str, xlabel: str,
          ylabel: str, output: Path, *, kind: str = "bar") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not table.empty and x in table and y in table:
        if kind == "line":
            ax.plot(table[x].astype(str), table[y], marker="o")
        else:
            ax.bar(table[x].astype(str), table[y])
        ax.tick_params(axis="x", labelrotation=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _recommendations(pipeline: RecommendationPipeline, users: list[Any], *, use_llm: bool,
                     options: PipelineOptions | None = None) -> pd.DataFrame:
    rows = []
    for user in users:
        for rank, item in enumerate(pipeline.recommend(user, use_llm=use_llm, options=options), 1):
            rows.append({"user_id": user, "parent_asin": item.item_id, "score": item.match_percent / 100,
                         "rank": rank, "is_safe": item.safe_badge, "reason": item.reason,
                         "llm_used": item.llm_used})
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "score", "rank", "is_safe", "reason", "llm_used"])


def _measure(recs: pd.DataFrame, test: pd.DataFrame, path: Path) -> dict[str, float]:
    result = evaluate(recs, test, k=10, output_path=path).iloc[0]
    return {key: float(result[key]) for key in ("Precision@K", "Recall@K", "NDCG@K", "HR@K", "Safety@K", "Coverage", "Diversity")}


def _grid_table(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else Path(__file__).with_name("parameter_grid.csv")
    raw = pd.read_csv(path)
    normalized: dict[str, str] = {}
    for canonical, choices in GRID_COLUMNS.items():
        match = next((column for column in choices if column in raw.columns), None)
        if match is None:
            raise ValueError(f"Grid CSV must contain table 2.4 parameter {canonical!r}; accepted names: {choices}")
        normalized[canonical] = match
    result = pd.DataFrame({key: pd.to_numeric(raw[source], errors="raise") for key, source in normalized.items()})
    for name in ("gamma1", "gamma2", "gamma3"):
        if name in raw.columns:
            result[name] = pd.to_numeric(raw[name], errors="raise")
    if result.empty:
        raise ValueError("Grid CSV has headers but no parameter rows")
    if (result[["T_half", "N", "M"]] <= 0).any().any():
        raise ValueError("T½, N and M grid values must be positive")
    for name in ("rho", "eta", "beta", "gamma", "mu"):
        if not result[name].between(0, 1).all():
            raise ValueError(f"{name} grid values must be in [0,1]")
    if {"gamma1", "gamma2", "gamma3"}.issubset(result.columns):
        gamma = result[["gamma1", "gamma2", "gamma3"]]
        if (gamma < 0).any().any() or (gamma > 1).any().any() or not np.allclose(gamma.sum(axis=1), 1):
            raise ValueError("gamma1, gamma2 and gamma3 must be in [0,1] and sum to 1")
    return result


def _options_from_row(row: pd.Series, **ablation: Any) -> PipelineOptions:
    if all(name in row.index for name in ("gamma1", "gamma2", "gamma3")):
        gamma1, gamma2, gamma3 = (float(row[name]) for name in ("gamma1", "gamma2", "gamma3"))
    else:
        # Legacy grid column gamma denotes the LLM weight; distribute the remainder
        # according to the documented initial 0.4:0.2 composition/content ratio.
        gamma3 = float(row.gamma)
        remainder = 1.0 - gamma3
        gamma1, gamma2 = remainder * 2 / 3, remainder / 3
    return PipelineOptions(
        time_decay_half_life_days=float(row.T_half), rho=float(row.rho), eta=float(row.eta),
        beta=float(row.beta), candidate_count=int(row.N), rag_top_m=int(row.M),
        gamma1=gamma1, gamma2=gamma2, gamma3=gamma3,
        mmr_mu=float(row.mu), **ablation,
    )


def _timestamp_days(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == values.notna().sum():
        return numeric.astype(float)
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = parsed.notna()
    result.loc[valid] = (parsed.loc[valid] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() / 86_400
    return result


def _reminder_metrics(
    interactions: pd.DataFrame, products: pd.DataFrame, profiles: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compare mean-interval, smoothed and stock/history restock forecasts.

    The evaluation is rolling-origin: every prediction only sees earlier purchases.
    Stock-aware rows are included only where product package/energy and pet weight
    are available; this avoids silently fabricating inputs.
    """
    if not {"user_id", "parent_asin", "timestamp"}.issubset(interactions.columns):
        return pd.DataFrame(columns=["model", "category", "MAE_days", "within_5_days", "n"])
    frame = interactions.copy()
    frame["day"] = _timestamp_days(frame.timestamp)
    product_info = products.set_index("parent_asin").to_dict(orient="index")
    profile_info = profiles.set_index("user_id").to_dict(orient="index") if profiles is not None and "user_id" in profiles else {}
    errors: dict[tuple[str, str], list[float]] = {}
    for (user_id, product_id), history in frame.dropna(subset=["day"]).groupby(["user_id", "parent_asin"]):
        history = history.sort_values("day")
        dates = history.day.to_numpy(dtype=float)
        if len(dates) < 3:
            continue
        product = product_info.get(product_id, {})
        profile = profile_info.get(user_id, {})
        category = str(product.get("category") or "невідома")
        for position in range(2, len(dates)):
            prior = dates[:position]
            gaps = np.diff(prior)
            gaps = gaps[gaps > 0]
            if not len(gaps):
                continue
            forecasts = {"середній інтервал": float(np.mean(gaps))}
            smoothed = smoothed_purchase_interval([pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(day)) for day in prior], alpha=0.3)
            if smoothed is not None:
                forecasts["експоненційне згладжування"] = float(smoothed)
            stock_forecast = None
            if all(product.get(key) is not None for key in ("package_weight_kg", "energy_kcal_per_kg")) and profile.get("weight_kg"):
                try:
                    stock_days = estimate_food_stock_days(
                        float(profile["weight_kg"]), str(profile.get("pet_species", "cat")),
                        float(product["energy_kcal_per_kg"]), float(product["package_weight_kg"]),
                        int(history.iloc[position - 1].get("quantity", 1) or 1), profile.get("lifestyle"),
                    )
                    omega = 1.0 if position < 3 else 0.5
                    stock_forecast = omega * stock_days + (1 - omega) * float(smoothed or np.mean(gaps))
                except (TypeError, ValueError):
                    pass
            if stock_forecast is not None:
                forecasts["запас + історія"] = stock_forecast
            actual = float(dates[position] - dates[position - 1])
            for model, forecast in forecasts.items():
                errors.setdefault((model, category), []).append(abs(actual - forecast))
    rows = []
    for (model, category), values in sorted(errors.items()):
        rows.append({"model": model, "category": category, "MAE_days": float(np.mean(values)),
                     "within_5_days": float(np.mean(np.asarray(values) <= 5)), "n": len(values)})
    return pd.DataFrame(rows, columns=["model", "category", "MAE_days", "within_5_days", "n"])


def run_all(
    train: pd.DataFrame,
    test: pd.DataFrame,
    products: pd.DataFrame,
    profiles: pd.DataFrame | None,
    grid_csv: str | Path | None,
    output_dir: str | Path,
    *,
    cloud_provider: str = "openai",
    user_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run five evaluations; persist all result tables as CSV and charts as PNG."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required = {"user_id", "parent_asin"}
    for label, frame in (("train", train), ("test", test), ("products", products)):
        missing = required.difference(frame.columns) if label != "products" else {"parent_asin"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing required columns: {sorted(missing)}")
    grid = _grid_table(grid_csv)
    users = sorted(test.user_id.dropna().unique().tolist(), key=str)
    if user_limit is not None:
        users = users[:user_limit]
    relevant_test = test.loc[test.user_id.isin(users)].copy()
    try:
        cloud_client = AnthropicClient() if cloud_provider.lower() == "anthropic" else OpenAIClient()
        cloud_error = ""
    except Exception as exc:
        cloud_client, cloud_error = None, str(exc)
    pipeline = RecommendationPipeline(train, products, profiles, llm_client=cloud_client)
    results: dict[str, pd.DataFrame] = {}

    # E1: thesis baselines plus LLM-only and the proposed hybrid.
    e1_rows = []
    popularity = Popularity().fit(train)
    cf = ItemBasedCF().fit(train)
    als = ImplicitALS().fit(train)
    bpr = BPRRecommender().fit(train)
    content = ContentBased().fit(train, products)
    baselines = (
        ("Popularity", popularity.recommend(users, k=10)),
        ("ItemKNN", cf.recommend(users, k=10)),
        ("Content", content.recommend(users, k=10)),
        ("ALS", als.recommend(users, k=10)),
        ("BPR", bpr.recommend(users, k=10)),
    )
    for name, recs in baselines:
        metrics = _measure(recs, relevant_test, output / f"_metrics_{name}.csv")
        e1_rows.append({"Модель": name, **metrics})
    llm_only = _recommendations(pipeline, users, use_llm=True,
        options=PipelineOptions(use_composition=False, use_rag=False, gamma1=0.5, gamma2=0.0, gamma3=0.5))
    e1_rows.append({"Модель": "LLM без складу та RAG",
                    **_measure(llm_only, relevant_test, output / "_metrics_llm_only.csv")})
    hybrid = _recommendations(pipeline, users, use_llm=True)
    e1_rows.append({"Модель": "Запропонований метод", **_measure(hybrid, relevant_test, output / "_metrics_hybrid.csv")})
    e1 = pd.DataFrame(e1_rows)
    e1.to_csv(output / "E1_baselines.csv", index=False, encoding="utf-8-sig")
    results["E1"] = e1
    _plot(e1, "Модель", "NDCG@K", "E1. Порівняння якості рекомендацій", "Модель", "NDCG@10", output / "E1_baselines.png")

    # E2: ranking quality by the number of historical interactions.
    lengths = train.groupby("user_id").size()
    buckets = pd.cut(lengths, bins=[-1, 2, 5, np.inf], labels=["1–2", "3–5", "6+"])
    e2_rows = []
    for bucket in ("1–2", "3–5", "6+"):
        group_users = [user for user in users if buckets.get(user) == bucket]
        if not group_users:
            continue
        group_recs = hybrid.loc[hybrid.user_id.isin(group_users)]
        group_test = relevant_test.loc[relevant_test.user_id.isin(group_users)]
        e2_rows.append({"Довжина історії": bucket, "Користувачі": len(group_users),
                        **_measure(group_recs, group_test, output / f"_metrics_E2_{bucket}.csv")})
    e2 = pd.DataFrame(e2_rows)
    e2.to_csv(output / "E2_history_length.csv", index=False, encoding="utf-8-sig")
    results["E2"] = e2
    _plot(e2, "Довжина історії", "NDCG@K", "E2. Якість за довжиною історії", "Взаємодії в історії", "NDCG@10", output / "E2_history_length.png")

    # E3: individual ablations and the complete table 2.4 parameter grid.
    ablations = [
        ("Без складу", {"use_composition": False}),
        ("Без RAG", {"use_rag": False}),
        ("Без LLM", {}),
        ("Без MMR", {"use_mmr": False}),
        ("Без часового згасання", {"time_decay_half_life_days": None}),
        # Unsafe ranking is computed only for this offline ablation table; it is
        # never used by the API or demo because normal PipelineOptions stay safe.
        ("Без фільтра безпеки (офлайн)", {"use_safety_filter": False}),
    ]
    e3_ablation_rows = []
    for name, changes in ablations:
        if name == "Без LLM":
            recs = _recommendations(pipeline, users, use_llm=False)
        else:
            recs = _recommendations(pipeline, users, use_llm=True, options=PipelineOptions(**changes))
        e3_ablation_rows.append({"Абляція": name, **_measure(recs, relevant_test, output / f"_metrics_ablation_{len(e3_ablation_rows)}.csv")})
    e3_ablation = pd.DataFrame(e3_ablation_rows)
    e3_ablation.to_csv(output / "E3_ablations.csv", index=False, encoding="utf-8-sig")
    results["E3_ablations"] = e3_ablation
    _plot(e3_ablation, "Абляція", "NDCG@K", "E3. Абляційний аналіз", "Варіант", "NDCG@10", output / "E3_ablations.png")

    e3_grid_rows = []
    for grid_index, row in grid.iterrows():
        options = _options_from_row(row)
        recs = _recommendations(pipeline, users, use_llm=True, options=options)
        metrics = _measure(recs, relevant_test, output / f"_metrics_grid_{grid_index}.csv")
        e3_grid_rows.append({**row.to_dict(), **metrics})
    e3_grid = pd.DataFrame(e3_grid_rows)
    e3_grid.to_csv(output / "E3_parameter_grid.csv", index=False, encoding="utf-8-sig")
    results["E3_grid"] = e3_grid
    _plot(e3_grid, "beta", "NDCG@K", "E3. Сітка параметрів таблиці 2.4", "β", "NDCG@10", output / "E3_parameter_grid.png", kind="line")

    # E4: stable 500-user subset, three evaluation repetitions, cloud vs Ollama.
    e4_users = sorted(users, key=str)[:500]
    cloud = cloud_client
    try:
        local = OllamaClient()
    except Exception as exc:
        local = None
        ollama_error = str(exc)
    else:
        ollama_error = ""
    e4_rows = []
    for model_name, client, error in (("Хмарна", cloud, cloud_error), ("Ollama", local, ollama_error)):
        if client is None:
            e4_rows.append({"Модель": model_name, "Повтор": "середнє", "Помилка": error,
                            "NDCG@K": math.nan, "Safety@K": math.nan,
                            "LLM-validity": math.nan, "latency_seconds": math.nan,
                            "cost_usd": math.nan})
            continue
        repetition_metrics = []
        for repetition in range(1, 4):
            pipeline.llm_client = client
            started = time.perf_counter()
            recs = _recommendations(pipeline, e4_users, use_llm=True)
            latency_seconds = time.perf_counter() - started
            metrics = _measure(recs, relevant_test.loc[relevant_test.user_id.isin(e4_users)],
                               output / f"_metrics_E4_{model_name}_{repetition}.csv")
            repetition_metrics.append(metrics)
            e4_rows.append({"Модель": model_name, "Повтор": repetition,
                            "LLM-validity": float(recs.llm_used.mean()) if not recs.empty else 0.0,
                            "latency_seconds": latency_seconds,
                            # Provider usage/token pricing is not exposed consistently by
                            # every client; leave cost unknown rather than estimate it.
                            "cost_usd": math.nan,
                            **metrics})
        e4_rows.append({"Модель": model_name, "Повтор": "середнє",
                        "LLM-validity": float(np.mean([row["LLM-validity"] for row in e4_rows[-3:]])),
                        "latency_seconds": float(np.mean([row["latency_seconds"] for row in e4_rows[-3:]])),
                        "cost_usd": math.nan,
                        **pd.DataFrame(repetition_metrics).mean(numeric_only=True).to_dict()})
    e4 = pd.DataFrame(e4_rows)
    e4.to_csv(output / "E4_llm_comparison.csv", index=False, encoding="utf-8-sig")
    results["E4"] = e4
    e4_mean = e4.loc[e4["Повтор"] == "середнє"]
    _plot(e4_mean, "Модель", "NDCG@K", "E4. Хмарна LLM та Ollama", "Модель", "NDCG@10", output / "E4_llm_comparison.png")
    pipeline.llm_client = None

    # E5: repeat-purchase reminder forecast MAE in days.
    e5 = _reminder_metrics(train, products, profiles)
    e5.to_csv(output / "E5_reminders_mae.csv", index=False, encoding="utf-8-sig")
    results["E5"] = e5
    if not e5.empty:
        e5_summary = pd.DataFrame([
            {"model": model, "MAE_days": float(np.average(rows.MAE_days, weights=rows.n)),
             "within_5_days": float(np.average(rows.within_5_days, weights=rows.n))}
            for model, rows in e5.groupby("model")
        ])
    else:
        e5_summary = e5
    _plot(e5_summary, "model", "MAE_days", "E5. Точність прогнозу нагадувань", "Метод прогнозування", "MAE, дні", output / "E5_reminders_mae.png")
    # Remove per-metric scratch files; final deliverables are the E1–E5 CSV/PNG files.
    for scratch in output.glob("_metrics_*.csv"):
        scratch.unlink()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="train.parquet")
    parser.add_argument("--test", required=True, help="test.parquet")
    parser.add_argument("--products", required=True, help="products.parquet")
    parser.add_argument("--profiles", help="profiles.parquet (optional)")
    parser.add_argument("--grid-csv", help="CSV values from thesis table 2.4 (defaults to experiments/parameter_grid.csv)")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--cloud-provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--user-limit", type=int, help="optional deterministic limit for E1–E3")
    args = parser.parse_args()
    run_all(pd.read_parquet(args.train), pd.read_parquet(args.test), pd.read_parquet(args.products),
            pd.read_parquet(args.profiles) if args.profiles else None, args.grid_csv, args.output_dir,
            cloud_provider=args.cloud_provider, user_limit=args.user_limit)


if __name__ == "__main__":
    main()
