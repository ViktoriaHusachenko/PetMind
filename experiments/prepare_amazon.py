"""Prepare Amazon Reviews 2023 Pet Supplies (official 5-core) and Open Pet Food Facts for experiments.

Inputs (data/raw/):
  Pet_Supplies.5core.csv.gz   benchmark/5core/rating_only/Pet_Supplies.csv.gz (McAuley Lab)
  meta_Pet_Supplies.jsonl.gz  raw/meta_categories/meta_Pet_Supplies.jsonl.gz
  opff.csv.gz                 Open Pet Food Facts export (tab-separated)

Writes data/processed/{reviews,products,profiles,train,val,test}.parquet using the thesis
data-preparation functions from petrec.data_prep (user sampling, OPFF matching, profiles,
leave-last-out temporal split). The metadata file is streamed so only sampled items are kept.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from petrec.data_prep import (  # noqa: E402
    OPFF_COLUMNS, PRODUCT_COLUMNS, build_pet_profiles, clean_reviews, match_open_pet_food_facts,
    print_statistics, sample_users, save_processed_data, temporal_split,
)


def _species(categories: list[str] | None, title: str) -> str | None:
    text = (" ".join(categories or []) + " " + (title or "")).lower()
    cat = text.count("cat") + text.count("kitten")
    dog = text.count("dog") + text.count("puppy")
    return "cat" if cat > dog else "dog" if dog > cat else None


def read_metadata(path: Path, items: set[str]) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("parent_asin") not in items:
                continue
            description = record.get("description") or []
            features = record.get("features") or []
            rows.append({
                "parent_asin": record["parent_asin"], "title": record.get("title") or "",
                "brand": (record.get("details") or {}).get("Brand") or record.get("store") or "",
                "categories": record.get("categories") or [],
                "description": " ".join(map(str, description + features)),
                "price": pd.to_numeric(record.get("price"), errors="coerce"),
                "species": _species(record.get("categories"), record.get("title") or ""),
            })
    frame = pd.DataFrame(rows)
    for column in PRODUCT_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = Path(args.raw)

    reviews = clean_reviews(pd.read_csv(raw / "Pet_Supplies.5core.csv.gz"))
    print_statistics(reviews, "5-core")
    reviews = sample_users(reviews, n_users=args.n_users, random_state=args.seed)
    print_statistics(reviews, "sample")

    products = read_metadata(raw / "meta_Pet_Supplies.jsonl.gz", set(reviews.parent_asin))
    print(f"metadata rows: {len(products):,}")
    opff = pd.read_csv(raw / "opff.csv.gz", sep="\t", low_memory=False, usecols=lambda c: c in OPFF_COLUMNS)
    opff = opff.dropna(subset=["product_name"])
    products = match_open_pet_food_facts(products, opff)
    print(f"OPFF matches: {int(products.matched_opff.sum()):,}; with ingredients: "
          f"{int(products.ingredients_text.fillna('').str.len().gt(0).sum()):,}")

    profiles = build_pet_profiles(reviews, products, random_state=args.seed)
    splits = temporal_split(reviews)
    save_processed_data(args.out, reviews, products, profiles, splits)
    for name, split in splits.items():
        print_statistics(split, name)


if __name__ == "__main__":
    main()
