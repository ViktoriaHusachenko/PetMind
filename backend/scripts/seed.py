"""Load processed Parquet products/interactions and seed demo pet profiles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import secrets
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from app.db.database import SessionLocal
from app.db.models import KBChunk, Pet, Product, ProductIngredient, Purchase, User
from petrec.composition import ingredient_list, is_safe, load_dictionary, tokenize
from petrec.profiles import ProductEmbedder
from petrec.rag import Embedder, chunk_text

NAMESPACE = uuid.UUID("f6d4197b-6338-4a82-a96f-1019607b23d2")


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _number(value: Any) -> float | None:
    value = _value(value)
    if value is None:
        return None
    try:
        number = float(str(value).replace("$", "").replace(",", ""))
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _weight_kg(row: dict[str, Any]) -> float | None:
    direct = _number(row.get("package_weight_kg"))
    if direct is not None and direct > 0:
        return direct
    value = str(_value(row.get("size", row.get("weight", ""))) or "").lower()
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(kg|g|lb|lbs|oz)\b", value)
    if not match:
        return None
    amount = float(match.group(1).replace(",", "."))
    return amount * {"kg": 1.0, "g": 0.001, "lb": 0.453592, "lbs": 0.453592, "oz": 0.0283495}[match.group(2)]
def _datetime_from_processed(value: Any) -> datetime | None:
    value = _value(value)
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(value))
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _age_birth_date(years: int, today: date | None = None) -> date:
    current = today or date.today()
    try:
        return current.replace(year=current.year - years)
    except ValueError:  # 29 February in a non-leap target year
        return current.replace(year=current.year - years, day=28)


def _demo_profiles() -> list[dict[str, Any]]:
    today = date.today()
    return [
        {"external_id": "petly-demo-murchyk", "display_name": "Демо-користувач 1",
         "pet": {"name": "Мурчик", "species": "cat", "birth_date": _age_birth_date(6, today),
                 "breed": None, "weight_kg": 4.5, "allergens": ["птиця"],
                 "diet_restrictions": [], "lifestyle": "стерилізований", "photo_url": None}},
        {"external_id": "petly-demo-richie", "display_name": "Демо-користувач 2",
         "pet": {"name": "Річі", "species": "dog", "birth_date": _age_birth_date(4, today),
                 "breed": "золотистий ретривер", "weight_kg": 31.0, "allergens": [],
                 "diet_restrictions": [], "lifestyle": None, "photo_url": None}},
        {"external_id": "petly-demo-third", "display_name": "Демо-користувач 3",
         "pet": {"name": "Луна", "species": "cat", "birth_date": _age_birth_date(2, today),
                 "breed": None, "weight_kg": 3.8, "allergens": [],
                 "diet_restrictions": [], "lifestyle": None, "photo_url": None}},
    ]


def _seed_demo_profiles() -> dict[str, int]:
    """Create demo logins and pets without requiring the research catalog files."""
    demos = _demo_profiles()
    user_ids = {demo["external_id"]: uuid.uuid5(NAMESPACE, demo["external_id"]) for demo in demos}
    with SessionLocal() as session:
        users = insert(User).values([
            {"id": user_ids[demo["external_id"]], "external_id": demo["external_id"],
             "display_name": demo["display_name"]}
            for demo in demos
        ])
        session.execute(users.on_conflict_do_nothing(index_elements=["external_id"]))
        for demo in demos:
            pet_id = uuid.uuid5(NAMESPACE, f"pet:{demo['external_id']}:{demo['pet']['name']}")
            pet = session.get(Pet, pet_id)
            if pet is None:
                session.add(Pet(id=pet_id, user_id=user_ids[demo["external_id"]], **demo["pet"]))
            else:
                for field, value in demo["pet"].items():
                    setattr(pet, field, value)
        session.commit()
    return {"users": len(demos), "pets": len(demos)}


def _seed_admin_account() -> None:
    """Create the local demo administrator; override credentials through environment variables."""
    email = os.getenv("PETMIND_ADMIN_EMAIL", "admin@petmind.local").strip().casefold()
    password = os.getenv("PETMIND_ADMIN_PASSWORD", "PetMindAdmin2026!")
    salt = secrets.token_bytes(16)
    rounds = 310_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    password_hash = (f"pbkdf2_sha256${rounds}${base64.urlsafe_b64encode(salt).decode()}$"
                     f"{base64.urlsafe_b64encode(digest).decode()}")
    with SessionLocal() as session:
        user = session.query(User).filter_by(email=email).one_or_none()
        if user is None:
            user = User(id=uuid.uuid5(NAMESPACE, "petmind-local-admin"), email=email,
                        external_id="petmind-local-admin", display_name="Адміністратор PetMind")
            session.add(user)
        user.password_hash = password_hash
        user.is_admin = True
        session.commit()


def _seed_demo_catalog() -> int:
    """Insert clearly identified sample goods to make the UI usable without research data."""
    rows = [
        {"id": "PETMIND-DEMO-CAT-SALMON", "title": "Сухий корм для котів із лососем", "brand": "PetMind Demo",
         "category": "Сухий корм", "species": "cat", "price": 649.0, "image_url": "/demo-products/cat-salmon.svg",
         "description": "Демонстраційний товар каталогу. Дані не є комерційною пропозицією.", "ingredients_text": "salmon, rice, salmon fat",
         "protein": 32.0, "fat": 16.0, "is_regular": True, "package_weight_kg": 2.0},
        {"id": "PETMIND-DEMO-CAT-STERILIZED-SALMON", "title": "Сухий корм для стерилізованих котів із лососем", "brand": "PetMind Demo",
         "category": "Сухий корм для стерилізованих котів", "species": "cat", "price": 689.0, "image_url": "/demo-products/cat-salmon.svg",
         "description": "Демонстраційний раціон для стерилізованих дорослих котів. Дані не є комерційною пропозицією.", "ingredients_text": "salmon, rice, salmon fat",
         "protein": 34.0, "fat": 12.0, "is_regular": True, "package_weight_kg": 2.0},
        {"id": "PETMIND-DEMO-CAT-CHICKEN", "title": "Корм для котів із курятиною", "brand": "PetMind Demo",
         "category": "Сухий корм", "species": "cat", "price": 529.0, "image_url": "/demo-products/cat-chicken.svg",
         "description": "Демонстраційний товар з птицею для перевірки фільтра безпеки.", "ingredients_text": "chicken, corn, rice",
         "protein": 30.0, "fat": 14.0, "is_regular": True, "package_weight_kg": 2.0},
        {"id": "PETMIND-DEMO-DOG-BEEF", "title": "Повнораціонний корм для собак із яловичиною", "brand": "PetMind Demo",
         "category": "Сухий корм", "species": "dog", "price": 799.0, "image_url": "/demo-products/dog-beef.svg",
         "description": "Демонстраційний товар каталогу. Дані не є комерційною пропозицією.", "ingredients_text": "beef, rice, carrot",
         "protein": 26.0, "fat": 14.0, "is_regular": True, "package_weight_kg": 3.0},
        {"id": "PETMIND-DEMO-DOG-SALMON", "title": "Корм для собак із лососем", "brand": "PetMind Demo",
         "category": "Вологий корм", "species": "dog", "price": 119.0, "image_url": "/demo-products/dog-salmon.svg",
         "description": "Демонстраційний товар каталогу. Дані не є комерційною пропозицією.", "ingredients_text": "salmon, pumpkin, rice",
         "protein": 10.0, "fat": 6.0, "is_regular": True, "package_weight_kg": 0.4},
    ]
    with SessionLocal() as session:
        statement = insert(Product).values(rows)
        update_fields = {key: getattr(statement.excluded, key) for key in rows[0] if key != "id"}
        session.execute(statement.on_conflict_do_update(index_elements=["id"], set_=update_fields))
        session.commit()
    return len(rows)


def seed(processed_dir: str | Path = ROOT / "data" / "processed", *, demo_only: bool = False) -> dict[str, int]:
    """Idempotently insert products, ingredient rows, purchases, chunks, and demos."""
    if demo_only:
        counts = _seed_demo_profiles()
        _seed_admin_account()
        counts["demo_products"] = _seed_demo_catalog()
        return counts
    processed_dir = Path(processed_dir)
    products_path = processed_dir / "products.parquet"
    reviews_path = processed_dir / "reviews.parquet"
    if not products_path.exists() or not reviews_path.exists():
        raise FileNotFoundError(
            f"Expected {products_path} and {reviews_path}; prepare Parquet data first, "
            "or use --demo-only to create only the demo users and pets."
        )
    products_frame = pd.read_parquet(products_path)
    reviews = pd.read_parquet(reviews_path)
    if "parent_asin" not in products_frame or not {"user_id", "parent_asin"}.issubset(reviews.columns):
        raise ValueError("Parquet inputs must have product parent_asin and review user_id/parent_asin columns")

    embedding_path = processed_dir / "product_embeddings.npy"
    product_embedder = ProductEmbedder(cache_path=embedding_path)
    product_vectors = product_embedder.embed_products(products_frame)
    rag_embedder = Embedder(model=product_embedder._get_model())
    normalizations = load_dictionary()

    product_rows: list[dict[str, Any]] = []
    ingredient_rows: list[dict[str, Any]] = []
    documents: list[tuple[str, str]] = []
    seen_products: set[str] = set()
    for row in products_frame.to_dict(orient="records"):
        parent = _value(row.get("parent_asin"))
        if parent is None:
            continue
        product_id = str(parent)
        if product_id in seen_products:
            continue
        seen_products.add(product_id)
        description = _value(row.get("description"))
        ingredients_text = _value(row.get("ingredients_text"))
        categories = row.get("categories")
        category_text = " ".join(map(str, categories)) if isinstance(categories, (list, tuple, set)) else str(_value(categories) or "")
        searchable = f"{row.get('title', '')} {category_text}".lower()
        species = "cat" if any(key in searchable for key in ("cat", "feline", "кіт")) else "dog" if any(key in searchable for key in ("dog", "canine", "собак")) else None
        image = _value(row.get("image_url"))
        if image is None and isinstance(row.get("images"), (list, tuple)) and row["images"]:
            first = row["images"][0]
            image = first.get("large", first.get("thumb")) if isinstance(first, dict) else first
        vector = product_vectors.get(parent)
        product_rows.append({
            "id": product_id,
            "title": str(_value(row.get("title")) or product_id),
            "brand": _value(row.get("brands", row.get("brand"))),
            "category": category_text or None,
            "species": species,
            "price": _number(row.get("price")),
            "image_url": str(image) if image is not None else None,
            "description": str(description) if description is not None else None,
            "ingredients_text": str(ingredients_text) if ingredients_text is not None else None,
            "protein": _number(row.get("proteins_100g", row.get("protein"))),
            "fat": _number(row.get("fat_100g", row.get("fat"))),
            "moisture": _number(row.get("moisture_100g", row.get("moisture"))),
            "kcal_100g": _number(row.get("energy-kcal_100g", row.get("kcal_100g"))),
            "is_regular": any(term in searchable for term in
                               ("food", "feed", "litter", "parasite", "flea", "tick", "worm", "vitamin", "supplement")),
            "package_weight_kg": _weight_kg(row),
            "energy_kcal_per_kg": (_number(row.get("energy-kcal_100g", row.get("kcal_100g"))) * 10
                                   if _number(row.get("energy-kcal_100g", row.get("kcal_100g"))) is not None else None),
            "embedding": vector.tolist() if vector is not None else None,
        })
        for position, token in enumerate(tokenize(str(ingredients_text or ""))):
            normalized = normalizations.get(token)
            ingredient_rows.append({"product_id": product_id, "position": position, "name": token,
                "canonical_name": normalized.canonical_name if normalized else token,
                "allergen_group": normalized.allergen_group if normalized else None,
                "theta": normalized.theta if normalized else None})
        for field, content in (("опис", description), ("склад", ingredients_text)):
            if content:
                documents.extend((product_id, piece) for piece in chunk_text(str(content)))

    # Store product vectors as normalized E5 vectors and chunk vectors in the same database.
    if product_rows:
        to_encode = [doc for _, doc in documents]
        chunk_vectors = rag_embedder.encode(to_encode) if to_encode else np.empty((0, 384), dtype=np.float32)
    else:
        chunk_vectors = np.empty((0, 384), dtype=np.float32)

    external_users = {str(value) for value in reviews.user_id.dropna().unique()}
    demo_users = {entry["external_id"] for entry in _demo_profiles()}
    external_users.update(demo_users)
    user_ids = {external: uuid.uuid5(NAMESPACE, external) for external in external_users}
    counts = {"users": 0, "pets": 0, "products": 0, "purchases": 0, "demo_purchases": 0,
              "ingredients": 0, "kb_chunks": 0}
    with SessionLocal() as session:
        pg_user = insert(User)
        users_to_upsert = [{"id": user_ids[external], "external_id": external,
                            "display_name": next((entry["display_name"] for entry in _demo_profiles()
                                                   if entry["external_id"] == external), None)}
                           for external in sorted(external_users)]
        session.execute(pg_user.values(users_to_upsert).on_conflict_do_nothing(index_elements=["external_id"]))
        counts["users"] = len(users_to_upsert)

        for demo in _demo_profiles():
            pet_id = uuid.uuid5(NAMESPACE, f"pet:{demo['external_id']}:{demo['pet']['name']}")
            exists = session.get(Pet, pet_id)
            if exists is None:
                session.add(Pet(id=pet_id, user_id=user_ids[demo["external_id"]], **demo["pet"]))
            else:
                for field, value in demo["pet"].items():
                    setattr(exists, field, value)
            counts["pets"] += 1
        session.flush()

        if product_rows:
            pg_product = insert(Product).values(product_rows)
            update_fields = {key: getattr(pg_product.excluded, key) for key in product_rows[0] if key != "id"}
            session.execute(pg_product.on_conflict_do_update(index_elements=["id"], set_=update_fields))
            session.execute(ProductIngredient.__table__.delete())
            if ingredient_rows:
                session.execute(insert(ProductIngredient), ingredient_rows)
            session.execute(KBChunk.__table__.delete())
            chunk_rows = [{"product_id": product_id, "text": text, "embedding": vector.tolist()}
                          for (product_id, text), vector in zip(documents, chunk_vectors)]
            for offset in range(0, len(chunk_rows), 500):
                session.execute(insert(KBChunk), chunk_rows[offset:offset + 500])
            counts["products"] = len(product_rows)
            counts["ingredients"] = len(ingredient_rows)
            counts["kb_chunks"] = len(chunk_rows)

        purchase_rows = []
        product_ids = {str(value) for value in products_frame.parent_asin.dropna().unique()}
        for row in reviews.to_dict(orient="records"):
            product_id = _value(row.get("parent_asin"))
            user_id = _value(row.get("user_id"))
            ts = _datetime_from_processed(row.get("timestamp"))
            if product_id is None or str(product_id) not in product_ids or user_id is None or ts is None:
                continue
            purchase_rows.append({"user_id": user_ids[str(user_id)], "product_id": str(product_id), "ts": ts,
                                  "rating": _number(row.get("rating"))})
        if purchase_rows:
            pg_purchase = insert(Purchase).values(purchase_rows)
            session.execute(pg_purchase.on_conflict_do_nothing(constraint="uq_purchase_user_product_ts"))
        counts["purchases"] = len(purchase_rows)

        # Repeatable demo history exercises the replenishment flow. The third
        # demo account remains purchase-free for cold-start verification.
        murchyk_id = user_ids["petly-demo-murchyk"]
        demo_food = next((row for row in product_rows
            if row["species"] == "cat" and row["is_regular"]
            and row["ingredients_text"]
            and is_safe(ingredient_list(row["ingredients_text"], normalizations), ["птиця"])), None)
        if demo_food is not None:
            demo_purchases = [
                {"user_id": murchyk_id, "product_id": demo_food["id"],
                 "ts": datetime(2026, 8, 1, 12, tzinfo=timezone.utc), "rating": None, "quantity": 1},
                {"user_id": murchyk_id, "product_id": demo_food["id"],
                 "ts": datetime(2026, 8, 31, 12, tzinfo=timezone.utc), "rating": None, "quantity": 1},
            ]
            pg_demo = insert(Purchase).values(demo_purchases)
            session.execute(pg_demo.on_conflict_do_nothing(constraint="uq_purchase_user_product_ts"))
            counts["demo_purchases"] = len(demo_purchases)
        session.commit()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--demo-only", action="store_true", help="create demo users, pets, and sample products without Parquet catalog data")
    args = parser.parse_args()
    result = seed(args.processed_dir, demo_only=args.demo_only)
    for name, count in result.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
