"""Load processed Parquet products/interactions and seed demo pet profiles."""

from __future__ import annotations

import argparse
import os
import re
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.auth import password_hasher
from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import KBChunk, Pet, Product, ProductIngredient, Purchase, User
from app.embeddings import EmbeddingService
from app.safety import default_evaluator
from demo_catalog import demo_catalog
from petrec.composition import load_dictionary, tokenize
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
        {"external_id": "petly-demo-murchyk", "display_name": "Демо-користувач 1", "email": "murchyk@petmind.local",
         "pet": {"name": "Мурчик", "species": "cat", "birth_date": _age_birth_date(6, today),
                 "breed": None, "weight_kg": 4.5, "allergens": ["птиця"],
                 "diet_restrictions": [], "lifestyle": "стерилізований", "photo_url": None}},
        {"external_id": "petly-demo-richie", "display_name": "Демо-користувач 2", "email": "richie@petmind.local",
         "pet": {"name": "Річі", "species": "dog", "birth_date": _age_birth_date(4, today),
                 "breed": "золотистий ретривер", "weight_kg": 31.0, "allergens": [],
                 "diet_restrictions": [], "lifestyle": None, "photo_url": None}},
        {"external_id": "petly-demo-third", "display_name": "Демо-користувач 3", "email": "luna@petmind.local",
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
        password_hash = password_hasher.hash(get_settings().demo_password)
        for demo in demos:
            user = session.get(User, user_ids[demo["external_id"]])
            # Demo owners can sign in (BUG-028); the password comes from PETMIND_DEMO_PASSWORD.
            if user is not None:
                user.email, user.password_hash = demo["email"], password_hash
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
    settings = get_settings()
    email = settings.admin_email
    password_hash = password_hasher.hash(settings.admin_password)
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
        {"id": "PETMIND-DEMO-ROYAL-CANIN-STERILISED", "title": "Royal Canin Sterilised 37 — корм для стерилізованих котів", "brand": "Royal Canin",
         "category": "Сухий корм", "species": "cat", "price": 899.0,
         "image_url": "https://www.petcorner.rs/cdn/shop/files/royal-canin-sterilised-37-suva-hrana-macke.png?v=1754066639&width=600",
         "description": "Демонстраційна позиція з реальним найменуванням бренду; ціна-приклад, продаж і наявність не підтверджені. Склад звіряйте з етикеткою конкретної упаковки.",
         "ingredients_text": "dehydrated poultry protein, maize, wheat gluten, rice, animal fats", "protein": 37.0, "fat": 12.0, "is_regular": True, "package_weight_kg": 2.0},
        {"id": "PETMIND-DEMO-PURINA-PROPLAN-SALMON", "title": "Purina Pro Plan Adult Salmon — сухий корм для дорослих котів", "brand": "Purina Pro Plan",
         "category": "Сухий корм", "species": "cat", "price": 799.0,
         "image_url": "https://www.purina.nl/sites/default/files/2023-06/07613036508315_3_frontplunge.png",
         "description": "Демонстраційна позиція з реальним найменуванням бренду; ціна-приклад, продаж і наявність не підтверджені. Склад звіряйте з етикеткою конкретної упаковки.",
         "ingredients_text": "salmon, poultry protein, rice, corn", "protein": 36.0, "fat": 16.0, "is_regular": True, "package_weight_kg": 1.5},
        {"id": "PETMIND-DEMO-HILLS-CHICKEN", "title": "Hill’s Science Diet Adult Chicken — сухий корм для котів", "brand": "Hill’s Science Diet",
         "category": "Сухий корм", "species": "cat", "price": 949.0,
         "image_url": "https://cdn11.bigcommerce.com/s-xfu1s3ki5p/images/stencil/700w/attribute_rule_images/4459_source_1772138542.png",
         "description": "Демонстраційна позиція з реальним найменуванням бренду; ціна-приклад, продаж і наявність не підтверджені. Склад звіряйте з етикеткою конкретної упаковки.",
         "ingredients_text": "chicken, brown rice, chicken meal", "protein": 32.0, "fat": 18.0, "is_regular": True, "package_weight_kg": 1.5},
        {"id": "PETMIND-DEMO-NEXGARD-DOG", "title": "NexGard для собак 4–10 кг (демо-картка)", "brand": "NexGard",
         "category": "Ветаптека", "species": "dog", "price": 1565.0,
         "image_url": "https://www.boehringer-ingelheim.com/sites/default/files/Animal_Health/NexGard_EU_Pack_3ct_Medium%20for%20web.png",
         "description": "Демонстраційна картка ветпрепарату. Ціна-приклад за онлайн-карткою Rozetka; товар і наявність у PetMind не продаються. Перед застосуванням проконсультуйтеся з ветеринаром.",
         "ingredients_text": "", "protein": None, "fat": None, "is_regular": False, "package_weight_kg": 0.0},
        {"id": "PETMIND-DEMO-ADVANTAGE-CAT", "title": "Advantage 80 для котів понад 4 кг (демо-картка)", "brand": "Advantage",
         "category": "Ветаптека", "species": "cat", "price": 221.26,
         "image_url": "https://cdn.webshopapp.com/shops/43367/files/470992585/advantage-advantage-katze.jpg",
         "description": "Демонстраційна картка ветпрепарату. Ціна-приклад за онлайн-карткою Pethouse; товар і наявність у PetMind не продаються. Перед застосуванням проконсультуйтеся з ветеринаром.",
         "ingredients_text": "", "protein": None, "fat": None, "is_regular": False, "package_weight_kg": 0.0},
    ]
    by_id = {row["id"]: row for row in rows}
    for row in demo_catalog():
        by_id.setdefault(row["id"], row)  # hand-written rows win on id collisions
    columns = sorted({key for row in by_id.values() for key in row})
    rows = [{column: row.get(column) for column in columns} for row in by_id.values()]
    with SessionLocal() as session:
        statement = insert(Product).values(rows)
        update_fields = {key: getattr(statement.excluded, key) for key in columns if key != "id"}
        session.execute(statement.on_conflict_do_update(index_elements=["id"], set_=update_fields))
        session.commit()
    return len(rows)


def _seed_demo_enrichment() -> dict[str, int]:
    """Ingredient rows, product vectors and knowledge chunks for demo goods; idempotent."""
    evaluator = default_evaluator()
    embeddings = EmbeddingService()
    with SessionLocal() as session:
        products = session.query(Product).filter(Product.id.like("PETMIND-DEMO-%")).all()
        ids = [product.id for product in products]
        session.query(ProductIngredient).filter(ProductIngredient.product_id.in_(ids)).delete(synchronize_session=False)
        ingredient_rows = []
        for product in products:
            for position, item in enumerate(evaluator.items(product.ingredients_text)):
                known = hasattr(item, "canonical_name")
                ingredient_rows.append({"product_id": product.id, "position": position,
                                        "name": item.variant if known else str(item),
                                        "canonical_name": item.canonical_name if known else str(item),
                                        "allergen_group": (item.allergen_group or None) if known else None,
                                        "theta": item.theta if known else None})
        if ingredient_rows:
            session.execute(insert(ProductIngredient), ingredient_rows)
        session.commit()
        vectors = embeddings.ensure_product_vectors(session, products)
        session.query(KBChunk).filter(KBChunk.product_id.in_(ids)).delete(synchronize_session=False)
        documents = []
        for product in products:
            composition = (f"Склад товару «{product.title}»: {product.ingredients_text}"
                           if product.ingredients_text else None)
            for content in (product.description, composition):
                if content:
                    documents.extend((product.id, piece) for piece in chunk_text(str(content)))
        if documents:
            encoded = embeddings.rag_embedder().encode([text for _, text in documents])
            session.execute(insert(KBChunk), [{"product_id": pid, "text": text, "embedding": vector.tolist()}
                                              for (pid, text), vector in zip(documents, encoded)])
        session.commit()
    return {"demo_ingredients": len(ingredient_rows), "demo_vectors_added": vectors, "demo_kb_chunks": len(documents)}


def _seed_demo_purchases() -> int:
    """Two purchases 30 days apart per regular demo food so replenishment reminders appear."""
    plan = {"petly-demo-murchyk": "PETMIND-DEMO-CAT-STERILIZED-SALMON",
            "petly-demo-richie": "PETMIND-DEMO-DOG-ADULT-BEEF"}
    today = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    rows = []
    with SessionLocal() as session:
        for external_id, product_id in plan.items():
            user_id = uuid.uuid5(NAMESPACE, external_id)
            if session.get(Product, product_id) is None or session.get(User, user_id) is None:
                continue
            already = session.query(Purchase).filter_by(user_id=user_id, product_id=product_id).count()
            if already:
                continue
            pet = session.query(Pet).filter_by(user_id=user_id).first()
            for days_ago in (55, 25):
                row = {"user_id": user_id, "product_id": product_id, "ts": today - timedelta(days=days_ago),
                       "rating": 5.0, "quantity": 1}
                if pet is not None and hasattr(Purchase, "pet_id"):
                    row["pet_id"] = pet.id
                rows.append(row)
        if rows:
            session.execute(insert(Purchase).values(rows).on_conflict_do_nothing(
                constraint="uq_purchase_user_product_ts"))
            session.commit()
    return len(rows)


def seed(processed_dir: str | Path = ROOT / "data" / "processed", *, demo_only: bool = False) -> dict[str, int]:
    """Idempotently insert products, ingredient rows, purchases, chunks, and demos."""
    if demo_only:
        counts = _seed_demo_profiles()
        _seed_admin_account()
        counts["demo_products"] = _seed_demo_catalog()
        counts.update(_seed_demo_enrichment())
        counts["demo_purchases"] = _seed_demo_purchases()
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
        species = (
            "cat" if any(key in searchable for key in ("cat", "feline", "кіт")) else
            "dog" if any(key in searchable for key in ("dog", "canine", "собак")) else
            "rabbit" if any(key in searchable for key in ("rabbit", "bunny", "lapin", "крол")) else
            "bird" if any(key in searchable for key in ("bird", "parrot", "avian", "птах", "папуг")) else
            "rodent" if any(key in searchable for key in ("rodent", "hamster", "guinea pig", "mouse", "rat", "гризун", "хом'як")) else
            "reptile" if any(key in searchable for key in ("reptile", "lizard", "turtle", "snake", "рептил", "черепах")) else
            "fish" if any(key in searchable for key in ("aquarium fish", "ornamental fish", "goldfish", "fish food", "акваріум", "риб")) else None
        )
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
            and default_evaluator().evaluate(row["ingredients_text"], ["птиця"]).safe), None)
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
