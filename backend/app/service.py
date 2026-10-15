"""Database-backed API operations; ranking and safety algorithms stay in petrec."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from petrec.feedback import update_preference_state
from petrec.llm import LLMClient
from petrec.pipeline import RecommendationPipeline
from petrec.rag import RAGIndex, SearchResult, VectorStore, search_pg_chunks

from .auth import TokenAuthMixin, password_hasher
from .config import get_settings
from .embeddings import EmbeddingService
from .llm import LLMClientFactory, LLMRuntime
from .safety import SafetyDecision, default_evaluator
from .search import SearchQuery, rank
from .db.database import SessionLocal, engine
from .db.models import Event, KBChunk, Pet, Product, ProductIngredient, Purchase, Reminder, User

logger = logging.getLogger("petmind.api")
PAGE_SIZE = 24
MAX_CF_INTERACTIONS = 50_000


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except ValueError as exc:
        raise LookupError("Invalid identifier") from exc


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _iso_days(value: datetime) -> float:
    return value.timestamp() / 86400.0


def _user_out(user: User) -> dict[str, Any]:
    return {"id": str(user.id), "external_id": user.external_id, "display_name": user.display_name,
            "is_admin": bool(user.is_admin)}


def _pet_out(pet: Pet) -> dict[str, Any]:
    return {"id": str(pet.id), "user_id": str(pet.user_id), "name": pet.name,
            "species": pet.species, "birth_date": pet.birth_date, "breed": pet.breed,
            "weight_kg": _float(pet.weight_kg), "allergens": pet.allergens or [],
            "diet_restrictions": pet.diet_restrictions or [], "lifestyle": pet.lifestyle,
            "photo_url": pet.photo_url}


def _product_out(product: Product, pet: Pet | None = None) -> dict[str, Any]:
    value = {"id": product.id, "title": product.title, "brand": product.brand,
             "category": product.category, "species": product.species,
             "price": _float(product.price), "image_url": product.image_url,
             "safe": None, "danger_reason": None}
    if pet is not None:
        decision = _safety(product, pet)
        value["safe"] = decision.safe
        value["danger_reason"] = decision.reason
    return value


def _purchase_days(rows: list[Purchase]) -> tuple[list[date], int]:
    """Purchase calendar days (same-day rows merged) and the pack count of the latest day."""
    quantities: dict[date, int] = {}
    for row in rows:
        day = row.ts.date() if isinstance(row.ts, datetime) else row.ts
        quantities[day] = quantities.get(day, 0) + max(1, int(row.quantity or 1))
    days = sorted(quantities)
    return days, quantities[days[-1]]


def _search_fields(product: Product) -> dict[str, str | None]:
    return {"title": product.title, "brand": product.brand, "category": product.category,
            "description": product.description, "ingredients_text": product.ingredients_text}


def _safety(product: Product, pet: Pet) -> SafetyDecision:
    """Deterministic Safe for one product and pet; the only source of the allergen badge."""
    return default_evaluator().evaluate(product.ingredients_text, pet.allergens or [],
                                        pet_species=pet.species, product_species=product.species)


class APIService(TokenAuthMixin):
    """CRUD/query service with short-lived recommendation and pipeline caches."""

    def __init__(self, session_factory: sessionmaker = SessionLocal, llm_client: LLMClient | None = None,
                 llm_runtime: LLMRuntime | None = None, embeddings: EmbeddingService | None = None):
        self.session_factory = session_factory
        self.embeddings = embeddings or EmbeddingService()
        if llm_runtime is None:
            llm_runtime = (LLMRuntime.from_client(llm_client) if llm_client is not None
                           else LLMClientFactory(get_settings().llm).build_runtime())
        self.llm_runtime = llm_runtime
        # Chat agent and rerank/reminder texts use separate clients with separate budgets.
        self.llm_client = llm_runtime.chat
        self.rerank_client = llm_runtime.rerank
        self._pipeline_cache: dict[tuple[str, str], tuple[float, RecommendationPipeline]] = {}
        self._recommendation_cache: dict[tuple[str, str, int, bool], tuple[float, list[dict[str, Any]]]] = {}
        self._cache_lock = threading.RLock()

    def llm_status(self) -> dict[str, Any]:
        return self.llm_runtime.status()

    def _timed(self, step: str, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            logger.info("step=%s latency_ms=%.1f", step, (time.perf_counter() - started) * 1000)

    def health(self) -> None:
        with self.session_factory() as session:
            session.execute(text("SELECT 1"))

    def stats(self, user_id: str | None = None, pet_id: str | None = None) -> dict[str, int]:
        """Return live catalog composition coverage and populated pet profile dimensions."""
        with self.session_factory() as session:
            composition_count = int(session.scalar(select(func.count(Product.id)).where(
                Product.ingredients_text.is_not(None), Product.ingredients_text != "")) or 0)
            characteristics_count = 0
            if user_id and pet_id:
                uid, pid = _uid(user_id), _uid(pet_id)
                pet = session.get(Pet, pid)
                if pet is None or pet.user_id != uid:
                    raise LookupError("Pet not found for user")
                values = (pet.species, pet.birth_date, pet.breed, pet.weight_kg,
                          pet.allergens, pet.diet_restrictions, pet.lifestyle)
                characteristics_count = sum(bool(value) for value in values)
            return {"characteristics_count": characteristics_count,
                    "composition_product_count": composition_count}

    def list_users(self) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.scalars(select(User).where(User.is_admin.is_(False)).order_by(User.display_name, User.external_id)).all()
            return [_user_out(row) for row in rows]


    @staticmethod
    def _password_hash(password: str, salt: bytes | None = None) -> str:
        return password_hasher.hash(password, salt)

    @staticmethod
    def _password_matches(password: str, encoded: str | None) -> bool:
        return password_hasher.verify(password, encoded)

    def register_user(self, email: str, password: str, display_name: str) -> dict[str, Any]:
        normalized_email = email.strip().casefold()
        if "@" not in normalized_email or "." not in normalized_email.rsplit("@", 1)[-1]:
            raise ValueError("Вкажіть коректну електронну адресу.")
        with self.session_factory() as session:
            if session.scalar(select(User).where(User.email == normalized_email)):
                raise FileExistsError("Обліковий запис із цією електронною адресою вже існує.")
            user = User(email=normalized_email, password_hash=self._password_hash(password),
                        display_name=display_name.strip())
            session.add(user)
            session.commit()
            session.refresh(user)
            return _user_out(user)

    def login_user(self, email: str, password: str) -> dict[str, Any] | None:
        normalized_email = email.strip().casefold()
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.email == normalized_email))
            if user is None or not self._password_matches(password, user.password_hash):
                return None
            return _user_out(user)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        try:
            key = _uid(user_id)
        except LookupError:
            return None
        with self.session_factory() as session:
            user = session.get(User, key)
            return _user_out(user) if user else None

    def get_pet_profile(self, user_id: str, pet_id: str) -> dict[str, Any]:
        uid, pid = _uid(user_id), _uid(pet_id)
        with self.session_factory() as session:
            user, pet = session.get(User, uid), session.get(Pet, pid)
            if user is None or pet is None or pet.user_id != uid:
                raise LookupError("Pet profile not found for user")
            profile = _pet_out(pet)
            profile["display_name"] = user.display_name
            profile["preference_state"] = dict(user.preference_state or {})
            return profile

    def get_product_basic(self, product_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            product = session.get(Product, product_id)
            if product is None:
                return None
            return _product_out(product)

    def _require_admin(self, session: Session, token: str) -> User:
        try:
            user_id = self._verify_access_token(token.removeprefix("Bearer ").strip())
        except ValueError as exc:
            raise ValueError("Потрібно увійти в обліковий запис адміністратора.") from exc
        user = session.get(User, _uid(user_id))
        if user is None or not user.is_admin:
            raise PermissionError("Для керування каталогом потрібен профіль адміністратора.")
        return user

    def admin_create_product(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.session_factory() as session:
            self._require_admin(session, token)
            if session.get(Product, payload["id"]) is not None:
                raise FileExistsError("Товар із таким ID уже існує.")
            product = Product(**payload)
            session.add(product)
            session.commit()
            session.refresh(product)
            self._invalidate_all_recommendations()
            return _product_out(product)

    def admin_update_product(self, token: str, product_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.session_factory() as session:
            self._require_admin(session, token)
            product = session.get(Product, product_id)
            if product is None:
                raise LookupError("Product not found")
            for field, value in payload.items():
                setattr(product, field, value)
            session.commit()
            session.refresh(product)
            self._invalidate_all_recommendations()
            return _product_out(product)

    def _invalidate_all_recommendations(self) -> None:
        with self._cache_lock:
            self._recommendation_cache.clear()
            self._pipeline_cache.clear()

    def check_safety(self, product_id: str, pet_id: str) -> dict[str, Any]:
        """The sole agent-facing source of an allergen safety decision."""
        with self.session_factory() as session:
            product = session.get(Product, product_id)
            pet = session.get(Pet, _uid(pet_id))
            if product is None or pet is None:
                return {"product_id": product_id, "safe": False, "reason": "Товар або тварину не знайдено."}
            decision = _safety(product, pet)
            return {"product_id": product.id, "safe": decision.safe, "reason": decision.reason}

    def chat(self, user_id: str, pet_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        from .agent import PetMindAgent
        return PetMindAgent(self, self.llm_client).run(user_id, pet_id, messages)

    def list_pets(self, user_id: str) -> list[dict[str, Any]]:
        key = _uid(user_id)
        with self.session_factory() as session:
            if session.get(User, key) is None:
                raise LookupError("User not found")
            pets = session.scalars(select(Pet).where(Pet.user_id == key).order_by(Pet.name)).all()
            return [_pet_out(pet) for pet in pets]

    def get_pet(self, pet_id: str) -> dict[str, Any] | None:
        try:
            key = _uid(pet_id)
        except LookupError:
            return None
        with self.session_factory() as session:
            pet = session.get(Pet, key)
            return _pet_out(pet) if pet else None

    def create_pet(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = _uid(user_id)
        with self.session_factory() as session:
            if session.get(User, key) is None:
                raise LookupError("User not found")
            if payload.get("photo_url") is not None:
                payload["photo_url"] = str(payload["photo_url"])
            pet = Pet(user_id=key, **payload)
            session.add(pet)
            session.commit()
            session.refresh(pet)
            self._invalidate_user(str(key))
            return _pet_out(pet)

    def update_pet(self, pet_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = _uid(pet_id)
        with self.session_factory() as session:
            pet = session.get(Pet, key)
            if pet is None:
                raise LookupError("Pet not found")
            if payload.get("photo_url") is not None:
                payload["photo_url"] = str(payload["photo_url"])
            for name, value in payload.items():
                setattr(pet, name, value)
            session.commit()
            session.refresh(pet)
            self._invalidate_user(str(pet.user_id))
            return _pet_out(pet)

    def list_products(self, *, species: str | None, category: str | None, brand: str | None,
                      q: str | None, sort: str, page: int, pet_id: str | None) -> dict[str, Any]:
        stmt = select(Product)
        if species:
            stmt = stmt.where(Product.species == species)
        if category:
            stmt = stmt.where(Product.category.ilike(f"%{category}%"))
        if brand:
            stmt = stmt.where(Product.brand.ilike(f"%{brand}%"))
        query = SearchQuery.parse(q)
        ordering = {"price_asc": Product.price.asc(), "price_desc": Product.price.desc(),
                    "title": Product.title.asc(), "relevance": Product.title.asc()}
        if sort not in ordering:
            raise ValueError("Unsupported product sort")
        base_stmt = stmt
        stmt = base_stmt.order_by(ordering[sort], Product.id).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
        with self.session_factory() as session:
            pet = self._pet_for_product_query(session, pet_id) if pet_id else None
            if not query.empty:
                # Token search ranks in Python; the catalog is small enough to score every row.
                rows = session.scalars(base_stmt.order_by(ordering[sort], Product.id)).all()
                by_id = {row.id: row for row in rows}
                ranked = [by_id[row_id] for row_id, _ in rank(query, ((r.id, _search_fields(r)) for r in rows))]
                if sort != "relevance":
                    order = {row.id: index for index, row in enumerate(rows)}
                    ranked.sort(key=lambda row: order[row.id])
                window = ranked[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
                return {"items": [_product_out(item, pet) for item in window], "total": len(ranked),
                        "page": page, "page_size": PAGE_SIZE}
            total_stmt = select(func.count()).select_from(base_stmt.order_by(None).subquery())
            total = int(session.scalar(total_stmt) or 0)
            items = session.scalars(stmt).all()
            return {"items": [_product_out(item, pet) for item in items], "total": total,
                    "page": page, "page_size": PAGE_SIZE}

    def search_catalog(self, query: str, *, species: str | None, limit: int = 10) -> list[dict[str, Any]]:
        """Agent search: token-ranked products of one species, with a relevance score."""
        parsed = SearchQuery.parse(query)
        with self.session_factory() as session:
            stmt = select(Product)
            if species:
                stmt = stmt.where(Product.species == species)
            rows = session.scalars(stmt).all()
            by_id = {row.id: row for row in rows}
            if parsed.empty:
                return [{**_product_out(row), "score": 0.0} for row in rows[:limit]]
            return [{**_product_out(by_id[row_id]), "score": round(score, 3)}
                    for row_id, score in rank(parsed, ((r.id, _search_fields(r)) for r in rows), limit)]

    @staticmethod
    def _pet_for_product_query(session: Session, pet_id: str) -> Pet:
        pet = session.get(Pet, _uid(pet_id))
        if pet is None:
            raise LookupError("Pet not found")
        return pet

    def get_product(self, product_id: str, *, pet_id: str | None = None) -> dict[str, Any] | None:
        with self.session_factory() as session:
            product = session.get(Product, product_id)
            if product is None:
                return None
            pet = self._pet_for_product_query(session, pet_id) if pet_id else None
            evaluator = default_evaluator()
            ingredients = evaluator.items(product.ingredients_text)
            comp = evaluator.composition(product.ingredients_text, pet.allergens if pet else [],
                nutrients={"protein": _float(product.protein), "fat": _float(product.fat)},
                moisture=_float(product.moisture) if product.moisture is not None else None,
                species=pet.species if pet else None)
            ingredient_rows = []
            for index, ingredient in enumerate(ingredients):
                canonical = ingredient.canonical_name if hasattr(ingredient, "canonical_name") else str(ingredient)
                group = ingredient.allergen_group if hasattr(ingredient, "allergen_group") else None
                highlighted = bool(pet and evaluator.is_allergen(ingredient, pet.allergens or []))
                ingredient_rows.append({"position": index, "name": str(ingredient.variant if hasattr(ingredient, "variant") else ingredient),
                    "canonical_name": canonical, "allergen_group": group, "is_allergen": highlighted})
            danger = _safety(product, pet).reason if pet else None
            rag_chunks = []
            if product.embedding is not None:
                raw = engine.raw_connection()
                try:
                    passages = self._timed("rag_search", search_pg_chunks, raw, product.embedding, 5, product.id)
                    rag_chunks = [{"id": p.id, "text": p.text, "score": p.score} for p in passages]
                finally:
                    raw.close()
            result = _product_out(product, pet)
            result.update({"description": product.description, "ingredients_text": product.ingredients_text,
                "ingredients": ingredient_rows, "S_nutr": comp["S_nutr"], "Q_ingr": comp["Q_ingr"],
                "S_comp": comp["S_comp"], "nutrient_data_missing": comp["nutrient_data_missing"],
                "rag_chunks": rag_chunks})
            result["danger_reason"] = danger
            return result

    def _get_pipeline(self, user_id: uuid.UUID, pet: Pet, session: Session) -> RecommendationPipeline:
        key = (str(user_id), str(pet.id))
        now = time.monotonic()
        cached = self._pipeline_cache.get(key)
        if cached and cached[0] > now:
            return cached[1]
        user = session.get(User, user_id)
        # All users' purchases feed collaborative filtering (BUG-025); ids stay opaque UUIDs.
        purchases = session.scalars(select(Purchase).order_by(Purchase.ts.desc()).limit(MAX_CF_INTERACTIONS)).all()
        products_db = session.scalars(select(Product)).all()
        self._timed("embedding_backfill", self.embeddings.ensure_product_vectors, session, products_db)
        evaluator = default_evaluator()
        product_rows = []
        item_vectors = {}
        for product in products_db:
            # Only goods for this pet's species enter the pipeline; the core filter covers cats and dogs only.
            if product.species != pet.species:
                continue
            product_rows.append({"parent_asin": product.id, "title": product.title, "brand": product.brand,
                "categories": product.category, "description": product.description, "species": product.species,
                "ingredients_text": evaluator.normalizer.canonical_text(product.ingredients_text), "proteins_100g": product.protein,
                "fat_100g": product.fat, "moisture_100g": product.moisture,
                "energy-kcal_100g": product.kcal_100g,
                "package_weight_kg": product.package_weight_kg,
                "energy_kcal_per_kg": product.energy_kcal_per_kg,
                "price": _float(product.price), "image_url": product.image_url})
            if product.embedding is not None:
                item_vectors[product.id] = np.asarray(product.embedding, dtype=np.float32)
        interactions = pd.DataFrame([{"user_id": str(purchase.user_id), "parent_asin": purchase.product_id,
            "rating": purchase.rating if purchase.rating is not None else 5.0,
            "timestamp": _iso_days(purchase.ts)} for purchase in purchases],
            columns=["user_id", "parent_asin", "rating", "timestamp"])
        profile = pd.DataFrame([{"user_id": str(user_id), "pet_species": pet.species,
            "age_group": self._age_group(pet.birth_date, pet.species), "allergen_groups": evaluator.allergen_groups(pet.allergens or []),
            "lifestyle": pet.lifestyle, "diet_restrictions": pet.diet_restrictions or [],
            "preference_state": user.preference_state if user else {}}])
        products_frame = pd.DataFrame(product_rows)
        bound_engine = self.session_factory.kw.get("bind", engine)
        rag_index = RAGIndex(store=_DatabaseKBStore(bound_engine), embedder=self.embeddings.rag_embedder())
        pipeline = self._timed("pipeline_build", RecommendationPipeline, interactions, products_frame,
            profile, llm_client=self.rerank_client, item_embeddings=item_vectors, rag_index=rag_index,
            build_rag_index=False, product_embedder=self.embeddings.product_embedder)
        self._pipeline_cache[key] = (now + get_settings().recommendation_ttl_seconds, pipeline)
        return pipeline

    @staticmethod
    def _age_group(birth_date: date | None, species: str) -> str:
        if birth_date is None:
            return "adult"
        years = (date.today() - birth_date).days / 365.2425
        if years < 1:
            return {"cat": "kitten", "dog": "puppy"}.get(species, "young")
        return "senior" if years >= 8 else "adult"

    def recommend(self, user_id: str, pet_id: str, *, k: int = 10, use_llm: bool = True) -> list[dict[str, Any]]:
        cache_key = (str(user_id), str(pet_id), k, use_llm)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._recommendation_cache.get(cache_key)
            if cached and cached[0] > now:
                logger.info("step=recommendation_cache latency_ms=0.0 hit=true")
                return cached[1]
        uid, pid = _uid(user_id), _uid(pet_id)
        with self.session_factory() as session:
            pet = session.get(Pet, pid)
            if pet is None or pet.user_id != uid:
                raise LookupError("Pet not found for user")
            pipeline = self._get_pipeline(uid, pet, session)
        started = time.perf_counter()
        try:
            results = pipeline.recommend(str(uid), use_llm=use_llm)
        except Exception:
            logger.exception("Recommendation pipeline failed; retrying deterministic fallback")
            results = pipeline.recommend(str(uid), use_llm=False)
        finally:
            logger.info("step=recommendation_pipeline latency_ms=%.1f", (time.perf_counter() - started) * 1000)
        output = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item.dict() for item in results[:k]]
        with self._cache_lock:
            self._recommendation_cache[cache_key] = (time.monotonic() + get_settings().recommendation_ttl_seconds, output)
        return output

    def list_reminders(self, user_id: str, pet_id: str) -> list[dict[str, Any]]:
        uid, pid = _uid(user_id), _uid(pet_id)
        with self.session_factory() as session:
            pet = session.get(Pet, pid)
            if pet is None or pet.user_id != uid:
                raise LookupError("Pet not found for user")
            reminders = session.scalars(select(Reminder).where(
                Reminder.user_id == uid, Reminder.pet_id == pid, Reminder.status == "new"
            ).order_by(Reminder.priority.desc(), Reminder.expected_date)).all()
            return [{"item_id": row.product_id, "title": row.product.title, "days_left": row.days_left,
                "expected_date": row.expected_date, "text": row.text, "severity": row.severity,
                "status": row.status, "priority": row.priority} for row in reminders]

    def record_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        uid = _uid(payload["user_id"])
        pid = _uid(payload["pet_id"]) if payload.get("pet_id") else None
        with self.session_factory() as session:
            user, product = session.get(User, uid), session.get(Product, payload["product_id"])
            if user is None or product is None:
                raise LookupError("User or product not found")
            if pid is not None:
                pet = session.get(Pet, pid)
                if pet is None or pet.user_id != uid:
                    raise LookupError("Pet not found for user")
            event = Event(user_id=uid, pet_id=pid, product_id=product.id, type=payload["type"], ts=datetime.now(timezone.utc))
            if payload["type"] == "dismiss":
                dismissed = session.scalars(select(Reminder).where(
                    Reminder.user_id == uid, Reminder.product_id == product.id, Reminder.status == "new"
                )).all()
                for reminder in dismissed:
                    reminder.status = "dismissed"
            user.preference_state = update_preference_state(user.preference_state, payload["type"], product.id,
                                                             category=product.category)
            session.add(event)
            session.commit()
            session.refresh(event)
            self._invalidate_user(str(uid))
            return {"id": str(event.id), "user_id": str(uid), "pet_id": str(pid) if pid else None,
                    "product_id": product.id, "type": event.type, "ts": event.ts}

    def record_purchases(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        uid = _uid(payload["user_id"])
        pid = _uid(payload["pet_id"]) if payload.get("pet_id") else None
        now = datetime.now(timezone.utc)
        with self.session_factory() as session:
            user = session.get(User, uid)
            if user is None:
                raise LookupError("User not found")
            outputs = []
            grouped_lines: dict[str, dict[str, Any]] = {}
            for line in payload["items"]:
                product_key = str(line["product_id"])
                line_pet_id = _uid(line["pet_id"]) if line.get("pet_id") else pid
                grouped = grouped_lines.setdefault(product_key, {"product_id": product_key, "pet_id": line_pet_id, "quantity": 0,
                                                                  "rating": line.get("rating")})
                if grouped["pet_id"] != line_pet_id:
                    raise ValueError("Один товар у кошику не можна прив’язати до кількох тварин.")
                grouped["quantity"] += int(line.get("quantity", 1))
            for line in grouped_lines.values():
                product = session.get(Product, line["product_id"])
                if product is None:
                    raise LookupError(f"Product {line['product_id']} not found")
                item_pet_id = line["pet_id"]
                item_pet = session.get(Pet, item_pet_id) if item_pet_id is not None else None
                if item_pet_id is not None and (item_pet is None or item_pet.user_id != uid):
                    raise LookupError("Обрана тварина не належить цьому обліковому запису.")
                decision = _safety(product, item_pet) if item_pet is not None else None
                if decision is not None and not decision.safe:
                    explanation = (decision.reason or "").rstrip(".")
                    raise ValueError(f"Не можна оформити «{product.title}» для {item_pet.name}: {explanation}. Змініть тварину для цієї позиції або вилучіть товар.")
                if item_pet_id is not None:
                    completed = session.scalars(select(Reminder).where(
                        Reminder.user_id == uid, Reminder.pet_id == item_pet_id,
                        Reminder.product_id == product.id, Reminder.status == "new"
                    )).all()
                    for reminder in completed:
                        reminder.status = "done"
                purchase = Purchase(user_id=uid, product_id=product.id, ts=now, rating=line.get("rating"),
                                    quantity=line["quantity"])
                event = Event(user_id=uid, pet_id=item_pet_id, product_id=product.id, type="purchase", ts=now)
                user.preference_state = update_preference_state(user.preference_state, "purchase", product.id,
                                                                 category=product.category)
                session.add_all([purchase, event])
                session.flush()
                outputs.append({"id": str(purchase.id), "user_id": str(uid), "product_id": product.id,
                                "ts": purchase.ts, "rating": purchase.rating})
            session.commit()
            self._invalidate_user(str(uid))
            # Reflect newly recorded purchases in replenishment forecasts immediately.
            self._timed("reminder_refresh_after_purchase", self.refresh_reminders)
            return outputs

    def _invalidate_user(self, user_id: str) -> None:
        with self._cache_lock:
            self._pipeline_cache = {key: value for key, value in self._pipeline_cache.items() if key[0] != user_id}
            self._recommendation_cache = {key: value for key, value in self._recommendation_cache.items() if key[0] != user_id}

    def refresh_reminders(self) -> int:
        """Recompute purchase-plus-stock forecasts, preserving completed reminder history."""
        from petrec.reminders import (
            CATEGORY_RULES, estimate_food_stock_days, get_reminders, predict_purchase_date,
            reminder_priority,
        )
        from .db.models import Pet
        current = date.today()
        with self.session_factory() as session:
            purchases = session.scalars(select(Purchase).order_by(Purchase.user_id, Purchase.product_id, Purchase.ts)).all()
            grouped: dict[tuple[uuid.UUID, str], list[Purchase]] = {}
            for purchase in purchases:
                grouped.setdefault((purchase.user_id, purchase.product_id), []).append(purchase)
            existing = session.scalars(select(Reminder).where(Reminder.status == "new")).all()
            active = {(row.user_id, row.pet_id, row.product_id): row for row in existing}
            candidates: dict[uuid.UUID, list[tuple[Any, ...]]] = {}
            for (uid, product_id), rows in grouped.items():
                product = session.get(Product, product_id)
                if product is None or not product.is_regular:
                    continue
                category_text = (product.category or "").lower()
                if any(token in category_text for token in ("litter", "наполн", "наповн")):
                    category_key = "litter"
                elif any(token in category_text for token in ("flea", "tick", "паразит", "бліх", "кліщ")):
                    category_key = "parasite"
                elif any(token in category_text for token in ("worm", "гельмінт")):
                    category_key = "deworming"
                elif any(token in category_text for token in ("vitamin", "supplement", "вітамін", "добав")):
                    category_key = "supplement"
                elif any(token in category_text for token in ("wet", "canned", "волог")):
                    category_key = "wet_food"
                else:
                    category_key = "dry_food"
                lead_days, importance = CATEGORY_RULES.get(category_key, CATEGORY_RULES["default"])
                # Same-day purchases are one restock of several packs (BUG-032).
                dates, last_quantity = _purchase_days(rows)
                for pet in session.scalars(select(Pet).where(Pet.user_id == uid)).all():
                    stock_days = None
                    if category_key == "dry_food" and pet.weight_kg and product.energy_kcal_per_kg and product.package_weight_kg:
                        try:
                            stock_days = estimate_food_stock_days(
                                float(pet.weight_kg), pet.species, product.energy_kcal_per_kg,
                                product.package_weight_kg, last_quantity, pet.lifestyle,
                            )
                        except ValueError:
                            logger.warning("step=reminder_stock_estimate_invalid product_id=%s pet_id=%s", product_id, pet.id)
                    expected = predict_purchase_date(dates, stock_days=stock_days, alpha=0.3)
                    if expected is None:
                        continue
                    if stock_days is None and last_quantity > 1:
                        # Without a stock estimate the interval covers one pack; n packs last n times longer.
                        expected = dates[-1] + (expected - dates[-1]) * last_quantity
                    reminder_date = expected - timedelta(days=lead_days)
                    priority = reminder_priority(expected, current, importance)
                    days_left = (expected - current).days
                    severity = "прострочено" if days_left < 0 else "сьогодні" if days_left == 0 else "скоро"
                    text_value = get_reminders(str(uid), str(pet.id), current,
                        items=[{"item_id": product_id, "title": product.title, "expected_date": expected}],
                        llm_client=self.rerank_client)[0]
                    candidates.setdefault(uid, []).append((priority, pet.id, product_id, expected,
                                                            reminder_date, severity, days_left, text_value))

            allowed: set[tuple[uuid.UUID, uuid.UUID, str]] = set()
            for uid, rows in candidates.items():
                for _, pet_id, product_id, *_ in sorted(rows, key=lambda row: (-row[0], row[3], row[2]))[:2]:
                    allowed.add((uid, pet_id, product_id))

            created = 0
            for uid, rows in candidates.items():
                for priority, pet_id, product_id, expected, reminder_date, severity, days_left, item in rows:
                    key = (uid, pet_id, product_id)
                    if key not in allowed:
                        continue
                    reminder = active.get(key)
                    if reminder is None:
                        reminder = Reminder(user_id=uid, pet_id=pet_id, product_id=product_id)
                        session.add(reminder)
                        created += 1
                    reminder.expected_date = expected
                    reminder.reminder_date = reminder_date
                    reminder.days_left = days_left
                    reminder.text = item.text
                    reminder.severity = severity
                    reminder.priority = priority
                    reminder.status = "new"
            session.commit()
            logger.info("step=reminder_refresh reminders=%d", created)
            return created

    def warm_up(self) -> None:
        """Load the embedding model and back-fill product vectors so the first request is fast."""
        self.embeddings.warm_up()
        try:
            with self.session_factory() as session:
                self.embeddings.ensure_product_vectors(session, session.scalars(select(Product)).all())
        except Exception:
            logger.exception("Product embedding back-fill failed; it will be retried on demand")

    def start_scheduler(self):
        threading.Thread(target=self.warm_up, name="petmind-warm-up", daemon=True).start()
        scheduler = BackgroundScheduler(timezone="UTC")
        try:
            self._timed("reminder_initial_refresh", self.refresh_reminders)
        except Exception:
            logger.exception("Initial reminder refresh failed; daily refresh remains scheduled")
        scheduler.add_job(self.refresh_reminders, "cron", hour=9, minute=0,
                          id="daily_petmind_reminders", replace_existing=True)
        scheduler.start()
        return scheduler


def _with_item_id(result: SearchResult) -> SearchResult:
    """pgvector rows carry `product_id`; the core pipeline attaches passages by `item_id` (BUG-026)."""
    metadata = dict(result.metadata or {})
    if "item_id" not in metadata and metadata.get("product_id") is not None:
        metadata["item_id"] = metadata["product_id"]
    return SearchResult(result.id, result.text, result.score, metadata)


class _DatabaseKBStore(VectorStore):
    """VectorStore adapter that delegates vector ranking to petrec.rag."""

    def __init__(self, database_engine):
        self.database_engine = database_engine

    def add(self, ids, texts, embeddings, metadatas=None) -> None:
        raise NotImplementedError("Knowledge chunks are populated by the seed script")

    def search(self, query_embedding, top_m: int = 5) -> list[SearchResult]:
        raw = self.database_engine.raw_connection()
        try:
            started = time.perf_counter()
            results = search_pg_chunks(raw, query_embedding, top_m)
            logger.info("step=rag_retrieval latency_ms=%.1f", (time.perf_counter() - started) * 1000)
            return [_with_item_id(result) for result in results]
        finally:
            raw.close()
