"""Route-contract tests use FastAPI TestClient and an injected MockLLM service."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from petrec.llm import MockLLM
from petrec.rerank import RecommendationItem
from app.auth import TokenAuthMixin
from app.main import create_app


UID = "4a3f3f1c-6180-4b73-b1f0-0cbb6d628241"
PID = "3e2105a7-44c8-4ea2-9cac-1266fd90a150"
PRODUCT = "B00TEST001"


class FakeService(TokenAuthMixin):
    def __init__(self):
        self.llm = MockLLM("Підходить за складом і потребами.")
        self.created_event = None

    def health(self): pass
    def list_users(self): return [{"id": UID, "external_id": "demo", "display_name": "Демо"}]
    def get_user(self, user_id): return self.list_users()[0] if user_id == UID else None
    def get_pet(self, pet_id): return self.list_pets(UID)[0] if pet_id == PID else None
    def list_pets(self, user_id): return [{"id": PID, "user_id": UID, "name": "Мурчик", "species": "cat", "birth_date": date(2020, 1, 1), "breed": None, "weight_kg": 4.5, "allergens": ["птиця"], "diet_restrictions": [], "lifestyle": "стерилізований", "photo_url": None}]
    def create_pet(self, user_id, payload): return {"id": PID, "user_id": user_id, **payload}
    def update_pet(self, pet_id, payload): return {**self.list_pets(UID)[0], **payload}
    def list_products(self, **kwargs): return {"items": [{"id": PRODUCT, "title": "Корм", "brand": "Petly", "category": "Корм", "species": "cat", "price": 99.0, "image_url": None, "safe": True if kwargs.get("pet_id") else None, "danger_reason": None}], "total": 1, "page": kwargs["page"], "page_size": 24}
    def get_product(self, product_id, *, pet_id=None):
        return {"id": PRODUCT, "title": "Корм", "brand": "Petly", "category": "Корм", "species": "cat", "price": 99.0, "image_url": None, "safe": True, "danger_reason": None, "description": "Повний опис", "ingredients_text": "рис, лосось", "ingredients": [{"position": 0, "name": "рис", "canonical_name": "рис", "allergen_group": None, "is_allergen": False}], "S_nutr": 0.9, "Q_ingr": 0.8, "S_comp": 0.85, "nutrient_data_missing": False, "rag_chunks": [{"id": "chunk-1", "text": "Опис товару", "score": 0.9}]}
    def recommend(self, user_id, pet_id, *, k=10, use_llm=True):
        if use_llm:
            reason = self.llm.complete("system", "user")
        else:
            reason = "Сумісність складу: 85%."
        item = RecommendationItem(item_id=PRODUCT, title="Корм", brand="Petly", price=99, image_url=None,
            match_percent=85, reason=reason, safe_badge=True, composition_summary=["рис", "лосось"], llm_used=use_llm)
        return [item.model_dump(mode="json")]
    def list_reminders(self, user_id, pet_id): return [{"item_id": PRODUCT, "title": "Корм", "days_left": 0, "expected_date": date(2026, 10, 5), "text": "Сьогодні поповніть запас.", "severity": "сьогодні"}]
    def record_event(self, payload):
        self.created_event = payload
        return {"id": "81b9e329-5482-4e1c-8c0c-751cc381066e", "user_id": payload["user_id"], "pet_id": payload["pet_id"], "product_id": payload["product_id"], "type": payload["type"], "ts": datetime.now(timezone.utc)}
    def record_purchases(self, payload): return [{"id": "81b9e329-5482-4e1c-8c0c-751cc381066e", "user_id": payload["user_id"], "product_id": payload["items"][0]["product_id"], "ts": datetime.now(timezone.utc), "rating": payload["items"][0].get("rating")}]


@pytest.fixture
def client():
    service = FakeService()
    with TestClient(create_app(service, enable_scheduler=False)) as test_client:
        test_client.service = service
        test_client.headers["Authorization"] = f"Bearer {service.issue_access_token(UID)}"
        yield test_client


def test_private_routes_require_valid_token(client):
    del client.headers["Authorization"]
    assert client.get("/api/users").status_code == 401
    assert client.get("/api/users", headers={"Authorization": "Bearer forged.token"}).status_code == 401
    other = client.service.issue_access_token("2b0c7f2e-0000-4000-8000-000000000000")
    assert client.get(f"/api/users/{UID}/pets", headers={"Authorization": f"Bearer {other}"}).status_code == 401


def test_users_and_pet_questionnaire_routes(client):
    assert client.get("/api/users").status_code == 200
    assert client.get(f"/api/users/{UID}").json()["display_name"] == "Демо"
    assert client.get(f"/api/users/{UID}/pets").json()[0]["name"] == "Мурчик"
    created = client.post(f"/api/users/{UID}/pets", json={"name": "Річі", "species": "dog", "weight_kg": 31})
    assert created.status_code == 201 and created.json()["species"] == "dog"
    updated = client.put(f"/api/pets/{PID}", json={"weight_kg": 4.7})
    assert updated.status_code == 200 and updated.json()["weight_kg"] == 4.7


def test_product_list_and_detail_routes(client):
    listed = client.get("/api/products?species=cat&category=food&brand=Petly&q=korm&sort=price_asc&page=1&pet_id=" + PID)
    assert listed.status_code == 200 and listed.json()["items"][0]["safe"] is True
    detail = client.get(f"/api/products/{PRODUCT}?pet_id={PID}")
    assert detail.status_code == 200
    assert {"ingredients", "S_nutr", "Q_ingr", "S_comp", "rag_chunks"}.issubset(detail.json())


def test_recommendations_reminders_events_and_purchases(client):
    recs = client.get(f"/api/recommendations?user_id={UID}&pet_id={PID}&k=10&use_llm=true")
    assert recs.status_code == 200 and recs.json()[0]["llm_used"] is True
    assert client.service.llm.calls
    reminders = client.get(f"/api/reminders?user_id={UID}&pet_id={PID}")
    assert reminders.status_code == 200 and reminders.json()[0]["severity"] == "сьогодні"
    event = client.post("/api/events", json={"user_id": UID, "pet_id": PID, "product_id": PRODUCT, "type": "like"})
    assert event.status_code == 201 and client.service.created_event["type"] == "like"
    purchases = client.post("/api/purchases", json={"user_id": UID, "pet_id": PID, "items": [{"product_id": PRODUCT, "rating": 5}]})
    assert purchases.status_code == 201 and purchases.json()[0]["product_id"] == PRODUCT


def test_health_cors_and_openapi_schemas(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok", "database": "ok"}
    preflight = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
    spec = client.get("/openapi.json").json()
    expected_paths = {"/api/users", "/api/users/{id}", "/api/users/{id}/pets", "/api/pets/{id}",
        "/api/products", "/api/products/{id}", "/api/recommendations", "/api/reminders",
        "/api/events", "/api/purchases", "/api/health"}
    assert expected_paths.issubset(spec["paths"])
    schemas = spec["components"]["schemas"]
    assert {"PetCreate", "PetUpdate", "PetOut", "ProductPage", "ProductDetail",
            "RecommendationItem", "ReminderOut", "EventCreate", "PurchaseCreate", "HealthOut"}.issubset(schemas)


def test_llm_health_endpoint_has_no_secrets(client):
    client.service.llm_status = lambda: {"provider": "gemini", "state": "not_configured", "detail": "GEMINI_API_KEY is not set"}
    body = client.get("/api/health/llm").json()
    assert body["provider"] == "gemini" and body["state"] == "not_configured"
