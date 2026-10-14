"""Pydantic request and response schemas for the Petly HTTP API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class UserOut(BaseModel):
    id: str
    external_id: str | None = None
    display_name: str | None = None
    is_admin: bool = False


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str


class PetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    species: str = Field(min_length=1, max_length=40)
    birth_date: date | None = None
    breed: str | None = None
    weight_kg: float | None = Field(default=None, gt=0)
    allergens: list[str] = Field(default_factory=list)
    diet_restrictions: list[str] = Field(default_factory=list)
    lifestyle: str | None = None
    photo_url: HttpUrl | None = None


class PetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    species: str | None = Field(default=None, max_length=40)
    birth_date: date | None = None
    breed: str | None = None
    weight_kg: float | None = Field(default=None, gt=0)
    allergens: list[str] | None = None
    diet_restrictions: list[str] | None = None
    lifestyle: str | None = None
    photo_url: HttpUrl | None = None


class PetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    user_id: str
    name: str
    species: str
    birth_date: date | None = None
    breed: str | None = None
    weight_kg: float | None = None
    allergens: list[str]
    diet_restrictions: list[str]
    lifestyle: str | None = None
    photo_url: str | None = None


class ProductListItem(BaseModel):
    id: str
    title: str
    brand: str | None = None
    category: str | None = None
    species: str | None = None
    price: float | None = None
    image_url: str | None = None
    safe: bool | None = None
    danger_reason: str | None = None


class ProductPage(BaseModel):
    items: list[ProductListItem]
    total: int
    page: int
    page_size: int


class AdminProductCreate(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1)
    brand: str | None = None
    category: str | None = None
    species: str | None = Field(default=None, max_length=40)
    price: float | None = Field(default=None, ge=0)
    image_url: str | None = None
    description: str | None = None
    ingredients_text: str | None = None
    protein: float | None = Field(default=None, ge=0)
    fat: float | None = Field(default=None, ge=0)


class AdminProductUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    brand: str | None = None
    category: str | None = None
    species: Literal["cat", "dog"] | None = None
    price: float | None = Field(default=None, ge=0)
    image_url: str | None = None
    description: str | None = None
    ingredients_text: str | None = None
    protein: float | None = Field(default=None, ge=0)
    fat: float | None = Field(default=None, ge=0)


class IngredientAnalysis(BaseModel):
    position: int
    name: str
    canonical_name: str
    allergen_group: str | None = None
    is_allergen: bool


class ProductDetail(ProductListItem):
    description: str | None = None
    ingredients_text: str | None = None
    ingredients: list[IngredientAnalysis]
    S_nutr: float
    Q_ingr: float
    S_comp: float
    nutrient_data_missing: bool
    rag_chunks: list[dict[str, Any]]


class RecommendationItem(BaseModel):
    item_id: str
    title: str
    brand: str = ""
    price: float | None = None
    image_url: str | None = None
    match_percent: int
    reason: str
    safe_badge: bool
    composition_summary: list[str]
    llm_used: bool


class ReminderOut(BaseModel):
    item_id: str
    title: str
    days_left: int
    expected_date: date
    text: str
    severity: Literal["скоро", "сьогодні", "прострочено"]
    status: Literal["new", "done", "dismissed"] = "new"
    priority: float = 0.0


class EventCreate(BaseModel):
    user_id: str
    pet_id: str | None = None
    product_id: str
    type: Literal["view", "add_to_cart", "dismiss", "like"]


class PurchaseLine(BaseModel):
    product_id: str
    pet_id: str | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    quantity: int = Field(default=1, ge=1, le=100)


class PurchaseCreate(BaseModel):
    user_id: str
    pet_id: str | None = None
    items: list[PurchaseLine] = Field(min_length=1)


class EventOut(BaseModel):
    id: str
    user_id: str
    pet_id: str | None
    product_id: str | None
    type: str
    ts: datetime


class PurchaseOut(BaseModel):
    id: str
    user_id: str
    product_id: str
    ts: datetime
    rating: float | None


class HealthOut(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]


class StatsOut(BaseModel):
    characteristics_count: int
    composition_product_count: int


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    user_id: str
    pet_id: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)


class ChatResponse(BaseModel):
    reply: str
    products: list[RecommendationItem]
    tool_trace: list[dict[str, Any]]
