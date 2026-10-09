"""SQLAlchemy 2 persistence models for Petly."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, CheckConstraint, Date, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(256))
    is_admin: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    preference_state: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    pets: Mapped[list[Pet]] = relationship(back_populates="user", cascade="all, delete-orphan")
    purchases: Mapped[list[Purchase]] = relationship(back_populates="user", cascade="all, delete-orphan")
    events: Mapped[list[Event]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Pet(Base):
    __tablename__ = "pets"
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    species: Mapped[str] = mapped_column(String(40), nullable=False)
    birth_date: Mapped[date | None] = mapped_column(Date)
    breed: Mapped[str | None] = mapped_column(String(160))
    weight_kg: Mapped[float | None] = mapped_column(Numeric(7, 3))
    allergens: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    diet_restrictions: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    lifestyle: Mapped[str | None] = mapped_column(Text)
    photo_url: Mapped[str | None] = mapped_column(Text)
    user: Mapped[User] = relationship(back_populates="pets")
    reminders: Mapped[list[Reminder]] = relationship(back_populates="pet", cascade="all, delete-orphan")
    events: Mapped[list[Event]] = relationship(back_populates="pet")


class Product(Base):
    __tablename__ = "products"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # parent_asin
    title: Mapped[str] = mapped_column(Text, nullable=False)
    brand: Mapped[str | None] = mapped_column(String(240))
    category: Mapped[str | None] = mapped_column(Text)
    species: Mapped[str | None] = mapped_column(String(40))
    price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    image_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    ingredients_text: Mapped[str | None] = mapped_column(Text)
    protein: Mapped[float | None] = mapped_column(Float)
    fat: Mapped[float | None] = mapped_column(Float)
    moisture: Mapped[float | None] = mapped_column(Float)
    kcal_100g: Mapped[float | None] = mapped_column(Float)
    is_regular: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    package_weight_kg: Mapped[float | None] = mapped_column(Float)
    energy_kcal_per_kg: Mapped[float | None] = mapped_column(Float)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    ingredients: Mapped[list[ProductIngredient]] = relationship(back_populates="product", cascade="all, delete-orphan", order_by="ProductIngredient.position")
    purchases: Mapped[list[Purchase]] = relationship(back_populates="product")
    chunks: Mapped[list[KBChunk]] = relationship(back_populates="product", cascade="all, delete-orphan")
    reminders: Mapped[list[Reminder]] = relationship(back_populates="product")
    events: Mapped[list[Event]] = relationship(back_populates="product")


class ProductIngredient(Base):
    __tablename__ = "product_ingredients"
    __table_args__ = (UniqueConstraint("product_id", "position", name="uq_product_ingredient_position"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str | None] = mapped_column(Text)
    allergen_group: Mapped[str | None] = mapped_column(Text)
    theta: Mapped[float | None] = mapped_column(Float)
    product: Mapped[Product] = relationship(back_populates="ingredients")


class Purchase(Base):
    __tablename__ = "purchases"
    __table_args__ = (UniqueConstraint("user_id", "product_id", "ts", name="uq_purchase_user_product_ts"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    rating: Mapped[float | None] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    user: Mapped[User] = relationship(back_populates="purchases")
    product: Mapped[Product] = relationship(back_populates="purchases")


class KBChunk(Base):
    __tablename__ = "kb_chunks"
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    product: Mapped[Product | None] = relationship(back_populates="chunks")


class Reminder(Base):
    __tablename__ = "reminders"
    __table_args__ = (CheckConstraint("severity IN ('скоро','сьогодні','прострочено')", name="ck_reminders_severity"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    pet_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pets.id", ondelete="CASCADE"), index=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    expected_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    days_left: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(24), nullable=False)
    priority: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="new", server_default="new")
    reminder_date: Mapped[date | None] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    pet: Mapped[Pet] = relationship(back_populates="reminders")
    product: Mapped[Product] = relationship(back_populates="reminders")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (CheckConstraint("type IN ('view','add_to_cart','purchase','dismiss','like')", name="ck_events_type"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    pet_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("pets.id", ondelete="SET NULL"), index=True)
    product_id: Mapped[str | None] = mapped_column(ForeignKey("products.id", ondelete="SET NULL"), index=True)
    type: Mapped[str] = mapped_column(String(24), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    user: Mapped[User] = relationship(back_populates="events")
    pet: Mapped[Pet | None] = relationship(back_populates="events")
    product: Mapped[Product | None] = relationship(back_populates="events")
