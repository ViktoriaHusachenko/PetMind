"""FastAPI application for PetMind."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AdminProductCreate, AdminProductUpdate, AuthResponse, ChatRequest, ChatResponse, EventCreate, EventOut, HealthOut, LoginRequest, PetCreate, PetOut, PetUpdate, ProductDetail,
    ProductListItem, ProductPage, PurchaseCreate, PurchaseOut, RecommendationItem, RegisterRequest, ReminderOut, StatsOut, UserOut,
)

logger = logging.getLogger("petly.api")


def create_app(service: Any | None = None, *, enable_scheduler: bool = True) -> FastAPI:
    if service is None:
        from .service import APIService
        service = APIService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = service.start_scheduler() if enable_scheduler and hasattr(service, "start_scheduler") else None
        app.state.scheduler = scheduler
        yield
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    app = FastAPI(title="PetMind API", version="1.0.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-User-Id"],
    )

    @app.middleware("http")
    async def log_request_latency(request: Request, call_next):
        started = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            logger.info("step=http_%s latency_ms=%.1f", request.url.path,
                        (time.perf_counter() - started) * 1000)

    from fastapi import APIRouter
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthOut)
    def health(request: Request):
        try:
            request.app.state.service.health()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        return {"status": "ok", "database": "ok"}

    @router.get("/stats", response_model=StatsOut)
    def stats(request: Request, user_id: str | None = None, pet_id: str | None = None):
        try:
            return request.app.state.service.stats(user_id, pet_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/users", response_model=list[UserOut])
    def users(request: Request):
        return request.app.state.service.list_users()

    @router.post("/auth/register", response_model=AuthResponse, status_code=201)
    def register(payload: RegisterRequest, request: Request):
        try:
            service = request.app.state.service
            user = service.register_user(payload.email, payload.password, payload.display_name)
            return {"user": user, "access_token": service.issue_access_token(user["id"])}
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/auth/login", response_model=AuthResponse)
    def login(payload: LoginRequest, request: Request):
        user = request.app.state.service.login_user(payload.email, payload.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Неправильна електронна адреса або пароль.")
        return {"user": user, "access_token": request.app.state.service.issue_access_token(user["id"])}

    @router.get("/users/{id}", response_model=UserOut)
    def user(id: str, request: Request):
        result = request.app.state.service.get_user(id)
        if result is None:
            raise HTTPException(status_code=404, detail="User not found")
        return result

    @router.get("/users/{id}/pets", response_model=list[PetOut])
    def pets(id: str, request: Request):
        return request.app.state.service.list_pets(id)

    @router.post("/users/{id}/pets", response_model=PetOut, status_code=201)
    def create_pet(id: str, payload: PetCreate, request: Request):
        try:
            return request.app.state.service.create_pet(id, payload.model_dump(mode="json"))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.put("/pets/{id}", response_model=PetOut)
    def update_pet(id: str, payload: PetUpdate, request: Request):
        try:
            return request.app.state.service.update_pet(id, payload.model_dump(mode="json", exclude_unset=True))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/products", response_model=ProductPage)
    def products(
        request: Request,
        species: str | None = None,
        category: str | None = None,
        brand: str | None = None,
        q: str | None = None,
        sort: str = "relevance",
        page: int = Query(default=1, ge=1),
        pet_id: str | None = None,
    ):
        try:
            return request.app.state.service.list_products(
                species=species, category=category, brand=brand, q=q,
                sort=sort, page=page, pet_id=pet_id,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/products/{id}", response_model=ProductDetail)
    def product_detail(id: str, request: Request, pet_id: str | None = None):
        try:
            result = request.app.state.service.get_product(id, pet_id=pet_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return result

    @router.post("/admin/products", response_model=ProductListItem, status_code=201)
    def admin_create_product(payload: AdminProductCreate, request: Request, authorization: str = Header(default="")):
        try:
            return request.app.state.service.admin_create_product(
                authorization, payload.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.put("/admin/products/{id}", response_model=ProductListItem)
    def admin_update_product(id: str, payload: AdminProductUpdate, request: Request, authorization: str = Header(default="")):
        try:
            return request.app.state.service.admin_update_product(
                authorization, id, payload.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/recommendations", response_model=list[RecommendationItem])
    async def recommendations(
        request: Request,
        user_id: str,
        pet_id: str,
        k: int = Query(default=10, ge=1, le=10),
        use_llm: bool = True,
    ):
        service_obj = request.app.state.service
        try:
            # Provider HTTP calls have a 20s timeout; petrec retries once and falls back safely.
            import anyio
            call = partial(service_obj.recommend, user_id, pet_id, k=k, use_llm=use_llm)
            return await anyio.to_thread.run_sync(call)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/reminders", response_model=list[ReminderOut])
    def reminders(user_id: str, pet_id: str, request: Request):
        try:
            return request.app.state.service.list_reminders(user_id, pet_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request):
        service_obj = request.app.state.service
        try:
            import anyio
            messages = [message.model_dump() for message in payload.messages]
            call = partial(service_obj.chat, payload.user_id, payload.pet_id, messages)
            return await anyio.to_thread.run_sync(call)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/events", response_model=EventOut, status_code=201)
    def create_event(payload: EventCreate, request: Request):
        try:
            return request.app.state.service.record_event(payload.model_dump())
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/purchases", response_model=list[PurchaseOut], status_code=201)
    def create_purchases(payload: PurchaseCreate, request: Request):
        try:
            return request.app.state.service.record_purchases(payload.model_dump())
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    app.include_router(router)
    return app


app = create_app()
