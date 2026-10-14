"""Agent tools: one class per tool, registered in a ToolRegistry (open for extension)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .context import ToolContext, is_sterilized_product, mentions_sterilization

NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}


class ToolArgumentError(ValueError):
    """The model called a tool with arguments that violate its schema."""


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any] = NO_ARGS

    def definition(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    @abstractmethod
    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        """Execute with validated-by-caller dict args; raise ToolArgumentError/LookupError on bad input."""

    @staticmethod
    def _no_args(args: dict[str, Any]) -> None:
        if args:
            raise ToolArgumentError("Інструмент не приймає аргументи")


def _card(basic: dict[str, Any], reason: str, *, llm_used: bool = False) -> dict[str, Any]:
    """Card for a product approved through check_safety. The match percent is unknown here (no s_final)."""
    return {"item_id": str(basic["id"]), "title": basic.get("title", ""), "brand": basic.get("brand") or "",
            "price": basic.get("price"), "image_url": basic.get("image_url"), "match_percent": None,
            "reason": reason, "safe_badge": True, "composition_summary": [], "llm_used": llm_used}


class GetPetProfileTool(Tool):
    name = "get_pet_profile"
    description = "Отримати профіль тварини поточного користувача."

    # Owner data never leaves the backend (free-tier prompts may be used by the provider).
    PRIVATE_FIELDS = frozenset({"display_name", "user_id", "preference_state", "photo_url", "email"})

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        self._no_args(args)
        return {key: value for key, value in ctx.profile.items() if key not in self.PRIVATE_FIELDS}


class SearchProductsTool(Tool):
    name = "search_products"
    description = ("Знайти товари каталогу за словами запиту (назва, бренд, категорія, склад). "
                   "Вид тварини береться з профілю автоматично.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "description": "Ключові слова, наприклад «лосось стерилізованих»"},
        "filters": {"type": "object", "properties": {"category": {"type": "string"}, "brand": {"type": "string"}},
                    "additionalProperties": False}},
        "required": ["query"], "additionalProperties": False}

    def __init__(self, limit: int = 8) -> None:
        self.limit = limit

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        query, filters = args.get("query"), args.get("filters") or {}
        if not isinstance(query, str) or not isinstance(filters, dict):
            raise ToolArgumentError("Очікуються query: string і filters: object")
        extra = " ".join(str(filters.get(key) or "") for key in ("category", "brand"))
        # The pet's species always comes from the profile; the model cannot override it.
        rows = ctx.service.search_catalog(f"{query} {extra}".strip(), species=ctx.species, limit=self.limit * 2)
        wants_sterilized = ctx.sterilized or mentions_sterilization(query)
        results = []
        for row in rows:
            pid = str(row["id"])
            basic = ctx.service.get_product_basic(pid)
            if basic is None or basic.get("species") != ctx.species:
                continue
            ctx.seen[pid] = basic
            sterilized = is_sterilized_product(basic)
            results.append({"id": pid, "title": basic.get("title", ""), "brand": basic.get("brand"),
                            "category": basic.get("category"), "price": basic.get("price"),
                            "for_sterilized": sterilized, "relevance": row.get("score", 0.0)})
        if wants_sterilized:
            # Preference, not exclusion: suitable non-sterilized food is still safe to show.
            results.sort(key=lambda item: (not item["for_sterilized"], -float(item["relevance"] or 0)))
        return results[: self.limit]


class CheckSafetyTool(Tool):
    name = "check_safety"
    description = "Перевірити безпечність товару для тварини. Це єдине джерело рішення про безпечність."
    parameters = {"type": "object", "properties": {"product_id": {"type": "string"}, "pet_id": {"type": "string"}},
                  "required": ["product_id", "pet_id"], "additionalProperties": False}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        product_id = args.get("product_id")
        if not isinstance(product_id, str) or args.get("pet_id") != ctx.pet_id:
            raise ToolArgumentError("Некоректні product_id або pet_id")
        basic = ctx.service.get_product_basic(product_id)
        if basic is None:
            raise LookupError("Товар не знайдено")
        ctx.seen[product_id] = basic
        decision = ctx.service.check_safety(product_id, ctx.pet_id)
        if decision.get("safe") is True:
            ctx.approved[product_id] = _card(basic, "Склад перевірено для профілю тварини.")
        else:
            ctx.approved.pop(product_id, None)
        return {"product_id": product_id, "safe": decision.get("safe") is True, "reason": decision.get("reason")}


class GetRecommendationsTool(Tool):
    name = "get_recommendations"
    description = "Отримати персональні рекомендації (вже перевірені за алергенами)."
    parameters = {"type": "object", "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 10}},
                  "additionalProperties": False}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        k = args.get("k", 5)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            raise ToolArgumentError("k має бути цілим числом від 1 до 10")
        safe_rows = []
        for row in ctx.service.recommend(ctx.user_id, ctx.pet_id, k=k, use_llm=True):
            product_id = str(row.get("item_id", ""))
            basic = ctx.service.get_product_basic(product_id) if product_id else None
            if basic is None:
                continue
            ctx.seen[product_id] = basic
            # Recommendation rows carry the deterministic s_final percent; keep it as is.
            if ctx.approve_safe(product_id, dict(row)):
                safe_rows.append({"item_id": product_id, "title": row.get("title") or basic.get("title", ""),
                                  "match_percent": row.get("match_percent"), "reason": row.get("reason", "")})
        return safe_rows


class GetRemindersTool(Tool):
    name = "get_reminders"
    description = "Отримати нагадування про поповнення запасів для тварини."

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        self._no_args(args)
        return ctx.service.list_reminders(ctx.user_id, ctx.pet_id)


class GetProductCompositionTool(Tool):
    name = "get_product_composition"
    description = "Отримати склад і поживні показники товару."
    parameters = {"type": "object", "properties": {"product_id": {"type": "string"}},
                  "required": ["product_id"], "additionalProperties": False}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        product_id = args.get("product_id")
        if not isinstance(product_id, str):
            raise ToolArgumentError("Очікується product_id")
        basic = ctx.service.get_product_basic(product_id)
        if basic is None:
            raise LookupError("Товар не знайдено")
        ctx.seen[product_id] = basic
        detail = ctx.service.get_product(product_id, pet_id=None)
        if detail is None:
            raise LookupError("Товар не знайдено")
        return {key: detail.get(key) for key in ("id", "title", "ingredients_text", "ingredients", "S_nutr", "Q_ingr")}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._tools.values()]

    def names(self) -> set[str]:
        return set(self._tools)

    def execute(self, name: str, args: Any, ctx: ToolContext) -> Any:
        tool = self._tools.get(name)
        if tool is None or not isinstance(args, dict) or "__invalid_json__" in args:
            raise ToolArgumentError("Невідомий інструмент або некоректні аргументи")
        return tool.run(ctx, args)


def default_tools() -> ToolRegistry:
    return ToolRegistry([GetPetProfileTool(), SearchProductsTool(), CheckSafetyTool(), GetRecommendationsTool(),
                         GetRemindersTool(), GetProductCompositionTool()])
