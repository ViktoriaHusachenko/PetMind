"""Deterministic assistant mode used when the LLM is disabled, over quota or failing.

Routes the user's message to one tool by keywords and writes the reply from tool results,
so the answer still depends on what the user asked (BUG-021).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .context import ToolContext
from .tools import ToolRegistry

SIMPLIFIED_NOTE = "Асистент працює в спрощеному режимі."


class Intent(str, Enum):
    GREETING = "greeting"
    REMINDERS = "reminders"
    COMPOSITION = "composition"
    SEARCH = "search"
    RECOMMENDATIONS = "recommendations"


_RULES: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (Intent.REMINDERS, re.compile(r"нагадуван\w*|закінч\w*|запас\w*|поповн\w*|коли\s+(?:купити|замовити)|reminder", re.I)),
    (Intent.COMPOSITION, re.compile(r"склад\w*|інгредієнт\w*|що\s+входить|з\s+чого\s+зроблен\w*|ingredient", re.I)),
    (Intent.RECOMMENDATIONS, re.compile(r"порад\w*|рекоменд\w*|що\s+(?:купити|взяти|обрати)|найкращ\w*|recommend", re.I)),
    (Intent.SEARCH, re.compile(r"знайд\w*|шука\w*|покаж\w*|підбер\w*|чи\s+є|є\s+у\s+вас|маєте|корм\w*|ласощ\w*|консерв\w*|search|find", re.I)),
    (Intent.GREETING, re.compile(r"^\s*(?:привіт\w*|вітаю|добр\w+\s+\w+|hello|hi)\b|що\s+ти\s+вмієш|допомож\w*", re.I)),
)

HELP = ("Я можу підібрати безпечні товари за профілем тварини, знайти товар у каталозі, показати склад "
        "або нагадування про поповнення запасів. Що вас цікавить?")


@dataclass(frozen=True)
class RoutedReply:
    intent: Intent
    reply: str


class DeterministicIntentRouter:
    def __init__(self, tools: ToolRegistry, trace: Callable[[str, dict[str, Any], Callable[[], Any]], Any]) -> None:
        self.tools = tools
        self._trace = trace  # executes a tool and records it in the tool trace

    def classify(self, text: str) -> Intent:
        for intent, pattern in _RULES:
            if pattern.search(text or ""):
                return intent
        return Intent.SEARCH if len((text or "").split()) >= 2 else Intent.RECOMMENDATIONS

    def _call(self, ctx: ToolContext, name: str, args: dict[str, Any]) -> Any:
        return self._trace(name, args, lambda: self.tools.execute(name, args, ctx))

    def answer(self, text: str, ctx: ToolContext) -> RoutedReply:
        intent = self.classify(text)
        handler = {Intent.GREETING: self._greeting, Intent.REMINDERS: self._reminders,
                   Intent.COMPOSITION: self._composition, Intent.SEARCH: self._search,
                   Intent.RECOMMENDATIONS: self._recommendations}[intent]
        return RoutedReply(intent, handler(text, ctx))

    def _greeting(self, text: str, ctx: ToolContext) -> str:
        return HELP

    def _reminders(self, text: str, ctx: ToolContext) -> str:
        rows = self._call(ctx, "get_reminders", {}) or []
        if not rows:
            return "Зараз немає нагадувань про поповнення запасів для цієї тварини."
        lines = [f"• {row.get('title')}: {row.get('text') or row.get('severity')}" for row in rows[:5]]
        return "Нагадування для вашої тварини:\n" + "\n".join(lines)

    def _composition(self, text: str, ctx: ToolContext) -> str:
        found = self._call(ctx, "search_products", {"query": text}) or []
        if not found:
            return "Не знайшов товар із запиту. Уточніть, будь ласка, назву товару, склад якого показати."
        product = found[0]
        detail = self._call(ctx, "get_product_composition", {"product_id": product["id"]}) or {}
        safety = self._call(ctx, "check_safety", {"product_id": product["id"], "pet_id": ctx.pet_id}) or {}
        verdict = ("Склад перевірено: алергенів з профілю тварини не знайдено." if safety.get("safe")
                   else f"Увага: {safety.get('reason') or 'товар не підходить вашій тварині.'}")
        return f"Склад «{product['title']}»: {detail.get('ingredients_text') or 'склад не вказано'}.\n{verdict}"

    def _search(self, text: str, ctx: ToolContext) -> str:
        found = self._call(ctx, "search_products", {"query": text}) or []
        safe_titles = []
        for product in found[:6]:
            result = self._call(ctx, "check_safety", {"product_id": product["id"], "pet_id": ctx.pet_id}) or {}
            if result.get("safe"):
                safe_titles.append(product["title"])
        if safe_titles:
            return "Знайшов безпечні для вашої тварини товари: " + "; ".join(safe_titles[:4]) + "."
        if found:
            return "Знайдені товари не підходять за профілем тварини. Можу підібрати безпечні рекомендації."
        return self._recommendations(text, ctx, prefix="За цим запитом нічого не знайшов. ")

    def _recommendations(self, text: str, ctx: ToolContext, prefix: str = "") -> str:
        rows = self._call(ctx, "get_recommendations", {"k": 5}) or []
        if not rows:
            return prefix + "Зараз не знайшов перевірених рекомендацій."
        return prefix + "Ось безпечні рекомендації для вашої тварини: " + "; ".join(r["title"] for r in rows[:4]) + "."
