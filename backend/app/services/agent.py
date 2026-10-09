"""Small, bounded function-calling chat agent for Petly."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from petrec.llm import LLMClient, LLMResponse

logger = logging.getLogger("petly.agent")

TOOLS = [
    {"type": "function", "function": {"name": "get_pet_profile", "description": "Отримати профіль тварини поточного користувача.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_products", "description": "Знайти товари за запитом і фільтрами.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object", "properties": {"species": {"type": "string"}, "category": {"type": "string"}, "brand": {"type": "string"}}, "additionalProperties": False}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_safety", "description": "Перевірити безпечність товару для тварини. Це єдине джерело рішення про безпечність.", "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}, "pet_id": {"type": "string"}}, "required": ["product_id", "pet_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_recommendations", "description": "Отримати персональні рекомендації.", "parameters": {"type": "object", "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 10}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_reminders", "description": "Отримати нагадування для тварини.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_product_composition", "description": "Отримати склад товару.", "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"], "additionalProperties": False}}},
]
_TOOL_NAMES = {row["function"]["name"] for row in TOOLS}
_MEDICAL = re.compile(r"лікув|виліку|діагноз|симптом|медикамент|ліки|лікуван|treat(?:ment)?|diagnos|medicine|medication|cure", re.I)
_PRODUCT_ID = re.compile(r"\b[A-Z0-9]{8,16}\b")
_SYSTEM = """Ти — помічник вебзастосунку Petly. Відповідай українською.
Використовуй інструменти для фактів про профіль, товари, склад, нагадування та рекомендації.
Не вигадуй даних. Рішення про безпечність бере́ться лише з check_safety; ніколи не рекомендуй товар, якщо інструмент повернув safe=false.
Рекомендуй лише знайдені й перевірені товари. На прохання про лікування, діагноз чи ліки порадь звернутися до ветеринара."""
_REFUSAL = "Не можу рекомендувати цей товар для вашої тварини. Можу підібрати інші варіанти, перевірені за її профілем."
_VET = "Я не надаю порад щодо лікування чи діагнозу. Будь ласка, зверніться до ветеринара."


class PetlyAgent:
    def __init__(self, service: Any, llm_client: LLMClient | None, max_steps: int = 5):
        self.service = service
        self.llm = llm_client
        self.max_steps = min(5, max(1, int(max_steps)))

    def run(self, user_id: str, pet_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        profile = self.service.get_pet_profile(user_id, pet_id)
        trace: list[dict[str, Any]] = []
        approved: dict[str, dict[str, Any]] = {}
        seen: dict[str, dict[str, Any]] = {}
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if any(term in last_user.lower() for term in (
            "steriliz", "neuter", "\u0441\u0442\u0435\u0440\u0438\u043b", "\u0441\u043f\u0435\u0440", "\u043a\u0430\u0441\u0442\u0440",
        )):
            profile["lifestyle"] = f"{profile.get('lifestyle') or ''} sterilized".strip()
        if _MEDICAL.search(last_user):
            return {"reply": _VET, "products": [], "tool_trace": trace}
        if self.llm is None:
            rows = self._tool("get_recommendations", {"k": 10}, user_id, pet_id, profile, approved, seen)
            reply = "Ось кілька безпечних рекомендацій для вашої тварини." if rows else "Зараз не знайшов перевірених рекомендацій."
            return {"reply": reply, "products": self._final_products(approved, user_id, pet_id, profile), "tool_trace": trace}

        conversation: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM}, *history]
        reply = "Зараз не знайшов перевірених рекомендацій."
        for step in range(1, self.max_steps + 1):
            started = time.perf_counter()
            try:
                response = self.llm.complete_with_tools(conversation, TOOLS)
            except Exception as exc:
                trace.append({"step": step, "tool": "llm", "ok": False, "error": type(exc).__name__})
                logger.exception("chat LLM call failed")
                break
            logger.info("step=agent_llm latency_ms=%.1f", (time.perf_counter() - started) * 1000)
            if not isinstance(response, LLMResponse):
                # Accept simple compatible mock/provider response objects.
                response = LLMResponse(str(getattr(response, "content", response)), getattr(response, "tool_call", None))
            call = response.tool_call
            if call is None:
                reply = response.content.strip() or reply
                break
            call_id = call.id or f"petly-{step}"
            conversation.append({"role": "assistant", "content": response.content or "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}]})
            started = time.perf_counter()
            try:
                result = self._tool(call.name, call.arguments, user_id, pet_id, profile, approved, seen)
                encoded = json.dumps(result, ensure_ascii=False, default=str)
                row = {"step": step, "tool": call.name, "arguments": call.arguments, "ok": True, "result_summary": encoded[:500], "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
            except (ValueError, LookupError, TypeError, KeyError) as exc:
                encoded = json.dumps({"error": "invalid_tool_call", "message": str(exc)}, ensure_ascii=False)
                row = {"step": step, "tool": call.name, "arguments": call.arguments, "ok": False, "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
            trace.append(row)
            logger.info("tool_trace=%s", json.dumps(row, ensure_ascii=False, default=str))
            conversation.append({"role": "tool", "tool_call_id": call_id, "content": encoded})
        else:
            reply = "Перевищено ліміт кроків. Можу допомогти пошуком перевірених товарів або нагадувань."

        # Never let unverified or unsafe product identifiers/names through in the answer.
        if self._contains_unapproved_product(reply, pet_id, seen):
            reply, approved = _REFUSAL, {}
        return {"reply": reply, "products": self._final_products(approved, user_id, pet_id, profile), "tool_trace": trace}

    def _tool(self, name: str, args: dict[str, Any], user_id: str, pet_id: str,
              profile: dict[str, Any], approved: dict[str, dict[str, Any]], seen: dict[str, dict[str, Any]]) -> Any:
        if name not in _TOOL_NAMES or not isinstance(args, dict):
            raise ValueError("Невідомий інструмент або некоректні аргументи")
        if name == "get_pet_profile":
            if args: raise ValueError("Інструмент не приймає аргументи")
            return profile
        if name == "search_products":
            query, filters = args.get("query"), args.get("filters", {})
            if not isinstance(query, str) or not isinstance(filters, dict): raise ValueError("Очікуються query: string і filters: object")
            # Let the model narrow its query, but never let it override the pet's species.
            pet_species = profile.get("species")
            rows = self.service.list_products(species=pet_species, category=filters.get("category"), brand=filters.get("brand"), q=query or None, sort="relevance", page=1, pet_id=None)["items"]
            needs = " ".join((query, str(profile.get("lifestyle") or ""), str(profile.get("diet_restrictions") or ""))).lower()
            sterilized_need = any(term in needs for term in (
                "steriliz", "neuter", "\u0441\u0442\u0435\u0440\u0438\u043b", "\u0441\u043f\u0435\u0440", "\u043a\u0430\u0441\u0442\u0440",
            ))
            clean = []
            for row in rows:
                pid = str(row["id"]); basic = self.service.get_product_basic(pid)
                if basic is None: continue
                if basic.get("species") != pet_species: continue
                if sterilized_need:
                    label = " ".join(str(basic.get(key) or "") for key in ("title", "category")).lower()
                    if not any(term in label for term in (
                        "steriliz", "neuter", "spay", "spayed", "\u0441\u0442\u0435\u0440\u0438\u043b", "\u043a\u0430\u0441\u0442\u0440",
                    )):
                        continue
                seen[pid] = basic
                clean.append({"id": pid, "title": basic.get("title", ""), "brand": basic.get("brand"), "category": basic.get("category"), "price": basic.get("price")})
            return clean
        if name == "check_safety":
            product_id = args.get("product_id")
            requested_pet = args.get("pet_id")
            if not isinstance(product_id, str) or requested_pet != pet_id: raise ValueError("Некоректні product_id або pet_id")
            basic = self.service.get_product_basic(product_id)
            if basic is None: raise LookupError("Товар не знайдено")
            seen[product_id] = basic
            decision = self.service.check_safety(product_id, pet_id)
            if decision.get("safe") is True:
                approved[product_id] = self._recommendation(basic, True, "Склад перевірено для профілю тварини.")
            else:
                approved.pop(product_id, None)
            return {"product_id": product_id, "safe": decision.get("safe") is True, "reason": decision.get("reason")}
        if name == "get_recommendations":
            k = args.get("k", 10)
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10: raise ValueError("k має бути цілим числом від 1 до 10")
            rows = self.service.recommend(user_id, pet_id, k=k, use_llm=True)
            safe_rows = []
            for row in rows:
                product_id = str(row.get("item_id", "")); basic = self.service.get_product_basic(product_id)
                if not product_id or basic is None: continue
                seen[product_id] = basic
                decision = self.service.check_safety(product_id, pet_id)
                if decision.get("safe") is True:
                    output = dict(row); output.update({"item_id": product_id, "safe_badge": True})
                    approved[product_id] = output; safe_rows.append({"item_id": product_id, "title": output.get("title", basic.get("title", "")), "reason": output.get("reason", "")})
                else: approved.pop(product_id, None)
            return safe_rows
        if name == "get_reminders":
            if args: raise ValueError("Інструмент не приймає аргументи")
            return self.service.list_reminders(user_id, pet_id)
        if name == "get_product_composition":
            product_id = args.get("product_id")
            if not isinstance(product_id, str): raise ValueError("Очікується product_id")
            basic = self.service.get_product_basic(product_id)
            if basic is None: raise LookupError("Товар не знайдено")
            seen[product_id] = basic
            detail = self.service.get_product(product_id, pet_id=None)
            if detail is None: raise LookupError("Товар не знайдено")
            return {key: detail.get(key) for key in ("id", "title", "ingredients_text", "ingredients", "S_nutr", "Q_ingr")}
        raise ValueError("Невідомий інструмент")

    @staticmethod
    def _recommendation(row: dict[str, Any], safe: bool, reason: str) -> dict[str, Any]:
        return {"item_id": str(row["id"]), "title": row.get("title", ""), "brand": row.get("brand") or "", "price": row.get("price"), "image_url": row.get("image_url"), "match_percent": 0, "reason": reason, "safe_badge": safe, "composition_summary": [], "llm_used": True}

    def _final_products(self, products: dict[str, dict[str, Any]], user_id: str, pet_id: str,
                        profile: dict[str, Any]) -> list[dict[str, Any]]:
        output = []
        for product_id, row in list(products.items()):
            basic = self.service.get_product_basic(product_id)
            if basic is None or basic.get("species") != profile.get("species"): continue
            pet_needs = " ".join((str(profile.get("lifestyle") or ""), str(profile.get("diet_restrictions") or ""))).lower()
            if any(term in pet_needs for term in ("steriliz", "neuter", "\u0441\u0442\u0435\u0440\u0438\u043b", "\u043a\u0430\u0441\u0442\u0440")):
                label = " ".join(str(basic.get(key) or "") for key in ("title", "category")).lower()
                if not any(term in label for term in (
                    "steriliz", "neuter", "spay", "spayed", "\u0441\u0442\u0435\u0440\u0438\u043b", "\u043a\u0430\u0441\u0442\u0440",
                )):
                    continue
            decision = self.service.check_safety(product_id, pet_id)
            if decision.get("safe") is True:
                item = dict(row); item["safe_badge"] = True; output.append(item)
        return output

    def _contains_unapproved_product(self, reply: str, pet_id: str, seen: dict[str, dict[str, Any]]) -> bool:
        candidates = set(_PRODUCT_ID.findall(reply))
        for pid, basic in seen.items():
            title = str(basic.get("title") or "")
            if title and title.casefold() in reply.casefold(): candidates.add(pid)
        for product_id in candidates:
            basic = self.service.get_product_basic(product_id)
            if basic is None or self.service.check_safety(product_id, pet_id).get("safe") is not True:
                return True
        return False
