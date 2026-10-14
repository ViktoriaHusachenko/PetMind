"""PetMind chat agent: a bounded function-calling loop with deterministic guards (no LangGraph)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from petrec.llm import LLMClient, LLMResponse

from ..llm import LLMQuotaExceeded, tool_calls_of
from ..llm.base import assistant_tool_message
from .context import AgentBackend, ToolContext, mentions_sterilization
from .guards import FinalCardsGuard, MedicalGuard, ProductMentionGuard
from .intents import SIMPLIFIED_NOTE, DeterministicIntentRouter
from .memory import ConversationWindow
from .prompts import build_system_prompt
from .tools import ToolArgumentError, ToolRegistry, default_tools

logger = logging.getLogger("petmind.agent")

REFUSAL = "Не можу рекомендувати цей товар для вашої тварини. Можу підібрати інші варіанти, перевірені за її профілем."
STEP_LIMIT = "Перевищено ліміт кроків. Можу допомогти пошуком перевірених товарів або нагадувань."
MAX_CALLS_PER_STEP = 6


class PetMindAgent:
    def __init__(self, service: AgentBackend, llm_client: LLMClient | None, max_steps: int = 5, *,
                 tools: ToolRegistry | None = None, medical_guard: MedicalGuard | None = None,
                 mention_guard: ProductMentionGuard | None = None, cards_guard: FinalCardsGuard | None = None,
                 window: ConversationWindow | None = None) -> None:
        self.service = service
        self.llm = llm_client
        self.max_steps = min(5, max(1, int(max_steps)))
        self.tools = tools or default_tools()
        self.medical_guard = medical_guard or MedicalGuard()
        self.mention_guard = mention_guard or ProductMentionGuard()
        self.cards_guard = cards_guard or FinalCardsGuard()
        self.window = window or ConversationWindow()

    # -- public API -----------------------------------------------------------------------
    def run(self, user_id: str, pet_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        profile = dict(self.service.get_pet_profile(user_id, pet_id))
        history = self.window.apply(messages)
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if mentions_sterilization(last_user):
            profile["lifestyle"] = f"{profile.get('lifestyle') or ''} sterilized".strip()
        ctx = ToolContext(self.service, user_id, pet_id, profile)
        trace: list[dict[str, Any]] = []

        verdict = self.medical_guard.check(last_user)
        if verdict.blocked:
            return self._response(verdict.reply, [], trace, "guard")
        if self.llm is None:
            return self._fallback(last_user, ctx, trace)

        conversation: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(profile, pet_id)},
                                              *history]
        reply: str | None = None
        for step in range(1, self.max_steps + 1):
            started = time.perf_counter()
            try:
                response = self.llm.complete_with_tools(conversation, self.tools.definitions())
            except Exception as exc:  # any provider failure → deterministic mode, never a 500
                trace.append({"step": step, "tool": "llm", "ok": False, "error": type(exc).__name__})
                if not isinstance(exc, LLMQuotaExceeded):
                    logger.warning("chat LLM call failed: %s", type(exc).__name__)
                if step == 1:
                    return self._fallback(last_user, ctx, trace)
                break
            logger.info("step=agent_llm latency_ms=%.1f", (time.perf_counter() - started) * 1000)
            if not isinstance(response, LLMResponse):
                response = LLMResponse(str(getattr(response, "content", response)), getattr(response, "tool_call", None))
            calls = tool_calls_of(response)[:MAX_CALLS_PER_STEP]
            if not calls:
                reply = response.content.strip() or None
                break
            ids = [call.id or f"petmind-{step}-{index}" for index, call in enumerate(calls)]
            conversation.append(assistant_tool_message(response.content or "", calls, ids))
            for call, call_id in zip(calls, ids):
                encoded = self._run_tool(step, call.name, call.arguments, ctx, trace)
                conversation.append({"role": "tool", "tool_call_id": call_id, "content": encoded})
        else:
            reply = STEP_LIMIT

        if reply is None:
            reply = "Зараз не знайшов перевірених рекомендацій." if not ctx.approved else \
                "Ось товари, перевірені за профілем вашої тварини."
        # Never let unverified or unsafe product identifiers/names through in the answer.
        if self.mention_guard.violations(reply, ctx):
            reply, ctx.approved = REFUSAL, {}
        return self._response(reply, self.cards_guard.filter(ctx), trace, "llm")

    # -- internals ------------------------------------------------------------------------
    def _run_tool(self, step: int, name: str, args: Any, ctx: ToolContext, trace: list[dict[str, Any]]) -> str:
        started = time.perf_counter()
        row: dict[str, Any] = {"step": step, "tool": name, "arguments": args}
        try:
            result = self.tools.execute(name, args, ctx)
            encoded = json.dumps(result, ensure_ascii=False, default=str)
            row.update(ok=True, result_summary=encoded[:500])
        except (ToolArgumentError, ValueError, LookupError, TypeError, KeyError) as exc:
            encoded = json.dumps({"error": "invalid_tool_call", "message": str(exc)}, ensure_ascii=False)
            row.update(ok=False, error=str(exc))
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        trace.append(row)
        logger.info("tool_trace=%s", json.dumps(row, ensure_ascii=False, default=str))
        return encoded

    def _fallback(self, text: str, ctx: ToolContext, trace: list[dict[str, Any]]) -> dict[str, Any]:
        def traced(name: str, args: dict[str, Any], call: Callable[[], Any]) -> Any:
            started = time.perf_counter()
            row: dict[str, Any] = {"step": 0, "tool": name, "arguments": args, "mode": "fallback"}
            try:
                result = call()
                row.update(ok=True)
                return result
            except (ToolArgumentError, ValueError, LookupError) as exc:
                row.update(ok=False, error=str(exc))
                return None
            finally:
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                trace.append(row)

        routed = DeterministicIntentRouter(self.tools, traced, self.cards_guard.filter).answer(text, ctx)
        reply = f"{routed.reply}\n\n{SIMPLIFIED_NOTE}"
        return self._response(reply, self.cards_guard.filter(ctx), trace, "fallback")

    @staticmethod
    def _response(reply: str, products: list[dict[str, Any]], trace: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        return {"reply": reply, "products": products, "tool_trace": trace, "mode": mode}
