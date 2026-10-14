"""Builds the decorated LLM clients for chat and rerank from settings and the provider registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from petrec.llm import LLMClient

from ..config import LLMSettings
from .base import ProviderState, Purpose
from .decorators import CircuitBreakerClient, RateLimitedClient, RetryingClient, SlidingWindowLimiter
from .errors import LLMConfigurationError, LLMProviderDisabled
from .registry import ProviderRegistry, default_registry

logger = logging.getLogger("petmind.llm")
DISABLED = "disabled"


@dataclass
class LLMRuntime:
    """The LLM clients the application uses; a missing client means deterministic fallback."""

    provider: str
    chat: LLMClient | None = None
    rerank: LLMClient | None = None
    state: ProviderState = ProviderState.DISABLED
    detail: str = ""
    models: dict[str, str] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.chat is not None

    def status(self) -> dict[str, Any]:
        state = self.state
        breaker = self.chat if isinstance(self.chat, CircuitBreakerClient) else None
        if breaker is not None and state is ProviderState.OK:
            state = breaker.state()
        report: dict[str, Any] = {"provider": self.provider, "state": state.value, "models": dict(self.models),
                                  "detail": self.detail}
        for purpose, client in (("chat", self.chat), ("rerank", self.rerank)):
            status = getattr(client, "status", None)
            if callable(status):
                budget = status().get("budget")
                if budget:
                    report.setdefault("budget", {})[purpose] = budget
        return report

    @classmethod
    def from_client(cls, client: LLMClient | None, provider: str = "injected") -> "LLMRuntime":
        """Wrap a single externally supplied client (tests, scripts) for both purposes."""
        state = ProviderState.OK if client is not None else ProviderState.DISABLED
        return cls(provider, client, client, state, models={"chat": getattr(client, "model", "")})


class LLMClientFactory:
    def __init__(self, settings: LLMSettings, registry: ProviderRegistry = default_registry) -> None:
        self.settings, self.registry = settings, registry

    def _resolve(self) -> str:
        name = self.settings.provider
        if name in ("", DISABLED, "none"):
            raise LLMProviderDisabled("LLM is disabled by LLM_PROVIDER")
        if self.registry.get(name) is None:
            raise LLMConfigurationError(f"Unknown LLM provider {name!r}; known: {', '.join(self.registry.names())}")
        if name not in self.settings.enabled_providers:
            raise LLMProviderDisabled(f"Provider {name!r} is implemented but not listed in LLM_ENABLED_PROVIDERS")
        return name

    def _decorate(self, client: LLMClient, purpose: Purpose) -> LLMClient:
        limits = self.settings.rerank_limits if purpose is Purpose.RERANK else self.settings.chat_limits
        limited = RateLimitedClient(client, SlidingWindowLimiter(limits))
        retrying = RetryingClient(limited, retries=self.settings.max_retries)
        return CircuitBreakerClient(retrying, failures=self.settings.breaker_failures,
                                    cooldown=self.settings.breaker_cooldown_seconds)

    def build(self, purpose: Purpose) -> LLMClient:
        name = self._resolve()
        spec = self.registry.get(name)
        configured = self.settings.rerank_model if purpose is Purpose.RERANK else self.settings.chat_model
        return self._decorate(spec.build(self.settings, configured or spec.default_model(purpose)), purpose)

    def build_runtime(self) -> LLMRuntime:
        provider = self.settings.provider or DISABLED
        try:
            chat, rerank = self.build(Purpose.CHAT), self.build(Purpose.RERANK)
        except LLMProviderDisabled as exc:
            logger.info("LLM disabled: %s", exc)
            return LLMRuntime(provider, state=ProviderState.DISABLED, detail=str(exc))
        except LLMConfigurationError as exc:
            logger.warning("LLM not configured, using deterministic fallback: %s", exc)
            return LLMRuntime(provider, state=ProviderState.NOT_CONFIGURED, detail=str(exc))
        logger.info("LLM provider %s ready (chat=%s, rerank=%s)", provider, chat.model, rerank.model)
        return LLMRuntime(provider, chat, rerank, ProviderState.OK,
                          models={"chat": chat.model, "rerank": rerank.model})
