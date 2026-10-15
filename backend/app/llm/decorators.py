"""Reliability decorators around any LLMClient: rate limit, retry, circuit breaker.

Each decorator is itself an LLMClient (no own disk cache) and forwards to `inner`, so they
compose in any order and stay substitutable for the core client.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from petrec.llm import LLMClient, LLMResponse

from ..config import RateLimitSettings
from .base import ProviderState
from .errors import LLMError, LLMQuotaExceeded, LLMUnavailable

logger = logging.getLogger("petmind.llm")
Clock = Callable[[], float]


class LLMClientDecorator(LLMClient):
    def __init__(self, inner: LLMClient) -> None:
        super().__init__(cache_path=None)
        self.inner = inner
        self.model = getattr(inner, "model", "")

    def _complete(self, system: str, user: str) -> str:  # pragma: no cover - complete() is overridden
        return self.inner.complete(system, user)

    def complete(self, system: str, user: str) -> str:
        return self.inner.complete(system, user)

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        return self.inner.complete_with_tools(messages, tools)

    def is_cached(self, operation: str, payload: Any) -> bool:
        inner = self.inner
        if isinstance(inner, LLMClientDecorator):
            return inner.is_cached(operation, payload)
        try:  # core clients keep a private response cache; a hit costs no quota
            return inner._key(operation, payload) in inner._cache
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        report = getattr(self.inner, "status", None)
        return dict(report()) if callable(report) else {"model": self.model}


class SlidingWindowLimiter:
    """Thread-safe per-minute and per-day request budget (client-side view of a free tier)."""

    def __init__(self, limits: RateLimitSettings, clock: Clock = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, max_wait_seconds: float = 8.0) -> None:
        self.limits, self.clock, self.sleep = limits, clock, sleep
        # A full minute window that frees up soon is waited out; the daily budget always fails fast.
        self.max_wait_seconds = max_wait_seconds
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        for window, span in ((self._minute, 60.0), (self._day, 86_400.0)):
            while window and now - window[0] >= span:
                window.popleft()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self.clock()
                self._prune(now)
                if self.limits.requests_per_day and len(self._day) >= self.limits.requests_per_day:
                    raise LLMQuotaExceeded("Client-side daily LLM budget exhausted")
                if not self.limits.requests_per_minute or len(self._minute) < self.limits.requests_per_minute:
                    self._minute.append(now)
                    self._day.append(now)
                    return
                wait = 60.0 - (now - self._minute[0]) + 0.05
                if wait > self.max_wait_seconds:
                    raise LLMQuotaExceeded("Client-side per-minute LLM budget exhausted")
            self.sleep(wait)

    def usage(self) -> dict[str, int]:
        with self._lock:
            return {"minute_used": len(self._minute), "minute_limit": self.limits.requests_per_minute,
                    "day_used": len(self._day), "day_limit": self.limits.requests_per_day}


class RateLimitedClient(LLMClientDecorator):
    """Fails fast with LLMQuotaExceeded instead of blocking a request thread; cache hits are free."""

    def __init__(self, inner: LLMClient, limiter: SlidingWindowLimiter) -> None:
        super().__init__(inner)
        self.limiter = limiter

    def complete(self, system: str, user: str) -> str:
        if not self.is_cached("complete", {"system": system, "user": user}):
            self.limiter.acquire()
        return self.inner.complete(system, user)

    def complete_with_tools(self, messages, tools) -> LLMResponse:
        # Tool calls are rarely repeated verbatim; always count them.
        self.limiter.acquire()
        return self.inner.complete_with_tools(messages, tools)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "budget": self.limiter.usage()}


class RetryingClient(LLMClientDecorator):
    """Retries only transient failures with exponential backoff; quota and config errors pass through."""

    def __init__(self, inner: LLMClient, retries: int = 2, base_delay: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        super().__init__(inner)
        self.retries, self.base_delay, self.sleep = max(0, retries), base_delay, sleep

    def _call(self, function: Callable[[], Any]) -> Any:
        for attempt in range(self.retries + 1):
            try:
                return function()
            except LLMUnavailable:
                if attempt == self.retries:
                    raise
                delay = self.base_delay * (2 ** attempt)
                logger.warning("LLM transient failure; retry %d/%d in %.1fs", attempt + 1, self.retries, delay)
                self.sleep(delay)

    def complete(self, system: str, user: str) -> str:
        return self._call(lambda: self.inner.complete(system, user))

    def complete_with_tools(self, messages, tools) -> LLMResponse:
        return self._call(lambda: self.inner.complete_with_tools(messages, tools))


class CircuitBreakerClient(LLMClientDecorator):
    """After `failures` consecutive LLM errors, short-circuit for `cooldown` seconds."""

    def __init__(self, inner: LLMClient, failures: int = 3, cooldown: float = 60.0,
                 clock: Clock = time.monotonic) -> None:
        super().__init__(inner)
        self.failures, self.cooldown, self.clock = failures, cooldown, clock
        self._count = 0
        self._opened_at: float | None = None
        self._last_error: LLMError | None = None
        self._lock = threading.Lock()

    def _guard(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if self.clock() - self._opened_at < self.cooldown:
                raise LLMUnavailable("LLM circuit open after repeated failures")
            self._opened_at = None  # half-open: let one request probe the provider

    def _call(self, function: Callable[[], Any]) -> Any:
        self._guard()
        try:
            result = function()
        except LLMError as exc:
            with self._lock:
                self._count += 1
                self._last_error = exc
                if self._count >= self.failures:
                    self._opened_at = self.clock()
                    logger.warning("LLM circuit opened for %.0fs: %s", self.cooldown, type(exc).__name__)
            raise
        with self._lock:
            self._count, self._last_error = 0, None
        return result

    def complete(self, system: str, user: str) -> str:
        return self._call(lambda: self.inner.complete(system, user))

    def complete_with_tools(self, messages, tools) -> LLMResponse:
        return self._call(lambda: self.inner.complete_with_tools(messages, tools))

    def state(self) -> ProviderState:
        with self._lock:
            if self._opened_at is not None and self.clock() - self._opened_at < self.cooldown:
                return ProviderState.QUOTA if isinstance(self._last_error, LLMQuotaExceeded) else ProviderState.UNAVAILABLE
            if isinstance(self._last_error, LLMQuotaExceeded):
                return ProviderState.QUOTA
            return ProviderState.OK

    def status(self) -> dict[str, Any]:
        return {**super().status(), "state": self.state().value}
