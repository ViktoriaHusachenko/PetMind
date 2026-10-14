"""Google Gemini provider (active by default, free tier).

Uses Gemini's OpenAI-compatible endpoint through the `openai` SDK that is already a dependency,
so the agent's OpenAI-shaped messages and tool definitions pass through unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from petrec.llm import LLMClient, LLMResponse

from ...config import LLMSettings
from ..base import MultiToolResponse, ProviderToolCall
from ..errors import LLMConfigurationError, LLMError, LLMQuotaExceeded, LLMUnavailable
from ..registry import ProviderSpec

logger = logging.getLogger("petmind.llm.gemini")
DEFAULT_MODEL = "gemini-flash-lite-latest"
_STANDARD_TOOL_CALL_KEYS = {"id", "type", "function", "index"}
# JSON-schema keywords Gemini function declarations reject.
_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "$schema"}


def _sanitize_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_schema(item) for key, item in value.items() if key not in _UNSUPPORTED_SCHEMA_KEYS}
    if isinstance(value, list):
        return [_sanitize_schema(item) for item in value]
    return value


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        # Invalid JSON becomes a non-dict argument that the agent rejects and traces.
        return {"__invalid_json__": str(raw)[:200]}
    return value if isinstance(value, dict) else {"__invalid_json__": str(raw)[:200]}


class GeminiClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, *,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
                 timeout: float = 20.0, cache_path: str | None = ".cache/petrec_llm_cache.json",
                 sdk_client: Any | None = None) -> None:
        super().__init__(cache_path)
        if not api_key and sdk_client is None:
            raise LLMConfigurationError("GEMINI_API_KEY is not set")
        self.model, self.base_url, self.timeout = model, base_url, timeout
        self._api_key = api_key
        self._sdk = sdk_client

    def _client(self) -> Any:
        if self._sdk is None:
            from openai import OpenAI
            # Retries are handled by RetryingClient so they respect the rate limiter.
            self._sdk = OpenAI(api_key=self._api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        return self._sdk

    def _create(self, **kwargs: Any) -> Any:
        try:
            return self._client().chat.completions.create(model=self.model, temperature=0, **kwargs)
        except Exception as exc:  # map SDK errors to the provider-neutral taxonomy
            raise self._translate(exc) from exc

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if name == "RateLimitError" or status == 429:
            return LLMQuotaExceeded("Gemini quota exceeded")
        if name in {"APITimeoutError", "APIConnectionError", "InternalServerError"} or (status and status >= 500):
            return LLMUnavailable(f"Gemini unavailable: {name}")
        if name in {"AuthenticationError", "PermissionDeniedError"} or status in {401, 403}:
            return LLMConfigurationError("Gemini rejected the API key")
        return LLMError(f"Gemini request failed: {name}")

    def _complete(self, system: str, user: str) -> str:
        result = self._create(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return result.choices[0].message.content or ""

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        native_tools = _sanitize_schema(tools) if tools else None
        key = self._key("tools", {"messages": messages, "tools": native_tools})
        if key in self._cache:
            return self._from_cache(self._cache[key])
        result = self._create(messages=messages, tools=native_tools)
        message = result.choices[0].message
        calls = []
        for index, raw in enumerate(getattr(message, "tool_calls", None) or []):
            dumped = raw.model_dump() if hasattr(raw, "model_dump") else {}
            extra = {k: v for k, v in dumped.items() if k not in _STANDARD_TOOL_CALL_KEYS and v is not None}
            calls.append(ProviderToolCall(raw.function.name, _parse_arguments(raw.function.arguments),
                                          raw.id or f"gemini-{index}", extra or None))
        response = MultiToolResponse.of(message.content or "", calls)
        self._cache[key] = {"content": response.content, "tool_calls": [asdict(call) for call in calls]}
        self._save_cache()
        return response

    @staticmethod
    def _from_cache(value: dict[str, Any]) -> MultiToolResponse:
        calls = [ProviderToolCall(**row) for row in value.get("tool_calls", [])]
        return MultiToolResponse.of(value.get("content", ""), calls)

    def status(self) -> dict[str, Any]:
        return {"provider": "gemini", "model": self.model}


def build(settings: LLMSettings, model: str) -> LLMClient:
    if not settings.gemini_api_key:
        raise LLMConfigurationError("GEMINI_API_KEY is not set")
    return GeminiClient(model=model or DEFAULT_MODEL, api_key=settings.gemini_api_key,
                        base_url=settings.gemini_base_url, timeout=settings.timeout_seconds,
                        cache_path=settings.cache_path)


SPEC = ProviderSpec("gemini", "Google Gemini", build, default_chat_model=DEFAULT_MODEL, default_rerank_model=DEFAULT_MODEL)
