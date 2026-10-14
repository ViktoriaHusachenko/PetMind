"""OpenAI adapter over the core client (implemented, disabled by default)."""

from __future__ import annotations

from petrec.llm import LLMClient, OpenAIClient

from ...config import LLMSettings
from ..errors import LLMConfigurationError
from ..registry import ProviderSpec


def build(settings: LLMSettings, model: str) -> LLMClient:
    if not settings.openai_api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not set")
    return OpenAIClient(model=model, api_key=settings.openai_api_key, cache_path=settings.cache_path)


SPEC = ProviderSpec("openai", "OpenAI", build, default_chat_model="gpt-4o-mini", default_rerank_model="gpt-4o-mini")
