"""Anthropic adapter over the core client (implemented, disabled by default)."""

from __future__ import annotations

from petrec.llm import AnthropicClient, LLMClient

from ...config import LLMSettings
from ..errors import LLMConfigurationError
from ..registry import ProviderSpec


def build(settings: LLMSettings, model: str) -> LLMClient:
    if not settings.anthropic_api_key:
        raise LLMConfigurationError("ANTHROPIC_API_KEY is not set")
    return AnthropicClient(model=model, api_key=settings.anthropic_api_key, cache_path=settings.cache_path)


SPEC = ProviderSpec("anthropic", "Anthropic", build,
                    default_chat_model="claude-haiku-5-5", default_rerank_model="claude-haiku-5-5")
