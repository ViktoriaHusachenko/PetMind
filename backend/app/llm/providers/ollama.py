"""Local Ollama adapter over the core client (implemented, disabled by default)."""

from __future__ import annotations

from petrec.llm import LLMClient, OllamaClient

from ...config import LLMSettings
from ..registry import ProviderSpec


def build(settings: LLMSettings, model: str) -> LLMClient:
    return OllamaClient(model=model or settings.ollama_model, host=settings.ollama_host, cache_path=settings.cache_path)


# Empty defaults: the model comes from LLM_CHAT_MODEL or OLLAMA_MODEL.
SPEC = ProviderSpec("ollama", "Ollama", build, default_chat_model="", default_rerank_model="")
