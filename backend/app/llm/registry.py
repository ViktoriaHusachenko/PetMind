"""Registry of implemented LLM providers (open for extension, closed for modification)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from petrec.llm import LLMClient

from ..config import LLMSettings
from .base import Purpose

ClientBuilder = Callable[[LLMSettings, str], LLMClient]


@dataclass(frozen=True)
class ProviderSpec:
    """How to build one provider. `build` raises LLMConfigurationError when settings are incomplete."""

    name: str
    label: str
    build: ClientBuilder
    default_chat_model: str
    default_rerank_model: str

    def default_model(self, purpose: Purpose) -> str:
        return self.default_rerank_model if purpose is Purpose.RERANK else self.default_chat_model


class ProviderRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ProviderSpec] = {}

    def register(self, spec: ProviderSpec) -> ProviderSpec:
        key = spec.name.lower()
        if key in self._specs:
            raise ValueError(f"LLM provider {spec.name!r} is already registered")
        self._specs[key] = spec
        return spec

    def get(self, name: str) -> ProviderSpec | None:
        return self._specs.get(name.lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


def _build_default_registry() -> ProviderRegistry:
    from .providers import anthropic, gemini, ollama, openai

    registry = ProviderRegistry()
    for module in (gemini, openai, anthropic, ollama):
        registry.register(module.SPEC)
    return registry


default_registry = _build_default_registry()
