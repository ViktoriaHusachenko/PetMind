"""Typed application settings, read once from the environment and the repository `.env`."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("petmind.config")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        logger.warning("%s must be an integer; using %s", key, default)
        return default


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        logger.warning("%s must be a number; using %s", key, default)
        return default


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in (value or "").split(",") if part.strip())


class RateLimitSettings(BaseModel):
    """Client-side budget for one LLM purpose (chat or rerank)."""

    model_config = ConfigDict(frozen=True)
    requests_per_minute: int = Field(default=10, ge=0)
    requests_per_day: int = Field(default=500, ge=0)


class LLMSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = "gemini"
    # Implemented providers stay switched off unless listed here.
    enabled_providers: tuple[str, ...] = ("gemini",)
    chat_model: str | None = None
    rerank_model: str | None = None
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    cache_path: str = ".cache/petrec_llm_cache.json"
    timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    breaker_failures: int = Field(default=3, ge=1)
    breaker_cooldown_seconds: float = Field(default=60.0, ge=0)
    chat_limits: RateLimitSettings = RateLimitSettings(requests_per_minute=10, requests_per_day=400)
    rerank_limits: RateLimitSettings = RateLimitSettings(requests_per_minute=5, requests_per_day=200)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    database_url: str = "postgresql+psycopg://petly:petly@localhost:5432/petly"
    api_port: int = 8000
    auth_secret: str | None = None
    admin_email: str = "admin@petmind.local"
    admin_password: str = "PetMindAdmin2026!"
    demo_password: str = "PetMindDemo2026!"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    recommendation_ttl_seconds: int = 60
    llm: LLMSettings = LLMSettings()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        legacy_model = env.get("LLM_MODEL") or None
        llm = LLMSettings(
            provider=(env.get("LLM_PROVIDER") or "gemini").strip().lower(),
            enabled_providers=_csv(env.get("LLM_ENABLED_PROVIDERS")) or ("gemini",),
            chat_model=env.get("LLM_CHAT_MODEL") or legacy_model,
            rerank_model=env.get("LLM_RERANK_MODEL") or legacy_model,
            gemini_api_key=env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or None,
            gemini_base_url=env.get("GEMINI_BASE_URL") or LLMSettings.model_fields["gemini_base_url"].default,
            openai_api_key=env.get("OPENAI_API_KEY") or None,
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
            ollama_host=env.get("OLLAMA_HOST") or env.get("OLLAMA_BASE_URL") or "http://localhost:11434",
            ollama_model=env.get("OLLAMA_MODEL") or "llama3.1",
            cache_path=env.get("LLM_CACHE_PATH") or ".cache/petrec_llm_cache.json",
            timeout_seconds=_float(env, "LLM_TIMEOUT_SECONDS", 20.0),
            max_retries=_int(env, "LLM_MAX_RETRIES", 2),
            breaker_failures=_int(env, "LLM_BREAKER_FAILURES", 3),
            breaker_cooldown_seconds=_float(env, "LLM_BREAKER_COOLDOWN_SECONDS", 60.0),
            chat_limits=RateLimitSettings(requests_per_minute=_int(env, "LLM_CHAT_RPM", 10),
                                          requests_per_day=_int(env, "LLM_CHAT_RPD", 400)),
            rerank_limits=RateLimitSettings(requests_per_minute=_int(env, "LLM_RERANK_RPM", 5),
                                            requests_per_day=_int(env, "LLM_RERANK_RPD", 200)),
        )
        url = env.get("DATABASE_URL") or cls.model_fields["database_url"].default
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return cls(
            database_url=url,
            api_port=_int(env, "API_PORT", 8000),
            auth_secret=env.get("PETMIND_AUTH_SECRET") or None,
            admin_email=(env.get("PETMIND_ADMIN_EMAIL") or "admin@petmind.local").strip().casefold(),
            admin_password=env.get("PETMIND_ADMIN_PASSWORD") or "PetMindAdmin2026!",
            demo_password=env.get("PETMIND_DEMO_PASSWORD") or "PetMindDemo2026!",
            cors_origins=tuple(o.strip() for o in (env.get("PETMIND_CORS_ORIGINS") or "").split(",") if o.strip())
            or cls.model_fields["cors_origins"].default,
            recommendation_ttl_seconds=_int(env, "RECOMMENDATION_TTL_SECONDS", 60),
            llm=llm,
        )


def load_env_file(path: Path | None = None) -> None:
    """Load the repository `.env` regardless of the working directory; real env vars win."""
    load_dotenv(path or PROJECT_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_env_file()
    return Settings.from_env()
