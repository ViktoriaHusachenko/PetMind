"""Settings are parsed once from the environment with safe defaults."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings


def test_defaults_select_gemini_and_only_enable_it():
    settings = Settings.from_env({})
    assert settings.llm.provider == "gemini"
    assert settings.llm.enabled_providers == ("gemini",)
    assert settings.api_port == 8000
    assert settings.database_url.startswith("postgresql+psycopg://")


def test_environment_overrides_and_invalid_numbers_fall_back():
    settings = Settings.from_env({
        "LLM_PROVIDER": "OpenAI", "LLM_ENABLED_PROVIDERS": "gemini, openai", "API_PORT": "9000",
        "LLM_CHAT_RPM": "not-a-number", "DATABASE_URL": "postgresql://u:p@h:5432/d",
        "GOOGLE_API_KEY": "k", "PETMIND_CORS_ORIGINS": "http://a, http://b",
    })
    assert settings.llm.provider == "openai"
    assert settings.llm.enabled_providers == ("gemini", "openai")
    assert settings.api_port == 9000
    assert settings.llm.chat_limits.requests_per_minute == 10
    assert settings.database_url == "postgresql+psycopg://u:p@h:5432/d"
    assert settings.llm.gemini_api_key == "k"
    assert settings.cors_origins == ("http://a", "http://b")


def test_password_hasher_round_trip_and_rejects_garbage():
    from app.auth import PasswordHasher

    hasher = PasswordHasher(rounds=1000)
    encoded = hasher.hash("секрет")
    assert hasher.verify("секрет", encoded) and not hasher.verify("інше", encoded)
    assert not hasher.verify("секрет", None) and not hasher.verify("секрет", "md5$1$x$y")


def test_demo_catalog_is_large_unique_and_marked_as_demo():
    sys.path.insert(0, str(ROOT / "backend" / "scripts"))
    from demo_catalog import BRAND, demo_catalog

    rows = demo_catalog()
    assert len(rows) >= 40 and len({row["id"] for row in rows}) == len(rows)
    assert {row["species"] for row in rows} == {"cat", "dog"}
    assert all(row["brand"] == BRAND and row["id"].startswith("PETMIND-DEMO-") for row in rows)
