"""Live smoke test against the Gemini free tier; skipped unless GEMINI_API_KEY is set."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from app.config import Settings, load_env_file
from app.llm import LLMClientFactory, Purpose, tool_calls_of

load_env_file()
pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")]


def test_gemini_text_and_tool_call():
    settings = Settings.from_env().llm.model_copy(update={"provider": "gemini", "cache_path": None})
    client = LLMClientFactory(settings).build(Purpose.CHAT)
    assert client.complete("Відповідай одним словом.", "Столиця України?").strip()
    tools = [{"type": "function", "function": {"name": "get_reminders", "description": "Нагадування для тварини.",
                                               "parameters": {"type": "object", "properties": {}}}}]
    response = client.complete_with_tools(
        [{"role": "system", "content": "Використовуй інструменти."},
         {"role": "user", "content": "Покажи нагадування для мого кота."}], tools)
    assert [call.name for call in tool_calls_of(response)] == ["get_reminders"]
