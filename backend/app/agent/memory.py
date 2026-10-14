"""Bounded, sanitized conversation history sent to the LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALLOWED_ROLES = frozenset({"user", "assistant"})


@dataclass(frozen=True)
class ConversationWindow:
    max_messages: int = 12
    max_chars: int = 6000

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Keep the most recent user/assistant turns within the budget; system prompts are server-only."""
        clean = [{"role": m["role"], "content": str(m.get("content", ""))}
                 for m in messages if m.get("role") in ALLOWED_ROLES and str(m.get("content", "")).strip()]
        window: list[dict[str, str]] = []
        used = 0
        for message in reversed(clean[-self.max_messages:]):
            size = len(message["content"])
            if window and used + size > self.max_chars:
                break
            window.append({**message, "content": message["content"][: self.max_chars]})
            used += size
        window.reverse()
        while window and window[0]["role"] != "user":  # a window must start with a user turn
            window.pop(0)
        return window
