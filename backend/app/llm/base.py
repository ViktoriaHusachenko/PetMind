"""Response types and small protocols shared by all PetMind LLM providers.

Every provider is a `petrec.llm.LLMClient` subclass, so it can replace any other client in the
core pipeline and the agent (Liskov substitution). Extra capabilities are opt-in protocols.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from petrec.llm import LLMResponse, ToolCall


@dataclass(frozen=True)
class MultiToolResponse(LLMResponse):
    """LLMResponse carrying every tool call of one model turn; `tool_call` stays the first one."""

    tool_calls: tuple[ToolCall, ...] = ()

    @classmethod
    def of(cls, content: str, calls: list[ToolCall] | tuple[ToolCall, ...]) -> "MultiToolResponse":
        calls = tuple(calls)
        return cls(content=content, tool_call=calls[0] if calls else None, tool_calls=calls)


def tool_calls_of(response: LLMResponse) -> tuple[ToolCall, ...]:
    """All tool calls of a response from any client, including core clients with a single call."""
    calls = getattr(response, "tool_calls", None)
    if calls:
        return tuple(calls)
    return (response.tool_call,) if response.tool_call is not None else ()


class Purpose(str, Enum):
    """Separate budgets keep the recommendation rerank from starving the chat agent."""

    CHAT = "chat"
    RERANK = "rerank"


class ProviderState(str, Enum):
    OK = "ok"
    DISABLED = "disabled"
    NOT_CONFIGURED = "not_configured"
    QUOTA = "quota"
    UNAVAILABLE = "unavailable"


@runtime_checkable
class StatusReporting(Protocol):
    """Clients and decorators that can report their runtime state without secrets."""

    def status(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ProviderToolCall(ToolCall):
    """Tool call with provider-specific fields that must be echoed back on the next turn
    (for example Gemini thought signatures in `extra_content`)."""

    provider_extra: dict[str, Any] | None = None


def assistant_tool_message(content: str, calls: tuple[ToolCall, ...], ids: list[str]) -> dict[str, Any]:
    """OpenAI-shaped assistant message for a turn with tool calls, preserving provider extras."""
    rows = []
    for call, call_id in zip(calls, ids):
        row: dict[str, Any] = {"id": call_id, "type": "function",
                               "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
        extra = getattr(call, "provider_extra", None)
        if extra:
            row.update(extra)
        rows.append(row)
    return {"role": "assistant", "content": content or "", "tool_calls": rows}
