"""Provider-neutral LLM clients with deterministic temperature and disk caching."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass(frozen=True)
class LLMResponse:
    content: str = ""
    tool_call: ToolCall | None = None


class LLMClient(ABC):
    """Base API for text completions and function/tool calls."""

    def __init__(self, cache_path: str | Path | None = ".cache/petrec_llm_cache.json") -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    @abstractmethod
    def _complete(self, system: str, user: str) -> str:
        """Provider implementation of text completion."""

    def complete(self, system: str, user: str) -> str:
        key = self._key("complete", {"system": system, "user": user})
        if key not in self._cache:
            self._cache[key] = self._complete(system, user)
            self._save_cache()
        return str(self._cache[key])

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Default fallback; providers override this to execute native tool calling."""
        user = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True)
        return LLMResponse(content=self.complete("Use the supplied tool definitions when appropriate.", user))

    def _key(self, operation: str, payload: Any) -> str:
        body = json.dumps({"client": type(self).__name__, "operation": operation, "model": getattr(self, "model", ""),
                           "payload": payload}, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.cache_path)


class MockLLM(LLMClient):
    """Test client accepting response strings or tool-call response objects."""

    def __init__(self, response: str | list[Any] = '{"items": []}', *, cache_path: str | Path | None = None) -> None:
        super().__init__(cache_path)
        self.responses = [response] if isinstance(response, str) else list(response)
        self.calls: list[tuple[str, str]] = []
        self.tool_calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def _complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise RuntimeError("MockLLM has no responses remaining")
        response = self.responses.pop(0)
        return response if isinstance(response, str) else str(response)

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.tool_calls.append((messages, tools))
        if not self.responses:
            raise RuntimeError("MockLLM has no responses remaining")
        response = self.responses.pop(0)
        if isinstance(response, LLMResponse):
            return response
        if isinstance(response, ToolCall):
            return LLMResponse(tool_call=response)
        if isinstance(response, Mapping) and "tool_call" in response:
            call = response["tool_call"]
            return LLMResponse(content=str(response.get("content", "")),
                                tool_call=ToolCall(call["name"], dict(call.get("arguments", {})), call.get("id", "")))
        return LLMResponse(content=str(response))


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 cache_path: str | Path | None = ".cache/petrec_llm_cache.json") -> None:
        super().__init__(cache_path)
        self.model, self.api_key = model, api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, timeout=20.0, max_retries=0)

    def _complete(self, system: str, user: str) -> str:
        result = self._client().chat.completions.create(model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return result.choices[0].message.content or ""

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        key = self._key("tools", {"messages": messages, "tools": tools})
        if key in self._cache:
            value = self._cache[key]
            call = ToolCall(**value["tool_call"]) if value.get("tool_call") else None
            return LLMResponse(value.get("content", ""), call)
        result = self._client().chat.completions.create(model=self.model, temperature=0,
            messages=messages, tools=tools or None)
        message = result.choices[0].message
        calls = getattr(message, "tool_calls", None)
        call = None
        if calls:
            raw = calls[0]
            call = ToolCall(raw.function.name, json.loads(raw.function.arguments), raw.id)
        response = LLMResponse(message.content or "", call)
        self._cache[key] = {"content": response.content, "tool_call": asdict(call) if call else None}
        self._save_cache()
        return response


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-3-5-haiku-latest", api_key: str | None = None,
                 cache_path: str | Path | None = ".cache/petrec_llm_cache.json") -> None:
        super().__init__(cache_path)
        self.model, self.api_key = model, api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

    def _client(self):
        from anthropic import Anthropic
        return Anthropic(api_key=self.api_key, timeout=20.0, max_retries=0)

    def _complete(self, system: str, user: str) -> str:
        result = self._client().messages.create(model=self.model, max_tokens=1200, temperature=0,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(part.text for part in result.content if getattr(part, "type", None) == "text")

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        key = self._key("tools", {"messages": messages, "tools": tools})
        if key in self._cache:
            value = self._cache[key]
            return LLMResponse(value.get("content", ""), ToolCall(**value["tool_call"]) if value.get("tool_call") else None)
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        # Agent conversations use the common OpenAI tool-call message shape.
        # Convert it to Anthropic's assistant tool_use / user tool_result blocks.
        user_messages: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                pending_results.append({"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""),
                                        "content": str(message.get("content", ""))})
                continue
            if pending_results:
                user_messages.append({"role": "user", "content": pending_results})
                pending_results = []
            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": str(message["content"])})
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                    except (TypeError, json.JSONDecodeError):
                        arguments = {}
                    blocks.append({"type": "tool_use", "id": tool_call.get("id", ""),
                                   "name": function.get("name", ""), "input": arguments})
                user_messages.append({"role": "assistant", "content": blocks})
            else:
                user_messages.append({"role": role, "content": message.get("content", "")})
        if pending_results:
            user_messages.append({"role": "user", "content": pending_results})
        native_tools = [{"name": t.get("name") or t.get("function", {}).get("name"),
                         "description": t.get("description") or t.get("function", {}).get("description", ""),
                         "input_schema": t.get("input_schema") or t.get("function", {}).get("parameters", {"type": "object", "properties": {}})} for t in tools]
        result = self._client().messages.create(model=self.model, max_tokens=1200, temperature=0,
            system=system, messages=user_messages, tools=native_tools or None)
        content, call = "", None
        for part in result.content:
            if getattr(part, "type", None) == "text": content += part.text
            elif getattr(part, "type", None) == "tool_use": call = ToolCall(part.name, dict(part.input), part.id)
        response = LLMResponse(content, call)
        self._cache[key] = {"content": content, "tool_call": asdict(call) if call else None}
        self._save_cache()
        return response


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1", host: str | None = None,
                 cache_path: str | Path | None = ".cache/petrec_llm_cache.json") -> None:
        super().__init__(cache_path)
        self.model, self.host = model, host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        import requests
        response = requests.post(f"{self.host.rstrip('/')}/api/chat",
            json={"model": self.model, "stream": False, "options": {"temperature": 0}, "messages": messages, "tools": tools or []}, timeout=20)
        response.raise_for_status()
        return response.json()

    def _complete(self, system: str, user: str) -> str:
        data = self._request([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return str(data.get("message", {}).get("content", ""))

    def complete_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        key = self._key("tools", {"messages": messages, "tools": tools})
        if key in self._cache:
            value = self._cache[key]
            return LLMResponse(value.get("content", ""), ToolCall(**value["tool_call"]) if value.get("tool_call") else None)
        data = self._request(messages, tools)
        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []
        call = None
        if tool_calls:
            raw = tool_calls[0]
            function = raw.get("function", {})
            call = ToolCall(function.get("name", ""), function.get("arguments", {}), raw.get("id", ""))
        response = LLMResponse(str(message.get("content", "")), call)
        self._cache[key] = {"content": response.content, "tool_call": asdict(call) if call else None}
        self._save_cache()
        return response
