"""LLM provider layer: registry, factory, Gemini adapter and reliability decorators (no network)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from petrec.llm import LLMClient, MockLLM

from app.config import LLMSettings, RateLimitSettings
from app.llm import (
    LLMClientFactory, LLMConfigurationError, LLMQuotaExceeded, LLMRuntime, LLMUnavailable, ProviderRegistry,
    ProviderState, Purpose, default_registry, tool_calls_of,
)
from app.llm.base import ProviderToolCall, assistant_tool_message
from app.llm.decorators import CircuitBreakerClient, RateLimitedClient, RetryingClient, SlidingWindowLimiter
from app.llm.providers.gemini import GeminiClient


class RateLimitError(Exception):
    status_code = 429


class APITimeoutError(Exception):
    pass


class FakeSDK:
    """Mimics `openai.OpenAI().chat.completions.create` for the Gemini adapter."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def text_reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])


class FakeCall:
    def __init__(self, call_id, name, arguments, extra=None):
        self.id, self.function = call_id, SimpleNamespace(name=name, arguments=arguments)
        self.extra = extra

    def model_dump(self):
        row = {"id": self.id, "type": "function", "function": {"name": self.function.name}}
        if self.extra:
            row["extra_content"] = self.extra
        return row


def tool_reply(*calls, content=""):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=list(calls)))])


TOOLS = [{"type": "function", "function": {"name": "check_safety", "parameters": {
    "type": "object", "properties": {"product_id": {"type": "string"}}, "additionalProperties": False}}}]


def gemini(*responses):
    sdk = FakeSDK(*responses)
    return GeminiClient(model="gemini-test", sdk_client=sdk, cache_path=None), sdk


def test_registry_lists_all_providers_and_rejects_duplicates():
    assert default_registry.names() == ("anthropic", "gemini", "ollama", "openai")
    registry = ProviderRegistry()
    registry.register(default_registry.get("gemini"))
    with pytest.raises(ValueError):
        registry.register(default_registry.get("gemini"))


@pytest.mark.parametrize("settings, state", [
    (LLMSettings(provider="disabled"), ProviderState.DISABLED),
    (LLMSettings(provider="openai", openai_api_key="k"), ProviderState.DISABLED),  # implemented, not enabled
    (LLMSettings(provider="gemini"), ProviderState.NOT_CONFIGURED),  # no key
    (LLMSettings(provider="mystery"), ProviderState.NOT_CONFIGURED),
])
def test_factory_falls_back_without_raising(settings, state):
    runtime = LLMClientFactory(settings).build_runtime()
    assert runtime.chat is None and runtime.rerank is None
    assert runtime.state is state
    assert runtime.status()["state"] == state.value


def test_factory_builds_decorated_gemini_with_separate_models():
    runtime = LLMClientFactory(LLMSettings(gemini_api_key="SECRET-KEY-123", chat_model="chat-m", rerank_model="rerank-m",
                                           cache_path=None)).build_runtime()
    assert runtime.state is ProviderState.OK
    assert isinstance(runtime.chat, CircuitBreakerClient) and isinstance(runtime.chat, LLMClient)
    assert runtime.models == {"chat": "chat-m", "rerank": "rerank-m"}
    status = runtime.status()
    assert status["budget"]["chat"]["minute_limit"] == 10 and "SECRET-KEY-123" not in str(status)


@pytest.mark.parametrize("name, settings", [
    ("openai", LLMSettings(provider="openai", enabled_providers=("openai",), openai_api_key="k", cache_path=None)),
    ("anthropic", LLMSettings(provider="anthropic", enabled_providers=("anthropic",), anthropic_api_key="k", cache_path=None)),
    ("ollama", LLMSettings(provider="ollama", enabled_providers=("ollama",), ollama_model="qwen", cache_path=None)),
])
def test_disabled_providers_switch_on_through_settings_only(name, settings):
    runtime = LLMClientFactory(settings).build_runtime()
    assert runtime.state is ProviderState.OK and runtime.provider == name
    if name == "ollama":
        assert runtime.models["chat"] == "qwen"


def test_disabled_provider_without_key_reports_not_configured():
    settings = LLMSettings(provider="openai", enabled_providers=("openai",))
    with pytest.raises(LLMConfigurationError):
        LLMClientFactory(settings).build(Purpose.CHAT)


def test_gemini_text_completion_is_deterministic_and_cached(tmp_path):
    sdk = FakeSDK(text_reply('{"items": []}'))
    client = GeminiClient(model="gemini-test", sdk_client=sdk, cache_path=tmp_path / "c.json")
    assert client.complete("sys", "user") == '{"items": []}'
    assert client.complete("sys", "user") == '{"items": []}'
    assert len(sdk.requests) == 1 and sdk.requests[0]["temperature"] == 0


def test_gemini_returns_every_tool_call_and_keeps_thought_signature():
    client, sdk = gemini(tool_reply(
        FakeCall("a", "check_safety", '{"product_id": "P1"}', {"google": {"thought_signature": "sig"}}),
        FakeCall("b", "check_safety", '{"product_id": "P2"}')))
    response = client.complete_with_tools([{"role": "user", "content": "?"}], TOOLS)
    calls = tool_calls_of(response)
    assert [c.arguments["product_id"] for c in calls] == ["P1", "P2"]
    assert response.tool_call == calls[0]
    assert "additionalProperties" not in str(sdk.requests[0]["tools"])
    message = assistant_tool_message("", calls, ["a", "b"])
    assert message["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert "extra_content" not in message["tool_calls"][1]


def test_gemini_invalid_tool_arguments_are_marked_not_crashing():
    client, _ = gemini(tool_reply(FakeCall("a", "check_safety", "{broken")))
    call = client.complete_with_tools([{"role": "user", "content": "?"}], TOOLS).tool_call
    assert "__invalid_json__" in call.arguments


@pytest.mark.parametrize("error, expected", [
    (RateLimitError("quota"), LLMQuotaExceeded),
    (APITimeoutError("slow"), LLMUnavailable),
])
def test_gemini_maps_sdk_errors(error, expected):
    client, _ = gemini(error)
    with pytest.raises(expected):
        client.complete("s", "u")


def test_core_tool_calls_still_supported():
    calls = tool_calls_of(MockLLM([ProviderToolCall("x", {})]).complete_with_tools([], []))
    assert calls[0].name == "x"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_limiter_enforces_minute_and_day_budgets():
    clock = Clock()
    limiter = SlidingWindowLimiter(RateLimitSettings(requests_per_minute=2, requests_per_day=3), clock)
    limiter.acquire(); limiter.acquire()
    with pytest.raises(LLMQuotaExceeded):
        limiter.acquire()
    clock.now = 61
    limiter.acquire()
    clock.now = 130
    with pytest.raises(LLMQuotaExceeded):  # daily budget of 3 used
        limiter.acquire()


def test_rate_limited_client_does_not_spend_budget_on_cache_hits(tmp_path):
    sdk = FakeSDK(text_reply("ok"))
    inner = GeminiClient(model="m", sdk_client=sdk, cache_path=tmp_path / "c.json")
    client = RateLimitedClient(inner, SlidingWindowLimiter(RateLimitSettings(requests_per_minute=1, requests_per_day=1)))
    assert client.complete("s", "u") == "ok"
    assert client.complete("s", "u") == "ok"  # cached: no quota error


def test_retry_only_transient_errors():
    sleeps = []
    client, sdk = gemini(APITimeoutError("t"), text_reply("ok"))
    assert RetryingClient(client, retries=2, sleep=sleeps.append).complete("s", "u") == "ok"
    assert sleeps == [0.5]
    quota, _ = gemini(RateLimitError("q"), text_reply("never"))
    with pytest.raises(LLMQuotaExceeded):
        RetryingClient(quota, retries=2, sleep=sleeps.append).complete("s", "u")
    assert sleeps == [0.5]


def test_circuit_breaker_opens_then_probes_after_cooldown():
    clock = Clock()
    client, sdk = gemini(RateLimitError("q"), RateLimitError("q"), text_reply("ok"))
    breaker = CircuitBreakerClient(client, failures=2, cooldown=30, clock=clock)
    for _ in range(2):
        with pytest.raises(LLMQuotaExceeded):
            breaker.complete("s", "u")
    with pytest.raises(LLMUnavailable):  # open: provider not called
        breaker.complete("s", "u")
    assert breaker.state() is ProviderState.QUOTA and len(sdk.requests) == 2
    clock.now = 31
    assert breaker.complete("s", "u") == "ok"
    assert breaker.state() is ProviderState.OK


def test_runtime_from_injected_client():
    runtime = LLMRuntime.from_client(MockLLM("x"))
    assert runtime.available and runtime.chat is runtime.rerank
    assert LLMRuntime.from_client(None).status()["state"] == "disabled"
