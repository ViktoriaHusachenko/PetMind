"""LLM failure taxonomy used by decorators, the agent and the API status endpoint."""


class LLMError(RuntimeError):
    """Base class for provider failures that callers may turn into a deterministic fallback."""


class LLMUnavailable(LLMError):
    """Transient failure: timeout, connection error, 5xx, open circuit breaker."""


class LLMQuotaExceeded(LLMError):
    """Provider (HTTP 429) or client-side budget is exhausted; retrying now will not help."""


class LLMProviderDisabled(LLMError):
    """The requested provider is implemented but not enabled in LLM_ENABLED_PROVIDERS."""


class LLMConfigurationError(LLMError):
    """The provider is enabled but misconfigured (for example, a missing API key)."""
