"""Pluggable LLM providers.

All providers implement :class:`LLMProvider.complete`, returning ``n`` raw JSON
strings. OpenAI and Ollama both speak the OpenAI chat-completions protocol, so a
single implementation serves both.
"""

from __future__ import annotations

from typing import Protocol

from firerag.config import Settings

# Reasoning-model families that reject a custom temperature (they sample at 1.0).
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class LLMProvider(Protocol):
    """Minimal interface the generator depends on."""

    name: str
    model: str

    def complete(self, messages: list[dict[str, str]], temperature: float, n: int = 1) -> list[str]:
        """Return ``n`` completions (raw strings, expected to be JSON)."""
        ...


class OpenAICompatibleProvider:
    """OpenAI API, or any OpenAI-compatible endpoint such as Ollama."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        supports_n: bool = True,
    ) -> None:
        """Create a client for ``model``."""
        from openai import OpenAI

        self.name = name
        self.model = model
        self.supports_n = supports_n
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete(self, messages: list[dict[str, str]], temperature: float, n: int = 1) -> list[str]:
        """Request ``n`` JSON-mode completions."""
        kwargs: dict = {"model": self.model, "messages": messages, "response_format": {"type": "json_object"}}
        if not self.model.startswith(_FIXED_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = temperature
        if self.supports_n:
            resp = self._client.chat.completions.create(n=n, **kwargs)
            return [c.message.content or "" for c in resp.choices]
        out = []
        for _ in range(n):
            resp = self._client.chat.completions.create(**kwargs)
            out.append(resp.choices[0].message.content or "")
        return out


def get_provider(settings: Settings) -> LLMProvider | None:
    """Build the configured provider, or ``None`` for retrieval-only mode."""
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAICompatibleProvider(
            "openai", settings.openai_model, settings.openai_api_key, timeout=settings.llm_timeout_s
        )
    if settings.llm_provider == "ollama":
        return OpenAICompatibleProvider(
            "ollama",
            settings.ollama_model,
            "ollama",
            base_url=settings.ollama_base_url,
            timeout=settings.llm_timeout_s,
            supports_n=False,
        )
    return None
