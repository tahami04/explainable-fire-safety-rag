"""Pluggable LLM providers.

All providers implement :class:`LLMProvider.complete`, returning ``n`` raw JSON
strings. OpenAI and Ollama both speak the OpenAI chat-completions protocol, so a
single implementation serves both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from openai import BadRequestError

from firerag.config import Settings


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
        self.supports_temperature = True  # set False if the API rejects it (some reasoning models)
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete(self, messages: list[dict[str, str]], temperature: float, n: int = 1) -> list[str]:
        """Request ``n`` JSON-mode completions."""
        kwargs: dict = {"model": self.model, "messages": messages, "response_format": {"type": "json_object"}}
        if self.supports_temperature:
            kwargs["temperature"] = temperature
        try:
            return self._create(kwargs, n)
        except BadRequestError as exc:
            if "temperature" not in str(exc) or not self.supports_temperature:
                raise
            # Model only samples at its default temperature; remember and retry.
            self.supports_temperature = False
            kwargs.pop("temperature")
            return self._create(kwargs, n)

    def _create(self, kwargs: dict, n: int) -> list[str]:
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


def _call_key(model: str, messages: list[dict[str, str]], temperature: float, n: int) -> str:
    payload = json.dumps({"m": model, "msgs": messages, "t": temperature, "n": n}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class RecordingProvider:
    """Wraps a provider and appends every call's outputs to a JSONL file.

    Used by the evaluation so that changes to the verification layer can be
    re-scored later against the exact same generations, with no API calls.
    """

    def __init__(self, inner: LLMProvider, path: Path) -> None:
        """Record calls made through ``inner`` to ``path``."""
        self.inner, self.path = inner, path
        self.name, self.model = inner.name, inner.model

    def complete(self, messages: list[dict[str, str]], temperature: float, n: int = 1) -> list[str]:
        """Call the wrapped provider and record the outputs."""
        out = self.inner.complete(messages, temperature, n)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"key": _call_key(self.model, messages, temperature, n), "outputs": out}) + "\n")
        return out


class ReplayProvider:
    """Serves previously recorded outputs; raises if a call was never recorded."""

    def __init__(self, path: Path, name: str, model: str) -> None:
        """Load recordings from ``path`` for ``model``."""
        self.name, self.model = name, model
        self._calls: dict[str, list[str]] = {}
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            self._calls[rec["key"]] = rec["outputs"]

    def complete(self, messages: list[dict[str, str]], temperature: float, n: int = 1) -> list[str]:
        """Return the recorded outputs for this exact call."""
        key = _call_key(self.model, messages, temperature, n)
        if key not in self._calls:
            raise KeyError("No recorded LLM output for this call (prompt, evidence or settings changed).")
        return self._calls[key]
