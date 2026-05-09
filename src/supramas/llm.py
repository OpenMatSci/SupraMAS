"""LLM interface placeholders used by the public SupraMAS skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMSettings:
    """Configuration shape for an OpenAI-compatible model endpoint."""

    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7


class LLMProvider(Protocol):
    """Minimal interface expected by SupraMAS agents."""

    def response(self, messages: list[dict], **kwargs) -> str:
        """Return a model response for chat-style messages."""


class ExternalLLMPlaceholder:
    """Placeholder for the production LLM adapter.

    The public release keeps the interface visible while omitting private
    gateway handling, retry policies, and provider-specific logic.
    """

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()

    def response(self, messages: list[dict], **kwargs) -> str:
        raise NotImplementedError(
            "The public skeleton does not include production LLM execution. "
            "Use the manuscript data and benchmark outputs for review."
        )


def get_llm(settings: LLMSettings | None = None) -> ExternalLLMPlaceholder:
    """Build the public placeholder LLM provider."""

    return ExternalLLMPlaceholder(settings=settings)


openai_llm = ExternalLLMPlaceholder

