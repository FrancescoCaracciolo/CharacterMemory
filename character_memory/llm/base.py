"""Abstract chat-completions client."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMClient(ABC):
    """Interface for chat-completion style models."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the assistant text for `messages` (openai-style dicts)."""

    @abstractmethod
    def chat_structured(
        self,
        messages: list[dict],
        schema: dict[str, Any],
        *,
        temperature: Optional[float] = None,
    ) -> dict:
        """Return a JSON object conforming (best-effort) to `schema`.

        `schema` is a JSON-schema dict describing the expected object.
        Implementations should request JSON output and parse it leniently.
        """
