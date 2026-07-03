"""Abstract chat-completions client."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional


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

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield assistant text chunks for `messages`.

        Default implementation (non-streaming fallback) yields the full
        :meth:`chat` reply at once, so every `LLMClient` works with
        ``generate_answer(stream=True)`` out of the box. Override to emit
        real incremental deltas from the backend.
        """
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens)
