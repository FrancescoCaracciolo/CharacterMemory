"""OpenAI-compatible chat client (works with any /v1/chat/completions server)."""

import json
import re
from typing import Any, Iterator, Optional

from openai import OpenAI

from ..config import LLMConfig
from .base import LLMClient

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM string."""
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    # Fenced code block first.
    m = _JSON_BLOCK.search(text)
    if m:
        return json.loads(m.group(1))
    # Already-JSON.
    if text[0] in "{[":
        return json.loads(text)
    # Greedy outermost braces.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"could not find JSON in response: {text[:200]!r}")


class OpenAICompatibleLLM(LLMClient):
    """Chat client over an OpenAI-compatible ``/v1`` endpoint."""

    def __init__(self, config: Optional[LLMConfig] = None, **overrides: Any) -> None:
        cfg = config or LLMConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.config = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield assistant text deltas as they arrive from the server."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    def chat_structured(
        self,
        messages: list[dict],
        schema: dict[str, Any],
        *,
        temperature: Optional[float] = None,
    ) -> dict:
        instr = (
            "Respond ONLY with a single JSON object matching this schema. "
            "No prose, no code fences.\n\nSchema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        msgs = list(messages) + [{"role": "system", "content": instr}]
        text = self.chat(msgs, temperature=0.0 if temperature is None else temperature)
        try:
            return _extract_json(text)
        except ValueError:
            # Retry once, explicitly scolding the model.
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "That was not valid JSON. Output ONLY the JSON object now."})
            return _extract_json(self.chat(msgs, temperature=0.0))
