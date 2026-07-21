"""OpenAI-compatible chat client (works with any /v1/chat/completions server)."""

import json
import re
from typing import Any, Iterator, Optional

from openai import OpenAI

from ..config import LLMConfig
from ..tools.base import TextChunk, ToolCall, ToolCallEvent
from .base import LLMClient, LLMResponse

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

    # ------------------------------------------------------------------ #
    # Tool calling
    # ------------------------------------------------------------------ #
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """One OpenAI tool-enabled round → :class:`LLMResponse` (text + tool_calls).

        ``tools`` is passed verbatim; render entries with
        :meth:`character_memory.tools.Tool.schema`. ``tool_choice`` is forwarded
        only when not ``None`` (letting the API default apply). Tool-call
        ``function.arguments`` is parsed leniently: empty → ``{}``, malformed
        JSON → the raw string wrapped so the agent can surface the error
        instead of crashing.
        """
        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw_args = getattr(getattr(tc, "function", None), "arguments", "")
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {"_value": args}
            except json.JSONDecodeError:
                # Keep the raw payload so the tool can report the bad input;
                # wrapping under a key preserves dict-shape for `**kwargs`.
                args = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=getattr(getattr(tc, "function", None), "name", "") or "",
                    arguments=args,
                )
            )
        return LLMResponse(content=content, tool_calls=calls)

    def chat_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Any]:
        """Stream a tool-enabled round as :class:`TextChunk` / :class:`ToolCallEvent`.

        Text deltas are forwarded as :class:`TextChunk` events as they arrive.
        OpenAI streams tool-call arguments in fragments keyed by ``index``; we
        accumulate them and, once the stream closes, emit a single
        :class:`ToolCallEvent` carrying the completed calls. The agent then
        executes the tools and re-prompts — tool execution is never streamed.
        """
        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        stream = self._client.chat.completions.create(**kwargs)
        # Per-index accumulator: the API sends name + arguments in pieces, and
        # the id sometimes arrives on the first fragment only.
        ids: dict[int, str] = {}
        names: dict[int, str] = {}
        arg_bufs: dict[int, list[str]] = {}
        seen_index_order: list[int] = []
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            # Forward text deltas immediately.
            text = getattr(delta, "content", None)
            if text:
                yield TextChunk(text)
            tcs = getattr(delta, "tool_calls", None) or []
            for tc in tcs:
                idx = getattr(tc, "index", 0)
                if idx not in arg_bufs:
                    arg_bufs[idx] = []
                    seen_index_order.append(idx)
                if getattr(tc, "id", None):
                    ids[idx] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        names[idx] = fn.name
                    frag = getattr(fn, "arguments", None)
                    if frag:
                        arg_bufs[idx].append(frag)
        # Emit one ToolCallEvent with the fully-assembled calls (in the order
        # they first appeared).
        calls: list[ToolCall] = []
        for idx in seen_index_order:
            raw_args = "".join(arg_bufs.get(idx, []))
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {"_value": args}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            calls.append(
                ToolCall(id=ids.get(idx, ""), name=names.get(idx, ""), arguments=args)
            )
        if calls:
            yield ToolCallEvent(calls)
