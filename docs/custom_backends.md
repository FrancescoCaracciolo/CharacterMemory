# Custom backends

Every external dependency of `character_memory` is an **abstract base class
with one reference implementation**. Swapping any of them is a single new
subclass — nothing else in the library changes.

| Seam | ABC | Reference impl | Pass it via |
|---|---|---|---|
| LLM | `llm/LLMClient` | `OpenAICompatibleLLM` | `CharacterAgent(llm=…)` or `load(llm, …)` |
| Embeddings | `llm/EmbeddingProvider` | `OpenAICompatibleEmbeddings` | `CharacterAgent(embedder=…)` or `load(…, embedder, …)` |
| Retrieval | `rag/RAGSystem` | `HybridSearch` | pass into any memory's constructor |
| Global selection | `MemoryReranker` | `ScoreMemoryReranker` | constructor `reranker=` or per-call override on recall/context/generation |
| Chunking | `chunking/Chunker` + `registry.py` | `header`, `dialogue` | `ChunkingConfig.*_chunker` or `register_chunker` |
| Memory | `memories/Memory` | 9 built-ins | `load(llm, embedder, memories=[…])` |
| Tool | `tools/base.Tool` + `@tool` | memory self-tools | `generate_answer(tools=…)` — see [Tools](tools.md) |

All ABCs are re-exported from the package top level:

```python
from character_memory import (
    LLMClient, EmbeddingProvider, RAGSystem, HybridSearch,
    Chunker, get_chunker, register_chunker, Memory, StructuredMemory, Tool,
    MemoryReranker, ScoreMemoryReranker, MemoryCandidate,
)
```

---

## 1. Custom LLM — `LLMClient`

`character_memory/llm/base.py` defines:

```python
class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages, *, temperature=None, max_tokens=None) -> str: ...
    @abstractmethod
    def chat_structured(self, messages, schema, *, temperature=None) -> dict: ...

    def chat_stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        # default: yield the full chat() reply at once — override for real streaming
        ...

    # Tool-calling (opt-in; default raises NotImplementedError):
    def chat_with_tools(self, messages, tools, *, tool_choice=None,
                        temperature=None, max_tokens=None) -> LLMResponse:
        raise NotImplementedError(...)
    def chat_with_tools_stream(self, messages, tools, *, tool_choice=None,
                               temperature=None, max_tokens=None) -> Iterator[ToolStreamEvent]:
        # default: delegate to chat_with_tools once and emit TextChunk / ToolCallEvent
        ...
```

You must implement `chat` and `chat_structured`. `chat_stream` has a default
that yields the whole reply at once — override it for real incremental
streaming. Tool-calling methods are **opt-in**: they raise
`NotImplementedError` by default so a backend that can't do tool calls fails
loudly rather than silently ignoring the `tools=` argument.

`chat_structured` is used by the extractor and the deduplicator. It receives
a JSON-schema dict and should return a parsed JSON object (best-effort). The
bundled impl asks for JSON output, leniently extracts JSON from a fenced
block / outermost braces, and retries once on failure.

### Example: Anthropic / Gemini / your own backend

```python
from typing import Any, Iterator, Optional
from character_memory import LLMClient, LLMResponse

class AnthropicLLM(LLMClient):
    def __init__(self, api_key: str, model: str): ...

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        # …call your backend, return the assistant text…
        return text

    def chat_structured(self, messages, schema, *, temperature=None) -> dict:
        # …ask for JSON conforming to `schema`, parse it, return a dict…
        return parsed

    def chat_stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        # …yield incremental deltas…
        yield chunk

    # Optional — only if you want tool calling:
    def chat_with_tools(self, messages, tools, *, tool_choice=None,
                        temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(content=text, tool_calls=[ToolCall(id=…, name=…, arguments=…)])
```

### Wiring it in

```python
agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
agent.load_from_config("assets/Kurisu/config.yaml", llm=AnthropicLLM(...))
# or DIY:
agent.load(AnthropicLLM(...), embedder, memories)
```

---

## 2. Custom embeddings — `EmbeddingProvider`

`character_memory/llm/embedding_base.py` defines:

```python
class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: str | list[str]) -> np.ndarray:  # shape (n, dim), float32
        ...

    def embed_queries(self, texts):
        return self.embed(texts)

    def embed_documents(self, texts):
        return self.embed(texts)

    @property
    def index_fingerprint(self) -> str | None:
        return None
```

`embed` must accept a single string or a list and return a `(n, dim)`
float32 `numpy` array. `dim` must be stable for the lifetime of the provider
(it's compared against FAISS index dims on load — see gotcha #2 in
[Architecture](architecture.md)).

Symmetric providers only need `embed`. Asymmetric providers override
`embed_queries` / `embed_documents`; the RAG layer selects the semantic role
without knowing which model is behind it. Providers may expose a safe, stable
`index_fingerprint` so persisted indexes are rebuilt when model behavior
changes without a dimension change.

### Example: local sentence-transformers

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from character_memory import EmbeddingProvider

class STEmbedder(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._m = SentenceTransformer(model_name)
        self._dim = self._m.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._m.encode(list(texts), convert_to_numpy=True).astype(np.float32)
```

### Wiring it in

```python
agent.load_from_config("assets/Kurisu/config.yaml", embedder=STEmbedder())
# or DIY:
agent.load(llm, STEmbedder(), memories)
```

> If you switch embedding models or retrieval-role settings on an existing
> character, `HybridSearch.load()` rebuilds the dense index from `nodes.json`
> when its dimension or fingerprint metadata differs. You don't need to wipe
> anything, but the first load after the switch will be slower.

---

## 3. Custom retrieval — `RAGSystem`

`character_memory/rag/base.py` defines:

```python
class RAGSystem(ABC):
    name: str = "base"

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None: ...
    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> None: ...
    @abstractmethod
    def search(self, query: Query, k: int = 5, where: dict | None = None) -> list[Hit]: ...
    @abstractmethod
    def persist(self, path: str) -> None: ...
    @abstractmethod
    def load(self, path: str) -> None: ...

    @property
    def count(self) -> int: return 0
    @property
    def documents(self) -> list[Chunk]: return []
```

`Query` is either a plain `str` (single query, weight 1.0) **or** a
`list[tuple[str, float]]` (one `(text, weight)` pair per recent chat message,
older ones weighted less). A serious backend fuses them with weight-scaled
reciprocal rank fusion; a minimal backend can just run the first query. Use
`as_queries(query)` (from `rag.base`) to normalize.

`where` is a metadata-equality filter (e.g. `{"user_id": "alice"}`).
Structured memories rely on it heavily.

### Example: a vector-only backend

```python
import numpy as np
from character_memory import RAGSystem, Hit, Chunk, as_queries

class VectorOnly(RAGSystem):
    def __init__(self, embedder): 
        self.embedder = embedder
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._vecs = None

    def build(self, chunks): self._texts, self._meta = [], []; self.add_documents(chunks)
    def add_documents(self, chunks):
        for c in chunks:
            self._texts.append(c.text); self._meta.append(dict(c.metadata))
        self._vecs = self.embedder.embed(self._texts).astype(np.float32)

    def search(self, query, k=5, where=None):
        if not self._texts: return []
        hits = []
        for (q, w) in as_queries(query):
            qv = self.embedder.embed(q).astype(np.float32).reshape(1, -1)
            sims = (self._vecs @ qv[0])
            for i, s in enumerate(sims):
                if where and not all(self._meta[i].get(k) == v for k, v in where.items()):
                    continue
                hits.append((i, w * float(s)))
        # fuse by max weighted score per doc, take top k
        best: dict[int, float] = {}
        for i, s in hits: best[i] = max(best.get(i, 0.0), s)
        return [Hit(text=self._texts[i], score=s, metadata=self._meta[i])
                for i, s in sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    def persist(self, path): ...   # write self._texts + self._meta
    def load(self, path): ...      # read them back, re-embed
    @property
    def count(self): return len(self._texts)
    @property
    def documents(self): return [Chunk(text=t, metadata=m) for t, m in zip(self._texts, self._meta)]
```

### Wiring it in

RAG memories take a `RAGSystem` in their constructor. To swap globally,
build memories yourself:

```python
from character_memory import (
    CharacterAgent, CharacterInfoMemory, DialogueStyleMemory, UserFactMemory,
    # …the rest of the standard memories…
)
my_rag = lambda: VectorOnly(embedder)
agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu").load(
    llm, embedder,
    memories=[
        CharacterInfoMemory(my_rag()),
        DialogueStyleMemory(my_rag()),
        UserFactMemory(store, my_rag()),
        # …etc…
    ],
)
agent.build()
```

---

## 4. Custom chunker — `Chunker`

`character_memory/chunking/base.py` defines:

```python
class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, source: str = "") -> list[Chunk]: ...

    def chunk_files(self, files: list[str]) -> list[Chunk]: ...     # has a default
    def chunk_directory(self, directory: str) -> list[Chunk]: ...   # has a default
```

`Chunk` is a `(text, source, metadata)` dataclass.

### Registering it

Two equivalent ways:

```python
from character_memory import Chunker, Chunk, register_chunker

class CodeChunker(Chunker):
    name = "code"
    def chunk(self, text, source=""):
        # split by top-level def/class, attach header metadata, etc.
        return [Chunk(text=p, source=source, metadata={"header": …}) for p in parts]

register_chunker("code", CodeChunker)   # now selectable by name
```

Then point the config at it:

```python
# in code:
ChunkingConfig(info_chunker="code")
# in config.yaml:
chunking:
  info_chunker: code
```

`get_chunker(name, **kwargs)` instantiates the registered factory; only the
`info_chunker` and `dialogue_chunker` knobs are used by the agent today
(wiki and example-dialogue indexing respectively).

The bundled chunkers:

- `header` → `MarkdownByHeaderChunker` — splits Markdown by `#`/`##` headers
  with a token budget (`header_max_tokens` / `header_min_tokens`).
- `dialogue` → `DialogueChunker` — groups consecutive turns
  (`dialogue_turns_per_chunk`) with surrounding context
  (`dialogue_context_width`).

---

## 5. Custom memory — `Memory` / `StructuredMemory`

This is the biggest extension surface. See [Memory systems](memories/index.md)
for the full ABC and the two DRY bases. In short:

- **Standalone storage** → subclass `Memory` directly; implement
  `recall` / `build` / `persist` / `load` and optionally `extraction_spec` +
  `apply_extraction`.
- **SQLite rows + decay + RAG** → subclass `StructuredMemory`; declare
  `table` / `extra_columns` / `text_column` and implement `row_text` /
  `row_item`. You get `add` / `recall` / `rebuild_index` / decay / extraction
  plumbing for free.

```python
from character_memory import StructuredMemory, ExtractionSpec, MemoryItem

class PreferenceMemory(StructuredMemory):
    name = "preferences"
    table = "preferences"
    scope = MemoryScope.PER_USER
    extra_columns = {"content": "TEXT NOT NULL", "score": "REAL NOT NULL DEFAULT 0.0"}
    text_column = "content"

    def row_text(self, row): return row.get("content", "")
    def row_item(self, row, score): return MemoryItem(text=row["content"], score=score,
                                                       kind=self.name, metadata=dict(row))

    def extraction_spec(self, context=None) -> ExtractionSpec:
        return ExtractionSpec(
            field="preferences",
            per_user=True,
            schema={"type": "array", "items": {"type": "object", …}},
            instruction="…",
        )

    def apply_extraction(self, value, user_id):
        for p in value or []:
            self.add(user_id, importance=0.5, content=p["content"], score=p["score"])
        return […]
```

Then pass it via `agent.load(llm, embedder, memories=[…])`. If you also want
it surfaced in the WebUI / MCP server, the generic `structured` renderer
handles it automatically; per-`kind` views can be added in `server/static/app.js`.

> A memory only opts into extraction when it returns a non-`None`
> `ExtractionSpec`. The agent skips disabled memories entirely.

---

## 6. Custom tool — `Tool`

See [Tools](tools.md) for the full guide. Short version:

```python
from character_memory import Tool, tool

# Option A: subclass (stateful, e.g. closing over the agent)
class GetWeather(Tool):
    name = "get_weather"
    description = "Get the current weather for a city."
    parameters = {"type": "object",
                  "properties": {"city": {"type": "string"}},
                  "required": ["city"]}
    def run(self, city: str) -> str:
        return fetch_weather(city)

# Option B: decorator (stateless, schema inferred from annotations)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return fetch_weather(city)
```

Pass either to `generate_answer(tools=[…])`.

## 7. Custom global memory selection — `MemoryReranker`

Implement `rerank(query, candidates, *, budget)` and return an ordered subset
of the supplied `MemoryCandidate` objects. Candidates carry the source memory,
original item/score, standalone rendering and token cost. The character
enforces the cap against the actual grouped rendering, then records only
selected exposure. Do not mutate candidates or return duplicates/replacements.

For custom memory classes, `record_recall(items)` is the post-selection
bookkeeping hook; `StructuredMemory` already implements it. Override
`format_selection(items, participants)` when a selected subset needs rendering
beyond the standard `format`/`format_grouped` paths. `prepare_recall()` handles
lifecycle work independent of exposure. Their default implementations preserve
subclass compatibility. See [Memory budget & reranking](memory_budget.md) for
complete examples, candidate fields, and tokenizer injection.
