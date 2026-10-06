# Character Memory


[![PyPI version](https://img.shields.io/pypi/v/charactermemory.svg)](https://pypi.org/project/charactermemory/)
[![Python versions](https://img.shields.io/pypi/pyversions/charactermemory.svg)](https://pypi.org/project/charactermemory/)
[![License](https://img.shields.io/pypi/l/charactermemory.svg)](https://pypi.org/project/charactermemory/)

<picture>
  <source srcset="https://github.com/user-attachments/assets/5648e15a-256d-4435-8a35-3d9864a78eab" media="(prefers-color-scheme: light)">
  <source srcset="https://github.com/user-attachments/assets/e77809e6-699f-4929-9397-9025df744ac0" media="(prefers-color-scheme: dark)">
  <img width="100%" alt="Banner character memory" src="https://github.com/user-attachments/assets/5648e15a-256d-4435-8a35-3d9864a78eab" />
</picture>

**AI Characters that live, remember and forget**

- - -
Unlike other memory systems, Character Memory is not created for perfect recall, but to **recall like a human**, **make bonds with users** and **keep track of the character's lifetime**.

Like humans, in CharacterMemory, memories are recalled based on **how emotionally impactful** an episode was, in which **location** the character is, how **recent** is the memory and how **often** he recalls it.

<table>
  <thead>
    <tr>
      <th align="center">Memory Explorer</th>
      <th align="center">World Editor</th>
      <th align="center">Knowledge Graph</th>
      <th align="center">Live Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/833f6c9d-0249-41d3-be58-a64c560536ef#gh-dark-mode-only">
          <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/17341121-cba2-4680-a343-642c618027cb#gh-light-mode-only">
          <img alt="Main View" src="https://github.com/user-attachments/assets/17341121-cba2-4680-a343-642c618027cb" width="100%">
        </picture>
      </td>
      <td align="center">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/d49b3d63-e85f-40ce-ab7e-b3a241d0d94a#gh-dark-mode-only">
          <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/b2a54804-b78b-408c-a1ed-15dd1a56bbf0#gh-light-mode-only">
          <img alt="World View" src="https://github.com/user-attachments/assets/b2a54804-b78b-408c-a1ed-15dd1a56bbf0" width="100%">
        </picture>
      </td>
      <td align="center">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/18fa52e4-5e8f-4635-b71c-3c63f23d2501#gh-dark-mode-only">
          <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/afc1aa6e-d61c-4eca-ae5a-0f0af454f8bd#gh-light-mode-only">
          <img alt="Knowledge Graph" src="https://github.com/user-attachments/assets/afc1aa6e-d61c-4eca-ae5a-0f0af454f8bd" width="100%">
        </picture>
      </td>
      <td align="center">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/7c0d3068-9df5-45f8-8394-bb6a1b9d6b8e#gh-dark-mode-only">
          <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/ce969580-662c-4d98-8bb1-f424d8425dd6#gh-light-mode-only">
          <img alt="Live Recall" src="https://github.com/user-attachments/assets/ce969580-662c-4d98-8bb1-f424d8425dd6" width="100%">
        </picture>
      </td>
    </tr>
  </tbody>
</table>


### Quick Start
**Character Memory is both a library for developers and a ready-to-integrate tool for third party applications**
It provides a:
- pip package, used by developers to integrate the library
- MCP server, to integrate tools to read and edit memory into Agents
- API to query and save memory
- A Web Interface to create characters, view memories and chat with them

#### Installation
```bash
pip install charactermemory              # the library
pip install 'charactermemory[server]'    # + FastAPI server, WebUI and MCP endpoint
```

The library talks to any OpenAI-compatible `/v1` endpoint for both the chat model and the embeddings server:
```bash
export OPENAI_BASE_URL="http://127.0.0.1:9999/v1"
export OPENAI_API_KEY="anything"
export OPENAI_MODEL="my-chat-model"
export OPENAI_EMBEDDINGS_BASE_URL="http://127.0.0.1:9999/v1"
export OPENAI_EMBEDDINGS_MODEL="my-embeddings-model"
```

#### Use it as a library
A character is just a directory: `Information/` lore files, `Dialogues/` examples, and an optional `config.yaml` with persona, prompts and memory toggles.

```python
from character_memory import CharacterAgent, LLMConfig, EmbeddingConfig, MemoryConfig

agent = CharacterAgent(directory="assets/Kurisu", name="Kurisu")
agent.load_from_config(LLMConfig(), EmbeddingConfig(), MemoryConfig())
agent.build()   # load or build the memory indexes (idempotent)

chat = agent.create_chat(user="michael", title="phonewave intro")
chat.add_message("user", "Hi, I'm Michael, a nuclear engineer called in by Daru.")

for chunk in agent.generate_answer(chat, stream=True):   # persisted + auto-extracted
    print(chunk, end="", flush=True)
```

#### Or run the server (WebUI + MCP)
```bash
charactermemory-server    # serves /context, /save, /gui and /mcp on :8000
```

Open <http://localhost:8000/gui> to create characters, browse and edit their memories and explore the knowledge graph, or point an MCP client (Claude Desktop, Cursor, …) at `http://localhost:8000/mcp?character=Kurisu`.

The full walkthrough of all four usage modes (full library, context-only, MCP server, HTTP API + WebUI) is in [docs/getting_started.md](docs/getting_started.md).

- - -

### General Idea
Character Memory provides the following built-in memory systems (which can be enabled/disabled):
1. Character Info: documents about the base character's lore, preferences and general information. Allows for lore consistency.
2. Dialogue Examples: documents about examples of conversations between the character and other characters, used to copy it conversation style
3. **Fact Memory**: save facts like "User is an engineer"
4. **Episodes Memory**: save episodes like "I talked with the user about quantum physics"
5. User directives: instructions given by the user that recall when needed, for example "When I ask you to do X, do Y". Mainly used for agent/assistant-like characters.
6. Conversation Events: just saves conversations with the user. Recall on this memory is only done when necessary
7. **Emotions**: keep track of current character's base emotions and emotions towards a user specifically
8. **World**: keep track of the character's location, routines, sleep, hunger and energy
9. **Calendar**: keep track of events
10. User Summary: a rolling summary of the user 

- **Recall** is usually done via Embeddings (Similarity) + BM25 (Lexical) search on all the memories, but also considering emotions, recency and other parameters. Some memories are made to stay regardless.
- Every 5 or 10 turns (user-defined), an **LLM extraction process begins**, which, considering already existing context, adds entries for all of the memories that need it.
- New extracted memories are compared to old memories, and a **deduplication process** begins if too similar memories or conflicting memories are detected.

#### Knowledge Graph Retrieval
Knowledge Graph Retrieval provides an aggregation of retrieval-based memories and connects them in order to provide a more advanced retrieval.
There are multiple node types:

- **Self node**: maps the character itself, central and main node of the graph.
- **Person node**: maps person and their relationship with them, activates when a person is metioned or the character is talking to that person
- **Episode node**: maps episodes
- **Fact node**: maps facts
- **Location node**: maps locations, activates automatically when the character is in that location, or the location is mentioned
- **Entity Node**: maps entities, special objects or things in the conversation

Depending on the nodes they connect, edges can map the **emotions** about an event, **relationship** with other poeple, **two events that get recalled together**, the importance and how recent an event is.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/065bd06d-43c8-48bb-8d86-ae2403fa031c#gh-dark-mode-only">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/675ce04f-8892-4852-8dca-04a086da43ae#gh-light-mode-only">
  <img alt="Knowledge Graph" src="https://github.com/user-attachments/assets/675ce04f-8892-4852-8dca-04a086da43ae" width="100%">
</picture>


At the end, 

- People with strong bonds are more likely to be recalled
- Episodes that cause strong emotions, or that caused similar emotions to current are more likely to be recalled
- Episodes that are related to a common entity are more likely to be recalled
- Episodes that happened in the same location the character currently is, are more likely to be recalled

The knowledge graph requires one additional LLM call after every extraction. (And tens of LLM calls to ingest existing memories)

### Credits and AI Disclosure
- The logo, banner and graphics are designed by [GiuliettesPhotography](https://www.instagram.com/giuliettesphotography) 
- The code architecture, memory classes, API desgin and core code is totally **human written** and coded, with the sole exception of some patches that were heavily human reviewed.
- The WebUI is currently **AI Generated**. It is mainly a viewer, so it is not considered a core part of the project.
- The server is AI Generated by heavily human reviewed
