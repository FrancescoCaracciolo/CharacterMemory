# Character Memory
<img width="927" height="339" alt="Screenshot 2026-09-02 alle 23 19 44" src="https://github.com/user-attachments/assets/d825da4e-d747-4942-819c-7627ec180ebf" />



**AI Characters that live, remember and forget**

- - -

Unlike other memory systems, Character Memory is not created for perfect recall, but to **recall like a human**, **make bonds with users** and **keep track of the character's lifetime**.

Like humans, in CharacterMemory, memories are recalled based on **how emotionally impactful** an episode was, in which **location** the character is, how **recent** is the memory and how **often** he recalls it.


### Quick Start
**Character Memory is both a library for developers and a ready-to-integrate tool for third party applications**
It provides a:
- pip package, used by developers to integrate the library
- MCP server, to integrate tools to read and edit memory into Agents
- API to query and save memory
- A Web Interface to create characters, view memories and chat with them

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


<img width="570" height="603" alt="image" src="https://github.com/user-attachments/assets/d7e67631-d307-4765-ac8d-2ddbfefc8931" />


At the end, 

- People with strong bonds are more likely to be recalled
- Episodes that cause strong emotions, or that caused similar emotions to current are more likely to be recalled
- Episodes that are related to a common entity are more likely to be recalled
- Episodes that happened in the same location the character currently is, are more likely to be recalled

The knowledge graph requires one additional LLM call after every extraction. (And tens of LLM calls to ingest existing memories)
