"""Tool-calling support for the character-memory library.

Two ways to define a tool — both produce a :class:`Tool`:

* Subclass :class:`Tool` (explicit, can hold state)::

      class SearchMemory(Tool):
          name = "search_memory"
          description = "..."
          parameters = {"type": "object", "properties": {...}, "required": [...]}
          def run(self, **kwargs) -> str: ...

* Decorate a function with :func:`tool` (schema inferred from annotations)::

      @tool
      def get_weather(city: str) -> str:
          \"\"\"Weather for a city.\"\"\"
          return f"{city}: 22C"

Collect tools in a :class:`ToolRegistry`, or just pass a ``list[Tool]`` straight
to ``CharacterAgent.generate_answer(tools=...)``. The agent runs the
model → tool → model loop and returns the final text reply (or a stream of
:class:`TextChunk` / :class:`ToolCallEvent` / :class:`ToolResultEvent` when
``stream=True``).

For a quick start, :func:`memory_tools` returns the built-in read-only tools
that let the character query its own memories mid-conversation.
"""

from .base import (
    TextChunk,
    Tool,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
    ToolOutput,
    ToolResult,
    ToolResultEvent,
    ToolStreamEvent,
    TurnEffect,
)
from .decorator import tool
from .event_tools import (
    CalculateTimeDifference,
    GetConversationEvents,
    GetEventNeighbors,
    ResolveTimeRange,
    SearchConversationEvents,
    conversation_event_tools,
    parse_time,
)
from .memory_tools import memory_tools
from .knowledge_graph_tools import (
    GetKnowledgeGraphNeighbors,
    GetKnowledgeGraphNodes,
    SearchKnowledgeGraph,
    knowledge_graph_tools,
)
from .world_tools import GetWorldState, SearchWorld, WorldAction, world_tools
from .calendar_tools import (
    SearchCalendarEvents,
    CreateCalendarEvent,
    UpdateCalendarEvent,
    EditCalendarEvent,
    CancelCalendarEvent,
    calendar_tools,
)
from .registry import (
    ToolRegistry,
    get_tool,
    global_registry,
    register_tool,
)

__all__ = [
    # Tool model
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "ToolOutput",
    "ToolResult",
    "ToolRegistry",
    "TurnEffect",
    # Definition helpers
    "tool",
    "register_tool",
    "get_tool",
    "global_registry",
    # Built-ins
    "memory_tools",
    "conversation_event_tools",
    "knowledge_graph_tools",
    "world_tools",
    "calendar_tools",
    "SearchCalendarEvents",
    "CreateCalendarEvent",
    "UpdateCalendarEvent",
    "EditCalendarEvent",
    "CancelCalendarEvent",
    "GetWorldState",
    "SearchWorld",
    "WorldAction",
    "SearchConversationEvents",
    "GetConversationEvents",
    "GetEventNeighbors",
    "ResolveTimeRange",
    "CalculateTimeDifference",
    "SearchKnowledgeGraph",
    "GetKnowledgeGraphNodes",
    "GetKnowledgeGraphNeighbors",
    "parse_time",
    # Streaming events
    "TextChunk",
    "ToolCallEvent",
    "ToolResultEvent",
    "ToolStreamEvent",
]
