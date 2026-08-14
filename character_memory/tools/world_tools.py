"""Perception-filtered read tools and the optional deferred world action."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import Tool, ToolOutput, TurnEffect

if TYPE_CHECKING:
    from ..agent import CharacterAgent


class _WorldTool(Tool):
    def __init__(self, agent: "CharacterAgent") -> None:
        self.agent = agent

    @property
    def world(self):
        world = self.agent.world_memory
        if world is None or not world.enabled:
            raise RuntimeError("WorldMemory is not enabled")
        return world


class GetWorldState(_WorldTool):
    name = "get_world_state"
    description = "Inspect the character's current perceived world state without changing it."
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self) -> dict[str, Any]:
        return self.world.snapshot(commit=False).to_dict()


class SearchWorld(_WorldTool):
    name = "search_world"
    description = "Search world facts and witnessed/public events visible to this character."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 4},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        return self.world.search_visible(query, limit=max(1, min(20, int(limit))))


class WorldAction(_WorldTool):
    name = "world_action"
    description = (
        "Stage a concrete world action. It takes effect only if the final assistant "
        "reply is successfully saved. Do not use it for hypothetical or future intent."
    )
    requires_persisted_chat = True

    def __init__(self, agent: "CharacterAgent") -> None:
        super().__init__(agent)
        features = self.world.state_store.effective_features(self.world.observer_id)
        commands = ["schedule"]
        if features["locations"]:
            commands.append("move")
        if features["activities"]:
            commands.append("start_activity")
        if features["hunger"] or features["activities"]:
            commands.append("eat")
        if features["sleep"]:
            commands.extend(["sleep", "wake"])
        self.parameters = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": commands},
                "location_id": {"type": "string"},
                "activity_kind": {"type": "string"},
                "activity": {"type": "string"},
                "until": {"type": "number"},
                "at": {"type": "number"},
                "payload": {"type": "object"},
                "dedupe_key": {"type": "string"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        }

    def run(
        self,
        kind: str,
        location_id: Optional[str] = None,
        activity_kind: Optional[str] = None,
        activity: Optional[str] = None,
        until: Optional[float] = None,
        at: Optional[float] = None,
        payload: Optional[dict[str, Any]] = None,
        dedupe_key: Optional[str] = None,
    ) -> ToolOutput:
        command = {
            "kind": kind,
            "actor_id": self.world.observer_id,
            "location_id": location_id,
            "activity_kind": activity_kind,
            "activity": activity,
            "until": until,
            "at": at,
            "payload": payload or {},
            "source": "model_tool",
            "dedupe_key": dedupe_key,
        }
        self.world.validate_command_dict(command)
        return ToolOutput(
            text="World action validated and staged for this reply.",
            data=command,
            effects=[TurnEffect("world_command", command)],
        )


def world_tools(
    agent: "CharacterAgent", *, include_actions: Optional[bool] = None
) -> list[Tool]:
    """Build fresh world tools, respecting ``allow_model_actions`` by default."""
    world = agent.world_memory
    if world is None or not world.enabled:
        return []
    tools: list[Tool] = [GetWorldState(agent), SearchWorld(agent)]
    allow = world.config.allow_model_actions if include_actions is None else bool(include_actions)
    if allow:
        tools.append(WorldAction(agent))
    return tools
