"""Events the agent loop emits (Observer pattern).

The loop never prints or renders anything itself. It yields these events, and each front end
(cli.py, the web UI via service.py) decides how to show them. That keeps the loop testable and
lets two very different UIs share it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class Event:
    type: ClassVar[str] = "event"

    def to_dict(self) -> dict:
        return {"type": self.type, **asdict(self)}


@dataclass
class TextDelta(Event):
    """A piece of the assistant's visible answer, streamed as it is generated."""
    type: ClassVar[str] = "text"
    text: str


@dataclass
class ReasoningDelta(Event):
    """A piece of the model's reasoning ("thinking"), for models that expose it."""
    type: ClassVar[str] = "reasoning"
    text: str


@dataclass
class ToolCall(Event):
    """The model asked to run a tool. `summary` is a one-line, human-readable form of the call."""
    type: ClassVar[str] = "tool_call"
    id: str
    name: str
    arguments: dict
    summary: str


@dataclass
class PermissionRequest(Event):
    """The loop is paused until the user allows or denies this tool call."""
    type: ClassVar[str] = "permission_request"
    id: str
    name: str
    arguments: dict
    summary: str
    risk: str


@dataclass
class ToolResult(Event):
    type: ClassVar[str] = "tool_result"
    id: str
    name: str
    ok: bool
    content: str                 # what the model sees (possibly truncated)
    error_code: str | None = None
    duration_s: float = 0.0


@dataclass
class TodoUpdate(Event):
    type: ClassVar[str] = "todos"
    todos: list[dict] = field(default_factory=list)


@dataclass
class Notice(Event):
    """Something the user should know that is not part of the answer (compaction, retries, guards)."""
    type: ClassVar[str] = "notice"
    level: str                   # "info" | "warn" | "error"
    text: str


@dataclass
class TurnEnd(Event):
    """The turn is over. `reason`: done | max_steps | loop | cancelled | error | blocked."""
    type: ClassVar[str] = "turn_end"
    reason: str
    steps: int
    usage: dict[str, Any] = field(default_factory=dict)
