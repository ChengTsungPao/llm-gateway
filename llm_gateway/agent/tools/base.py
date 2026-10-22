"""The Tool interface and the registry (Command pattern + Registry).

A tool is a small class with:
    name / description / parameters   what the model sees (an OpenAI "function" schema)
    risk                              read | write | exec; the permission system decides from it
    run(args, ctx)                    does the work and returns text for the model, or raises ToolError

The loop only knows this interface. Adding a tool means writing one class and registering it;
the loop, the permission system and both UIs need no change.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ToolError

if TYPE_CHECKING:
    from ..skills import Skill


class Risk(str, Enum):
    READ = "read"     # looks at things: files, git status, web pages
    WRITE = "write"   # changes files inside the workspace
    EXEC = "exec"     # runs arbitrary commands: can do anything the user can


@dataclass
class ToolContext:
    """What a tool may touch. Tools get everything through this object, never through globals,
    so a test can hand them a temporary directory."""
    workspace: Path
    shell: list[str]                                  # e.g. ["bash", "-lc"]; see tools/shell.py
    max_output_chars: int = 30_000
    read_files: set[Path] = field(default_factory=set)   # edit/write require a prior read
    todos: list[dict] = field(default_factory=list)
    skills: dict[str, Skill] = field(default_factory=dict)

    def resolve(self, path: str) -> Path:
        """Turn a model-supplied path into an absolute path inside the workspace, or refuse.
        The model can write `../../etc` or an absolute path; both are checked after resolving."""
        p = Path(path)
        p = (p if p.is_absolute() else self.workspace / p).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ToolError("outside_workspace", f"{path} is outside the workspace ({self.workspace}).")
        return p

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix() or "."
        except ValueError:
            return str(path)


class Tool(ABC):
    name: str
    description: str
    parameters: dict          # JSON Schema for the arguments
    risk: Risk = Risk.READ

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def risk_for(self, args: dict) -> Risk:
        """Risk of this particular call. Most tools have a fixed risk; git overrides this
        because `git status` is harmless and `git push` is not."""
        return self.risk

    def summary(self, args: dict) -> str:
        """One line shown in the UI and in permission prompts."""
        first = next(iter(args.values()), "")
        return f"{self.name}({str(first)[:80]})"

    @abstractmethod
    async def run(self, args: dict, ctx: ToolContext) -> str: ...


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(schema: dict, args: Any) -> list[str]:
    """A deliberately small JSON-Schema check (required, type, enum), enough to catch the usual
    mistakes of small models, and every problem is reported so the model can fix them in one go."""
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    problems = [f"missing required field '{k}'" for k in schema.get("required", []) if k not in args]
    props = schema.get("properties", {})
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            problems.append(f"unknown field '{key}' (allowed: {', '.join(props)})")
            continue
        expected = _TYPES.get(spec.get("type", ""))
        if expected and (not isinstance(value, expected) or (spec.get("type") != "boolean" and isinstance(value, bool))):
            problems.append(f"field '{key}' must be {spec['type']}, got {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            problems.append(f"field '{key}' must be one of {spec['enum']}")
    return problems


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self):
        return iter(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


def truncate(text: str, limit: int) -> str:
    """Keep the head and the tail of long output: errors are usually at the end, context at the start."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n\n... [{cut} characters cut to fit the context] ...\n\n{text[-tail:]}"


def parse_arguments(raw: str | dict | None) -> dict:
    """Tool-call arguments arrive as a JSON string. Small models sometimes send nothing, a dict,
    or broken JSON; broken JSON becomes a model_error the model can fix."""
    if isinstance(raw, dict):
        return raw
    if not raw or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ToolError("bad_json", f"arguments are not valid JSON ({e.msg} at position {e.pos}): {raw[:200]}") from e
    if not isinstance(value, dict):
        raise ToolError("bad_json", "arguments must be a JSON object")
    return value
