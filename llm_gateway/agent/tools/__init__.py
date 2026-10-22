"""Built-in tools. `builtin_tools()` is the default tool set of a new agent."""

from __future__ import annotations

from .base import Risk, Tool, ToolContext, ToolRegistry
from .files import EditFile, Glob, Grep, ReadFile, WriteFile
from .git import Git
from .shell import Bash
from .skill import SkillTool
from .todo import TodoWrite
from .web import WebFetch

__all__ = ["Risk", "Tool", "ToolContext", "ToolRegistry", "builtin_tools"]


def builtin_tools(shell: list[str]) -> list[Tool]:
    return [ReadFile(), Glob(), Grep(), EditFile(), WriteFile(), Bash(shell), Git(), WebFetch(), TodoWrite(),
            SkillTool()]
