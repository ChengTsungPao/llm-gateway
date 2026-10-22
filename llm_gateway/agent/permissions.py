"""Permission modes and approval (human in the loop).

Two separate questions, two separate pieces (Strategy pattern):

  1. Policy: given the mode and a tool call's risk, is the call allowed, denied, or must we ask?
     Pure logic, the same for every UI.
  2. Approver: if we must ask, how? The CLI prints a prompt and reads y/n; the web UI sends a
     permission_request event and waits for a click. Both implement `Approver.ask`.

Modes, from safest to most trusting (same idea as Claude Code):

    mode           read   write (files)   exec (bash, git push, MCP)
    plan           allow  deny            deny      look around and propose, change nothing
    default        allow  ask             ask
    accept_edits   allow  allow           ask       file edits flow, commands still need a yes
    bypass         allow  allow           allow     no questions; only for trusted, throwaway setups

"Always allow" answers are remembered per tool for the session (for git, per risk level, so
allowing `git commit` does not silently allow `git push`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .tools.base import Risk

MODES = ("plan", "default", "accept_edits", "bypass")


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_TABLE = {
    "plan":         {Risk.READ: Decision.ALLOW, Risk.WRITE: Decision.DENY,  Risk.EXEC: Decision.DENY},
    "default":      {Risk.READ: Decision.ALLOW, Risk.WRITE: Decision.ASK,   Risk.EXEC: Decision.ASK},
    "accept_edits": {Risk.READ: Decision.ALLOW, Risk.WRITE: Decision.ALLOW, Risk.EXEC: Decision.ASK},
    "bypass":       {Risk.READ: Decision.ALLOW, Risk.WRITE: Decision.ALLOW, Risk.EXEC: Decision.ALLOW},
}


@dataclass
class PermissionPolicy:
    mode: str = "default"
    always_allowed: set[str] = field(default_factory=set)   # "tool:risk" keys

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    @staticmethod
    def key(tool: str, risk: Risk) -> str:
        return f"{tool}:{risk.value}"

    def decide(self, tool: str, risk: Risk) -> Decision:
        decision = _TABLE[self.mode][risk]
        if decision is Decision.ASK and self.key(tool, risk) in self.always_allowed:
            return Decision.ALLOW
        return decision

    def remember(self, tool: str, risk: Risk) -> None:
        self.always_allowed.add(self.key(tool, risk))


@dataclass
class Answer:
    allow: bool
    always: bool = False          # remember "allow" for this tool for the rest of the session
    reason: str = ""              # optional note from the user, passed to the model on denial


class Approver(Protocol):
    async def ask(self, request_id: str, tool: str, arguments: dict, summary: str, risk: Risk) -> Answer: ...


class DenyAll:
    """Approver for non-interactive runs (`llm-agent -p`): anything that needs a question is refused."""

    async def ask(self, request_id: str, tool: str, arguments: dict, summary: str, risk: Risk) -> Answer:
        return Answer(False, reason="non-interactive run; use --mode accept_edits or bypass to allow this")
