"""Hooks: your own shell commands at fixed points of the loop (deterministic, not up to the model).

Configured in config.json under agent.hooks, Claude Code style:

    "hooks": {
      "PreToolUse":  [{"matcher": "bash|write_file", "command": "python guard.py"}],
      "PostToolUse": [{"matcher": "edit_file|write_file", "command": "ruff check --quiet ."}],
      "Stop":        [{"command": "pytest -q -x"}]
    }

The command gets a JSON description of the event on stdin. What its exit code means:

    event              exit 0 (stdout ...)            exit 2 = block (output goes to ...)
    SessionStart       added to the model's context   -
    UserPromptSubmit   added to the model's context   the prompt is rejected; output shown to the user
    PreToolUse         -                              the call is denied; output tells the model why
    PostToolUse        -                              output appended to the tool result for the model
    Stop               -                              the model must continue; output says why

Any other exit code is reported to the user and otherwise ignored. Hooks only come from the
gateway's own config file, never from the workspace, so opening a repository cannot make you run
its commands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ToolError
from .tools.shell import run_shell

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")


@dataclass
class HookOutcome:
    blocked: bool = False
    message: str = ""          # for the model (block reason) or extra context (exit 0)
    errors: list[str] = field(default_factory=list)   # for the user


class Hooks:
    def __init__(self, config: dict, shell: list[str], workspace: Path):
        unknown = set(config) - set(EVENTS)
        if unknown:
            raise ValueError(f"unknown hook events {sorted(unknown)}; valid: {EVENTS}")
        self.config = config
        self.shell = shell
        self.workspace = workspace

    def matching(self, event: str, tool: str | None) -> list[dict]:
        return [h for h in self.config.get(event, [])
                if tool is None or re.fullmatch(h.get("matcher", ".*") or ".*", tool)]

    async def run(self, event: str, payload: dict, tool: str | None = None) -> HookOutcome:
        outcome = HookOutcome()
        messages = []
        for hook in self.matching(event, tool):
            data = json.dumps({"event": event, "workspace": str(self.workspace), **payload}, ensure_ascii=False)
            try:
                code, out = await run_shell(self.shell, hook["command"], self.workspace,
                                            hook.get("timeout", 60), stdin=data)
            except ToolError as e:
                outcome.errors.append(f"{event} hook `{hook['command']}`: {e.message}")
                continue
            out = out.strip()
            if code == 2:
                outcome.blocked = True
                messages.append(out or f"blocked by hook `{hook['command']}`")
                break                     # first blocking hook wins
            if code != 0:
                outcome.errors.append(f"{event} hook `{hook['command']}` exited {code}: {out[:300]}")
            elif out:
                messages.append(out)
        outcome.message = "\n".join(messages)
        return outcome
