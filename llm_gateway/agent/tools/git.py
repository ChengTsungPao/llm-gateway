"""The git tool. Its risk depends on the arguments: `git status` is read-only, `git commit`
changes the repository, and `git push` / `reset --hard` reach outside or destroy work.

A separate git tool (instead of "use bash") lets the permission system auto-allow the
read-only commands while still asking before anything that changes history.
"""

from __future__ import annotations

import shlex

from ..errors import ToolError
from .base import Risk, Tool, ToolContext, truncate
from .shell import run_shell

READ_ONLY = {"status", "diff", "log", "show", "blame", "ls-files", "rev-parse", "describe", "shortlog", "grep"}
# Local changes that are easy to undo (or only touch the working tree in an expected way)
LOCAL_WRITE = {"add", "commit", "restore", "switch", "checkout", "stash", "mv", "rm", "tag", "merge", "rebase",
               "cherry-pick", "revert", "init", "fetch", "pull", "branch", "remote", "config"}
# Anything that leaves the machine or throws work away, whatever the permission mode
DANGEROUS_FLAGS = {"--force", "-f", "--hard", "-D", "--force-with-lease"}


class Git(Tool):
    name = "git"
    description = ("Run a git command in the workspace, e.g. `status`, `diff --stat`, `log --oneline -10`, "
                   "`add -A`, `commit -m \"msg\"`. Pass the arguments after `git`. Read-only commands run "
                   "without asking; commits, pushes and history changes need the user's permission.")
    parameters = {"type": "object", "properties": {
        "args": {"type": "string", "description": "Arguments after `git`, e.g. 'log --oneline -5'."},
    }, "required": ["args"]}

    @staticmethod
    def _split(args: dict) -> list[str]:
        try:
            parts = shlex.split(args.get("args", ""))
        except ValueError as e:
            raise ToolError("invalid_args", f"cannot parse git arguments: {e}") from e
        if parts and parts[0] == "git":   # models often include it anyway
            parts = parts[1:]
        if not parts:
            raise ToolError("invalid_args", "no git subcommand given")
        return parts

    def risk_for(self, args: dict) -> Risk:
        try:
            parts = self._split(args)
        except ToolError:
            return Risk.READ  # it will fail validation in run() anyway
        sub = parts[0]
        if (sub == "branch" and len(parts) == 1) or (sub == "remote" and parts[1:] in ([], ["-v"])):
            return Risk.READ  # listing branches / remotes
        if sub in READ_ONLY:
            return Risk.READ
        if sub == "push" or sub == "clean" or DANGEROUS_FLAGS & set(parts):
            return Risk.EXEC
        if sub in LOCAL_WRITE:
            return Risk.WRITE
        return Risk.EXEC

    def summary(self, args: dict) -> str:
        return f"git {args.get('args', '')}"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        parts = self._split(args)
        # Run through the shell with each argument quoted, so commit messages with spaces survive.
        # --no-pager and GIT_TERMINAL_PROMPT=0 make sure git never waits for a pager or a password.
        command = "git --no-pager " + " ".join(shlex.quote(p) for p in parts)
        if "powershell" in ctx.shell[0].lower():
            command = "git --no-pager " + " ".join("'" + p.replace("'", "''") + "'" for p in parts)
        code, out = await run_shell(ctx.shell, command, ctx.workspace, 120,
                                    env={"GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true"})
        out = truncate(out.rstrip(), ctx.max_output_chars) or "(no output)"
        if code != 0:
            raise ToolError("git_failed", f"git exited with code {code}:\n{out}")
        return out
