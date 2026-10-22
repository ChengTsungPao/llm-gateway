"""The bash tool, and the shell runner it shares with hooks.

On Windows "bash" means Git Bash when it is installed (most coding agents assume a POSIX shell,
and models write POSIX commands); otherwise PowerShell. The tool description tells the model
which shell it actually gets.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..errors import ToolError
from .base import Risk, Tool, ToolContext, ToolOutput, truncate


def detect_shell() -> list[str]:
    if sys.platform != "win32":
        return ["bash", "-lc"] if shutil.which("bash") else ["sh", "-c"]
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if Path(candidate).exists():
            return [candidate, "-c"]
    git = shutil.which("git")  # a Git install elsewhere: <git>\cmd\git.exe -> <git>\bin\bash.exe
    if git and (bash := Path(git).parent.parent / "bin" / "bash.exe").exists():
        return [str(bash), "-c"]
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]


def shell_name(shell: list[str]) -> str:
    return "PowerShell" if "powershell" in shell[0].lower() else "bash"


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the shell and everything it started (a timed-out `npm test` leaves children behind)."""
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


async def run_shell(shell: list[str], command: str, cwd: Path, timeout_s: float,
                    stdin: str | None = None, env: dict | None = None) -> tuple[int, str]:
    """Run a command, return (exit code, combined stdout+stderr). Raises ToolError on timeout."""
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
    proc = await asyncio.create_subprocess_exec(
        *shell, command, cwd=cwd, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **(env or {})}, **kwargs)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin.encode() if stdin is not None else None), timeout_s)
    except asyncio.TimeoutError:
        _kill_tree(proc)
        await proc.wait()
        # side_effect: the command ran for a while and may have changed things before it was killed
        raise ToolError("timeout", f"command did not finish within {timeout_s:.0f}s and was stopped. If it needs "
                        "longer, run it again with a larger timeout. A stopped git command can leave a "
                        "*.lock file in .git that must be removed first.", side_effect=True) from None
    except asyncio.CancelledError:
        _kill_tree(proc)
        raise
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


class Bash(Tool):
    name = "bash"
    risk = Risk.EXEC
    parameters = {"type": "object", "properties": {
        "command": {"type": "string", "description": "The command to run."},
        "timeout": {"type": "integer", "description": "Seconds before the command is stopped. Default 120, max 1800. "
                                                  "Give long jobs (git clone, npm install, builds) 600 or more."},
    }, "required": ["command"]}

    def __init__(self, shell: list[str]):
        self.description = (
            f"Run a command in {shell_name(shell)} in the workspace directory, e.g. tests, builds, package "
            "managers. Returns the output and exit code. Each call starts a fresh shell (no state is kept). "
            "Do not use it to read, search or edit files: use read_file, grep, glob, edit_file. "
            "Never run commands that wait for input. Downloads and installs can take minutes: set timeout.")

    def summary(self, args: dict) -> str:
        return f"$ {args.get('command', '')}"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        timeout = min(max(args.get("timeout") or 120, 1), 1800)
        # GIT_TERMINAL_PROMPT=0: a git command that wants a password fails at once instead of hanging
        code, out = await run_shell(ctx.shell, args["command"], ctx.workspace, timeout, env={"GIT_TERMINAL_PROMPT": "0"})
        out = truncate(out.rstrip(), ctx.max_output_chars) or "(no output)"
        return ToolOutput(f"{out}\n[exit code {code}]", ok=code == 0)
