"""`llm-agent`: the terminal front end. It renders the loop's events and asks for permissions.

    llm-agent                          interactive session in the current folder
    llm-agent -p "fix the failing test" --mode accept_edits     one task, then exit
    llm-agent --resume <id>            continue a saved session
    llm-agent --list                   list saved sessions

The agent runs in this process; only model calls go to the gateway (which must be running).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

import httpx

from ..config import ConfigError, load_config
from .events import Notice, PermissionRequest, ReasoningDelta, TextDelta, TodoUpdate, ToolCall, ToolResult, TurnEnd
from .loop import Agent
from .permissions import MODES, Answer, DenyAll
from .runtime import AgentRuntime
from .tools.base import Risk

DIM, BOLD, RED, GREEN, YELLOW, CYAN, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
HELP = """Commands:
  /mode [plan|default|accept_edits|bypass]   show or change the permission mode
  /model [id]      show or change the model      /tools    list tools
  /skills          list skills                    /todos    show the task list
  /compact         summarize older messages now   /clear    start a new session
  /sessions        list saved sessions            /exit     quit
Ctrl+C stops the current turn; Ctrl+C at the prompt quits."""


class CLIApprover:
    """Asks in the terminal. Blocking input() is fine here: nothing else runs while we wait."""

    async def ask(self, request_id: str, tool: str, arguments: dict, summary: str, risk: Risk) -> Answer:
        details = ""
        if tool == "edit_file":
            details = _diff_preview(arguments)
        elif tool == "write_file":
            details = textwrap.indent(arguments.get("content", "")[:1500], "    ")
        if details:
            print(DIM + details + RESET)
        while True:
            try:
                reply = input(f"{YELLOW}Allow {risk.value}: {summary}?{RESET} [y]es / [a]lways / [n]o (or type a reason) > ")
            except (EOFError, KeyboardInterrupt):
                print()
                return Answer(False, reason="the user cancelled")
            reply = reply.strip()
            if reply.lower() in ("y", "yes"):
                return Answer(True)
            if reply.lower() in ("a", "always"):
                return Answer(True, always=True)
            if reply.lower() in ("n", "no", ""):
                return Answer(False)
            return Answer(False, reason=reply)


def _diff_preview(args: dict) -> str:
    old = [f"    - {line}" for line in args.get("old_string", "").splitlines()[:20]]
    new = [f"    + {line}" for line in args.get("new_string", "").splitlines()[:20]]
    return "\n".join(old + new)


class Renderer:
    """Turns events into terminal output (the CLI's Observer)."""

    def __init__(self, show_thinking: bool, out=sys.stdout, show_text: bool = True):
        self.show_thinking = show_thinking
        self.show_text = show_text      # off in -p mode, where only the final answer goes to stdout
        self.out = out
        self.in_text = False

    def print(self, *parts, end="\n"):
        print(*parts, end=end, file=self.out, flush=True)

    def _end_text(self):
        if self.in_text:
            self.print()
            self.in_text = False

    def __call__(self, event) -> None:
        if isinstance(event, TextDelta):
            if self.show_text:
                self.print(event.text, end="")
                self.in_text = True
        elif isinstance(event, ReasoningDelta):
            if self.show_thinking:
                self.print(DIM + event.text + RESET, end="")
                self.in_text = True
        elif isinstance(event, ToolCall):
            self._end_text()
            self.print(f"{CYAN}● {event.summary}{RESET}")
        elif isinstance(event, PermissionRequest):
            self._end_text()
        elif isinstance(event, ToolResult):
            lines = event.content.strip().splitlines() or [""]
            color = DIM if event.ok else RED
            shown = lines[:6] + ([f"... ({len(lines) - 6} more lines)"] if len(lines) > 6 else [])
            self.print(color + "\n".join(("  ⎿ " if i == 0 else "    ") + line[:200] for i, line in enumerate(shown)) + RESET)
        elif isinstance(event, TodoUpdate):
            for t in event.todos:
                mark = {"completed": GREEN + "✔", "in_progress": YELLOW + "▶"}.get(t["status"], DIM + "○")
                self.print(f"  {mark} {t['content']}{RESET}")
        elif isinstance(event, Notice):
            self._end_text()
            color = {"error": RED, "warn": YELLOW}.get(event.level, DIM)
            self.print(f"{color}! {event.text}{RESET}")
        elif isinstance(event, TurnEnd):
            self._end_text()
            used = event.usage.get("prompt_tokens", 0) + event.usage.get("completion_tokens", 0)
            self.print(f"{DIM}[{event.reason} · {event.steps} steps · {used} tokens]{RESET}")


def _models(gateway: str, api_key: str) -> dict[str, int]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(f"{gateway}/models", headers=headers, timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as e:
        sys.exit(f"Cannot reach the gateway at {gateway} ({e}). Start it with `uv run llm-gateway`.")
    return {m["id"]: m.get("context_length", 8192) for m in r.json()["data"]}


async def run_turn(agent: Agent, prompt: str, render: Renderer) -> str:
    final = ""
    async for event in agent.run(prompt):
        render(event)
        if isinstance(event, TextDelta):
            final += event.text
        elif isinstance(event, ToolCall):
            final = ""      # only the text after the last tool call is "the answer"
    return final


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-agent", description="Coding agent on local models via llm-gateway.")
    parser.add_argument("-p", "--print", dest="prompt", help="run one task non-interactively and print the answer")
    parser.add_argument("--model", help="profile id (must have \"tools\": true)")
    parser.add_argument("--workspace", default=".", help="folder the agent works in (default: current folder)")
    parser.add_argument("--mode", choices=MODES, help="permission mode")
    parser.add_argument("--resume", metavar="ID", help="continue a saved session")
    parser.add_argument("--list", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--config", type=Path, help="gateway config.json (default: ./config.json, then the repo's)")
    parser.add_argument("--thinking", action="store_true", help="show the model's reasoning")
    args = parser.parse_args()

    if sys.platform == "win32":
        os.system("")   # turns on ANSI colour handling in the Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    config_path = args.config or next((p for p in (Path("config.json"), Path(__file__).parents[2] / "config.json")
                                       if p.exists()), Path("config.json"))
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        sys.exit(f"config: {e}")
    gateway = f"http://127.0.0.1:{cfg.port}/v1"
    runtime = AgentRuntime(cfg.agent, gateway, cfg.api_key)

    if args.list:
        for s in runtime.store.list():
            print(f"{s['id']}  {s['model']:<20} {s['messages']:>4} msgs  {s['title']}")
        return

    contexts = _models(gateway, cfg.api_key)
    agent_models = [p.id for p in cfg.profiles if p.tools]
    if args.resume:
        try:
            session = runtime.store.load(args.resume)
        except KeyError:
            sys.exit(f"no saved session {args.resume} (see --list)")
    else:
        model = args.model or (agent_models[0] if agent_models else None)
        if not model or model not in contexts:
            listed = ", ".join(agent_models) or "none (mark profiles with \"tools\": true)"
            sys.exit(f"choose a model with --model; agent models: {listed}")
        workspace = Path(args.workspace).resolve()
        session = runtime.new_session(workspace, model, args.mode)
    if args.mode:
        session.mode = args.mode

    interactive = args.prompt is None
    approver = CLIApprover() if interactive else DenyAll()
    render = Renderer(args.thinking)

    with asyncio.Runner() as runner:
        runner.run(runtime.start())
        for err in runtime.startup_errors:
            print(f"{YELLOW}! {err}{RESET}")
        agent = runtime.agent(session, approver, contexts.get(session.model, 8192))
        try:
            if not interactive:
                # progress goes to stderr, the final answer to stdout, so `llm-agent -p ... > out.md` works
                final = runner.run(run_turn(agent, args.prompt, Renderer(False, out=sys.stderr, show_text=False)))
                print(final)
                return
            print(f"{BOLD}llm-agent{RESET} · {session.model} · {session.mode} mode · {session.workspace}")
            print(f"{DIM}session {session.id} · /help for commands{RESET}")
            while True:
                try:
                    prompt = input(f"\n{BOLD}›{RESET} ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not prompt:
                    continue
                if prompt.startswith("/"):
                    agent = _command(prompt, agent, runtime, approver, contexts, runner)
                    if agent is None:
                        break
                    continue
                try:
                    runner.run(run_turn(agent, prompt, render))
                except KeyboardInterrupt:
                    agent.checkpoint()
                    print(f"\n{YELLOW}! stopped{RESET}")
        finally:
            runner.run(runtime.close())


def _command(line: str, agent: Agent, runtime: AgentRuntime, approver, contexts: dict, runner) -> Agent | None:
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()
    s = agent.session
    if cmd in ("exit", "quit"):
        return None
    if cmd == "help":
        print(HELP)
    elif cmd == "mode":
        if arg:
            if arg not in MODES:
                print(f"modes: {', '.join(MODES)}")
            else:
                agent.set_mode(arg)
        print(f"mode: {s.mode}")
    elif cmd == "model":
        if arg:
            if arg not in contexts:
                print(f"unknown model; available: {', '.join(contexts)}")
            else:
                s.model = arg
                agent.checkpoint()
                agent = runtime.agent(s, approver, contexts[arg])
        print(f"model: {s.model} (context {contexts.get(s.model)})")
    elif cmd == "tools":
        for t in agent.tools:
            print(f"  {t.name:<28} {t.risk.value:<6} {t.description.splitlines()[0][:70]}")
    elif cmd == "skills":
        for sk in agent.skills.values():
            print(f"  {sk.name}: {sk.description}")
        if not agent.skills:
            print("  no skills (add .agents/skills/<name>/SKILL.md)")
    elif cmd == "todos":
        print(json.dumps(s.todos, ensure_ascii=False, indent=2) if s.todos else "  no task list")
    elif cmd == "compact":
        before = len(s.messages)
        trigger, agent.context.trigger = agent.context.trigger, 0.0     # force both stages this once
        notes = runner.run(agent.context.fit(s, agent.system_prompt(), agent.tools.schemas(), agent.llm))
        agent.context.trigger = trigger
        agent.checkpoint()
        print("\n".join(notes) or "nothing to compact", f"({before} -> {len(s.messages)} messages)")
    elif cmd == "clear":
        new = runtime.new_session(Path(s.workspace), s.model, s.mode)
        agent = runtime.agent(new, approver, contexts.get(s.model, 8192))
        print(f"new session {new.id}")
    elif cmd == "sessions":
        for m in runtime.store.list()[:15]:
            print(f"  {m['id']}  {m['messages']:>4} msgs  {m['title']}")
    else:
        print(f"unknown command /{cmd}; /help lists them")
    return agent


if __name__ == "__main__":
    main()
