"""The agent loop.

Everything in the agent diagram (planner, reasoning, reflection, tools, memory) comes down to a
loop with two states:

            ┌──────────────────────────────────────────────────────────────┐
            v                                                              │
    user ─> CALL LLM ── answer has tool calls? ──yes──> RUN TOOLS ── results appended to messages
               │
               no
               v
             done

- Planning and reasoning happen inside the LLM call (plus the todo_write tool).
- Reflection is simply the next LLM call, which sees the tool results, including errors.
- Memory is handled right before each call (context.py) and after each step (session checkpoint).

`Agent.run()` is an async generator: it yields events (events.py) and the caller renders them.
Around the core sit the guards: permission checks, hooks, error classification and retries,
a step budget, a loop detector and a circuit breaker.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from . import errors
from .context import ContextManager, build_system_prompt
from .events import Event, Notice, PermissionRequest, TodoUpdate, ToolCall, ToolResult, TurnEnd
from .hooks import HookOutcome, Hooks
from .llm import Completion, LLMClient, LLMError
from .permissions import Approver, Decision, PermissionPolicy
from .session import Session, SessionStore
from .skills import Skill
from .tools.base import ToolContext, ToolOutput, ToolRegistry, parse_arguments, validate
from .tools.shell import shell_name


@dataclass
class AgentSettings:
    max_steps: int = 40           # model calls per user turn (the step budget)
    think: bool = False
    max_output_chars: int = 12_000


@dataclass
class Agent:
    session: Session
    llm: LLMClient
    tools: ToolRegistry
    context: ContextManager
    approver: Approver
    store: SessionStore | None = None
    hooks: Hooks | None = None
    skills: dict[str, Skill] = field(default_factory=dict)
    memory: list[tuple[str, str]] = field(default_factory=list)
    shell: list[str] = field(default_factory=lambda: ["bash", "-c"])
    scratch: Path | None = None        # extra fast temporary folder (pods: /tmp)
    settings: AgentSettings = field(default_factory=AgentSettings)

    def __post_init__(self):
        self.policy = PermissionPolicy(self.session.mode, set(self.session.always_allowed))
        self.breaker = errors.CircuitBreaker()
        self.loops = errors.LoopDetector()
        self.ctx = ToolContext(workspace=Path(self.session.workspace).resolve(), shell=self.shell,
                               max_output_chars=self.settings.max_output_chars,
                               read_files={Path(p) for p in self.session.read_files},
                               todos=self.session.todos, skills=self.skills, scratch=self.scratch)
        self.hook_context = ""     # SessionStart / UserPromptSubmit output, given to the model
        self._started = False

    # ── public API ────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        self.policy = PermissionPolicy(mode, self.policy.always_allowed)
        self.session.mode = mode
        self.checkpoint()

    def checkpoint(self) -> None:
        """Save the session (called after every step)."""
        self.session.always_allowed = sorted(self.policy.always_allowed)
        self.session.read_files = sorted(str(p) for p in self.ctx.read_files)
        if self.store:
            self.store.save(self.session)

    async def run(self, prompt: str) -> AsyncIterator[Event]:
        """One user turn: loop Call LLM <-> Run tools until the model answers without tool calls."""
        if not self._started and self.hooks:
            self._started = True
            outcome = await self.hooks.run("SessionStart", {"session_id": self.session.id})
            async for e in self._hook_notices(outcome):
                yield e
            self.hook_context = outcome.message

        if self.hooks:
            outcome = await self.hooks.run("UserPromptSubmit", {"session_id": self.session.id, "prompt": prompt})
            async for e in self._hook_notices(outcome):
                yield e
            if outcome.blocked:
                yield Notice("error", f"Prompt blocked by a hook: {outcome.message}")
                yield TurnEnd("blocked", 0)
                return
            if outcome.message:
                prompt = f"{prompt}\n\n<hook_context>\n{outcome.message}\n</hook_context>"

        if not self.session.title:
            self.session.title = prompt.strip().splitlines()[0][:80] if prompt.strip() else "(empty)"
        self.session.messages.append({"role": "user", "content": prompt})
        self.loops.reset()
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        steps = 0
        nudges = 0

        while True:
            if steps >= self.settings.max_steps:
                yield Notice("warn", f"Stopped after {steps} steps (step budget). Send a message to continue.")
                yield TurnEnd("max_steps", steps, usage)
                return
            steps += 1

            # ── state 1: CALL LLM ─────────────────────────────────────────
            schemas = self.tools.schemas()
            system = self.system_prompt()
            for text in await self.context.fit(self.session, system, schemas, self.llm):
                yield Notice("info", text)
            system = self.system_prompt()     # the summary may have changed
            completion: Completion | None = None
            try:
                async for item in self.llm.stream([{"role": "system", "content": system}, *self.session.messages],
                                                  schemas, think=self.settings.think):
                    if isinstance(item, Completion):
                        completion = item
                    else:
                        yield item
            except LLMError as e:
                yield Notice("error", f"Model call failed: {e}")
                self.checkpoint()
                yield TurnEnd("error", steps, usage)
                return
            assert completion is not None
            for k in usage:
                usage[k] += completion.usage.get(k, 0) or 0
            self.session.messages.append(completion.message())

            if not completion.tool_calls:
                if completion.finish_reason == "length":
                    yield Notice("warn", "The answer was cut off (output token limit).")
                elif errors.announces_unfinished_work(completion.content) and nudges < 2:
                    nudges += 1    # "Let me fix it now." with no tool call: tell the model to do it
                    yield Notice("info", "The model announced an action without doing it; asking it to continue.")
                    self.session.messages.append({"role": "user", "content": "Go ahead: do that now with the tools. "
                                                  "Do not stop until the task is complete."})
                    self.checkpoint()
                    continue
                stop = await self._stop_hook()
                if stop:            # a Stop hook (e.g. "tests must pass") sends the model back to work
                    yield Notice("info", "A Stop hook asked the agent to continue.")
                    self.session.messages.append({"role": "user", "content": stop})
                    self.checkpoint()
                    continue
                if not completion.content.strip():
                    yield Notice("warn", "The model returned an empty answer.")
                self.checkpoint()
                yield TurnEnd("done", steps, usage)
                return

            # ── state 2: RUN TOOLS ────────────────────────────────────────
            stop_turn = False
            for call in completion.tool_calls:
                result = None
                async for event in self._run_tool_call(call):
                    if isinstance(event, ToolResult):
                        result = event
                    yield event
                assert result is not None
                self.session.messages.append({"role": "tool", "tool_call_id": call["id"],
                                              "name": call["function"]["name"], "content": result.content})
                if result.error_code == "loop":
                    stop_turn = True
            self.checkpoint()
            if stop_turn:
                yield Notice("error", "Stopped: the model kept repeating the same tool call.")
                yield TurnEnd("loop", steps, usage)
                return

    # ── one tool call: validate → guard → permission → hook → execute → verify ─────

    async def _run_tool_call(self, call: dict) -> AsyncIterator[Event]:
        name = call["function"]["name"]
        call_id = call["id"]
        tool = self.tools.get(name)
        started = time.monotonic()

        def fail(err: errors.ToolError) -> ToolResult:
            self.breaker.record(name, False)
            schema = tool.parameters if tool else None
            return ToolResult(call_id, name, False, errors.feedback(err, schema), err.code,
                              round(time.monotonic() - started, 2))

        # 1. Is it a real tool, with arguments that parse and match the schema? (model errors)
        try:
            if tool is None:
                raise errors.ToolError("unknown_tool", f"there is no tool named '{name}'. "
                                       f"Available: {', '.join(t.name for t in self.tools)}")
            args = parse_arguments(call["function"].get("arguments"))
        except errors.ToolError as e:
            yield ToolCall(call_id, name, {}, f"{name}(?)")
            yield fail(e)
            return
        yield ToolCall(call_id, name, args, tool.summary(args))
        problems = validate(tool.parameters, args)
        if problems:
            yield fail(errors.ToolError("invalid_args", "; ".join(problems)))
            return

        # 2. Guards: repeated identical calls, and tools that keep failing
        repeats = self.loops.check(name, args)
        if repeats >= 2:
            yield ToolResult(call_id, name, False, "Error (loop): same call repeated; stopping.", "loop")
            return
        if self.breaker.is_open(name):
            yield fail(errors.ToolError("circuit_open", f"{name} failed {self.breaker.threshold} times in a row "
                                        "and is switched off for this session. Use a different approach."))
            return

        # 3. Permission (policy first, then ask the human if needed)
        risk = tool.risk_for(args)
        decision = self.policy.decide(name, risk)
        if decision is Decision.DENY:
            yield fail(errors.ToolError("denied", f"{name} ({risk.value}) is not allowed in {self.policy.mode} mode."))
            return
        if decision is Decision.ASK:
            yield PermissionRequest(call_id, name, args, tool.summary(args), risk.value)
            answer = await self.approver.ask(call_id, name, args, tool.summary(args), risk)
            if not answer.allow:
                why = f" The user said: {answer.reason}" if answer.reason else ""
                yield ToolResult(call_id, name, False, f"The user denied this call.{why} Do not retry it unchanged; "
                                 "ask the user or choose another approach.", "denied")
                return
            if answer.always:
                self.policy.remember(name, risk)

        # 4. PreToolUse hook can veto
        if self.hooks:
            outcome = await self.hooks.run("PreToolUse", {"session_id": self.session.id, "tool_name": name,
                                                          "tool_input": args}, tool=name)
            async for e in self._hook_notices(outcome):
                yield e
            if outcome.blocked:
                yield fail(errors.ToolError("blocked_by_hook", outcome.message))
                return

        # 5. Execute; transient failures are retried with backoff inside retry_transient
        retries: list[str] = []
        try:
            content = await errors.retry_transient(
                lambda: tool.run(args, self.ctx),
                on_retry=lambda n, e, d: retries.append(f"{name}: {e.message}; retry {n} in {d:.1f}s"))
        except errors.ToolError as e:
            for r in retries:
                yield Notice("warn", r)
            yield fail(e)
            return
        except Exception as e:   # a bug in a tool must not kill the session; report it like any failure
            yield fail(errors.ToolError("tool_crashed", f"{type(e).__name__}: {e}"))
            return
        for r in retries:
            yield Notice("warn", r)
        self.breaker.record(name, True)     # a command that ran but exited non-zero is not a broken tool
        ok = True
        if isinstance(content, ToolOutput):
            ok, content = content.ok, content.text

        # 6. Post-tool checks: silent failures, then the PostToolUse hook
        content = self._check_silent_failure(name, content)
        if self.hooks:
            outcome = await self.hooks.run("PostToolUse", {"session_id": self.session.id, "tool_name": name,
                                                           "tool_input": args, "tool_result": content[:20_000]},
                                           tool=name)
            async for e in self._hook_notices(outcome):
                yield e
            if outcome.blocked:
                content += f"\n\n[PostToolUse hook]\n{outcome.message}"
        if name == "todo_write":
            yield TodoUpdate(list(self.ctx.todos))
        if repeats == 1:
            content += "\n\n[Note: you made this exact call twice in a row. Do something different next.]"
        yield ToolResult(call_id, name, ok, content, None, round(time.monotonic() - started, 2))

    # ── helpers ───────────────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return build_system_prompt(self.session, shell_name=shell_name(self.shell), skills=self.skills,
                                   memory=self.memory, extra_context=self.hook_context, scratch=self.scratch)

    @staticmethod
    def _check_silent_failure(name: str, content: str) -> str:
        """'Success' that is probably wrong: say so, so the model double-checks instead of
        concluding that something does not exist."""
        if name in ("grep", "glob") and content.startswith(("No matches", "No files match")):
            return content + "\n[Check the pattern and path before concluding that nothing exists.]"
        return content

    async def _stop_hook(self) -> str | None:
        if not self.hooks or not self.hooks.matching("Stop", None):
            return None
        outcome = await self.hooks.run("Stop", {"session_id": self.session.id})
        return f"[Stop hook] Not done yet:\n{outcome.message}" if outcome.blocked else None

    @staticmethod
    async def _hook_notices(outcome: HookOutcome) -> AsyncIterator[Event]:
        for err in outcome.errors:
            yield Notice("warn", err)


def to_json(event: Event) -> str:
    return json.dumps(event.to_dict(), ensure_ascii=False)

