"""Runs agents for the web UI.

A turn runs as a background task, not inside the HTTP request. Its events go into a per-session
log, and the browser streams that log (Server-Sent Events) starting from any index. So a phone
that locks its screen mid-turn just reconnects and continues from where it stopped, and the
agent keeps working meanwhile.

Permission questions travel the same way: the loop emits a permission_request event and awaits
a Future; POST .../permission resolves it (WebApprover).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .events import Notice, TurnEnd
from .loop import Agent
from .permissions import Answer, Risk
from .runtime import AgentRuntime
from .session import Session


class WebApprover:
    def __init__(self):
        self.pending: dict[str, asyncio.Future] = {}

    async def ask(self, request_id: str, tool: str, arguments: dict, summary: str, risk: Risk) -> Answer:
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            return await future
        finally:
            self.pending.pop(request_id, None)

    def answer(self, request_id: str, answer: Answer) -> bool:
        future = self.pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(answer)
        return True


@dataclass
class LiveSession:
    agent: Agent
    approver: WebApprover
    events: list[dict] = field(default_factory=list)    # events of the current/last turn
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class AgentService:
    def __init__(self, runtime: AgentRuntime, context_of: callable):
        self.runtime = runtime
        self.context_of = context_of          # model id -> context size in tokens
        self.live: dict[str, LiveSession] = {}

    def _live(self, session: Session) -> LiveSession:
        approver = WebApprover()
        live = LiveSession(self.runtime.agent(session, approver, self.context_of(session.model)), approver)
        self.live[session.id] = live
        return live

    def get(self, session_id: str) -> LiveSession:
        live = self.live.get(session_id)
        if live is None:
            live = self._live(self.runtime.store.load(session_id))   # KeyError if unknown
        return live

    def create(self, workspace: str, model: str, mode: str | None) -> LiveSession:
        session = self.runtime.new_session(self.runtime.allowed_workspace(workspace), model, mode)
        return self._live(session)

    def set_model(self, session_id: str, model: str) -> LiveSession:
        live = self.get(session_id)
        if live.running:
            raise RuntimeError("a turn is running")
        live.agent.session.model = model
        return self._live(live.agent.session)   # rebuild: the context budget depends on the model

    def set_workspace(self, session_id: str, workspace: str) -> LiveSession:
        """Move a session to another folder. The conversation is kept; the agent is rebuilt so tools,
        skills and AGENTS.md come from the new folder, and it must re-read files before editing."""
        live = self.get(session_id)
        if live.running:
            raise RuntimeError("a turn is running")
        session = live.agent.session
        session.workspace = str(self.runtime.allowed_workspace(workspace))
        session.read_files = []    # the system prompt states the new workspace on the next call
        self.runtime.store.save(session)
        return self._live(session)

    async def _publish(self, live: LiveSession, event: dict) -> None:
        async with live.changed:
            live.events.append(event)
            live.changed.notify_all()

    def start_turn(self, session_id: str, prompt: str) -> LiveSession:
        live = self.get(session_id)
        if live.running:
            raise RuntimeError("this session is already working on a turn")
        live.events = []

        async def turn():
            try:
                async for event in live.agent.run(prompt):
                    await self._publish(live, event.to_dict())
            except asyncio.CancelledError:
                live.agent.checkpoint()
                await self._publish(live, Notice("warn", "Stopped by the user.").to_dict())
                await self._publish(live, TurnEnd("cancelled", 0).to_dict())
            except Exception as e:   # never leave the browser waiting on a turn that died
                live.agent.checkpoint()
                await self._publish(live, Notice("error", f"Agent error: {type(e).__name__}: {e}").to_dict())
                await self._publish(live, TurnEnd("error", 0).to_dict())

        live.task = asyncio.create_task(turn())
        return live

    async def stream(self, session_id: str, start: int = 0) -> AsyncIterator[tuple[int, dict]]:
        """Yield (index, event) from `start` until the turn ends."""
        live = self.get(session_id)
        i = start
        while True:
            async with live.changed:
                while i >= len(live.events) and live.running:
                    await live.changed.wait()
                batch = live.events[i:]
            for event in batch:
                yield i, event
                i += 1
                if event["type"] == "turn_end":
                    return
            if not batch and not live.running:
                return

    def cancel(self, session_id: str) -> bool:
        live = self.get(session_id)
        if not live.running:
            return False
        live.task.cancel()
        return True
