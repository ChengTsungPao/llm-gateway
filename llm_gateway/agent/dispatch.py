"""Layer 2 of scheduling: how many agent tasks run at the same time (agent.max_running).

Layer 1 (manager.py) decides who may use the GPU for each single model call. This layer sits
in front of the AgentHost and decides when an agent turn may start at all:

    send a task ──> waiting list ──> dispatcher (at most max_running turns at once) ──> host.start_turn

Running many agents at once is not faster: the GPU generates for one request at a time, so five
parallel agents each crawl, hold five pods' worth of memory, and keep chat waiting. A waiting task
costs nothing; its pod is only woken when the task starts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .hosting import AgentHost, HostError

log = logging.getLogger("uvicorn.error")


@dataclass
class Waiting:
    agent_id: str
    prompt: str
    title: str
    since: float = field(default_factory=time.time)


class Dispatcher:
    def __init__(self, host: AgentHost, max_running: int = 1, poll_s: float = 2):
        self.host = host
        self.max_running = max(1, max_running)
        self.poll_s = poll_s
        self.waiting: list[Waiting] = []
        self.running: dict[str, str] = {}         # agent id -> title
        self.titles: dict[str, str] = {}          # for status displays
        self.errors: dict[str, str] = {}          # a task that failed to start, until it is looked at
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    def position(self, agent_id: str) -> int | None:
        """1-based place in the waiting list, or None when not waiting."""
        return next((i + 1 for i, w in enumerate(self.waiting) if w.agent_id == agent_id), None)

    async def submit(self, agent_id: str, prompt: str) -> int:
        """Queue a turn; returns its waiting position (0 = it starts right away)."""
        view = await self.host.get(agent_id)            # 404 for an unknown agent
        if agent_id in self.running or view.get("running") or self.position(agent_id):
            raise HostError(409, "this task is already working on a turn")
        title = view.get("title") or (prompt.strip().splitlines()[0][:80] if prompt.strip() else agent_id)
        self.titles[agent_id] = title
        self.errors.pop(agent_id, None)
        self.waiting.append(Waiting(agent_id, prompt, title))
        self._wake.set()
        return 0 if len(self.running) < self.max_running and len(self.waiting) == 1 else len(self.waiting)

    async def cancel(self, agent_id: str) -> bool:
        pos = self.position(agent_id)
        if pos:
            self.waiting.pop(pos - 1)
            return True
        return await self.host.cancel(agent_id)

    def snapshot(self) -> dict:
        now = time.time()
        return {"max_running": self.max_running,
                "running": [{"agent": a, "title": t} for a, t in self.running.items()],
                "waiting": [{"agent": w.agent_id, "title": w.title, "position": i + 1,
                             "waiting_s": round(now - w.since)} for i, w in enumerate(self.waiting)]}

    async def _loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self.waiting and len(self.running) < self.max_running:
                w = self.waiting.pop(0)
                try:
                    await self.host.start_turn(w.agent_id, w.prompt)
                except HostError as e:
                    self.errors[w.agent_id] = e.message
                    log.warning("agent %s could not start: %s", w.agent_id, e.message)
                    continue
                self.running[w.agent_id] = w.title
                asyncio.create_task(self._watch(w.agent_id))

    async def _watch(self, agent_id: str) -> None:
        """Free the slot when the turn ends (asking the host works the same for local and pod agents)."""
        try:
            while True:
                await asyncio.sleep(self.poll_s)
                try:
                    if not (await self.host.get(agent_id)).get("running"):
                        return
                except HostError:
                    return            # deleted while running
        finally:
            self.running.pop(agent_id, None)
            self._wake.set()
