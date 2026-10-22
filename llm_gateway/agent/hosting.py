"""Where web agents run (Strategy pattern). server.py's /api/agent routes only talk to an AgentHost:

    LocalHost   agents run inside the gateway process; tools act on this machine (agent.sandbox = "local")
    PodHost     every agent runs in its own Kubernetes pod (k8s.py; agent.sandbox = "k8s")

Both expose the same operations, so the web UI does not care which one is configured.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

from .permissions import MODES, Answer
from .runtime import AgentRuntime
from .service import AgentService, LiveSession


class HostError(Exception):
    """An error with the HTTP status the route should answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class AgentHost(Protocol):
    kind: str

    async def info(self) -> dict: ...
    async def list(self) -> list[dict]: ...
    async def create(self, model: str, mode: str, workspace: str | None) -> dict: ...
    async def get(self, agent_id: str) -> dict: ...
    async def delete(self, agent_id: str) -> None: ...
    async def update(self, agent_id: str, changes: dict) -> dict: ...
    async def start_turn(self, agent_id: str, prompt: str) -> None: ...
    def events(self, agent_id: str, start: int) -> AsyncIterator[bytes]: ...
    async def permission(self, agent_id: str, request_id: str, answer: Answer) -> None: ...
    async def cancel(self, agent_id: str) -> bool: ...
    async def close(self) -> None: ...


def check_mode(mode: str) -> None:
    if mode not in MODES:
        raise HostError(400, f"mode must be one of {MODES}")


def session_view(live: LiveSession) -> dict:
    s = live.agent.session
    return {**s.meta(), "messages": s.messages, "todos": s.todos, "summary": s.summary,
            "running": live.running, "pending_permissions": list(live.approver.pending)}


def sse(index: int, event: dict) -> bytes:
    return f"id: {index}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


class LocalHost:
    """Agents run in this process (the AgentService); their tools act on this machine."""
    kind = "local"

    def __init__(self, runtime: AgentRuntime, service: AgentService):
        self.runtime = runtime
        self.service = service

    def _live(self, agent_id: str) -> LiveSession:
        try:
            return self.service.get(agent_id)
        except KeyError:
            raise HostError(404, f"no agent {agent_id}") from None

    async def info(self) -> dict:
        roots = self.runtime.workspace_roots()
        return {"sandbox": self.kind, "default_workspace": str(roots[0]), "workspaces": [str(r) for r in roots]}

    async def list(self) -> list[dict]:
        return self.runtime.store.list()

    async def create(self, model: str, mode: str, workspace: str | None) -> dict:
        check_mode(mode)
        try:
            live = self.service.create(workspace or str(self.runtime.cfg.default_workspace), model, mode)
        except ValueError as e:
            raise HostError(400, str(e)) from e
        return session_view(live)

    async def get(self, agent_id: str) -> dict:
        return session_view(self._live(agent_id))

    async def delete(self, agent_id: str) -> None:
        live = self.service.live.pop(agent_id, None)
        if live and live.running:
            live.task.cancel()
        self.runtime.store.delete(agent_id)

    async def update(self, agent_id: str, changes: dict) -> dict:
        live = self._live(agent_id)
        try:
            if "title" in changes:
                live.agent.session.title = str(changes["title"]).strip()[:120] or live.agent.session.title
                live.agent.checkpoint()
            if "mode" in changes:
                check_mode(changes["mode"])
                live.agent.set_mode(changes["mode"])     # applies from the next tool call, even mid-turn
            if "workspace" in changes:
                live = self.service.set_workspace(agent_id, str(changes["workspace"]))
            if "model" in changes:
                live = self.service.set_model(agent_id, changes["model"])
        except ValueError as e:
            raise HostError(400, str(e)) from e
        except RuntimeError as e:
            raise HostError(409, str(e)) from e
        return session_view(live)

    async def start_turn(self, agent_id: str, prompt: str) -> None:
        self._live(agent_id)
        try:
            self.service.start_turn(agent_id, prompt)
        except RuntimeError as e:
            raise HostError(409, str(e)) from e

    async def events(self, agent_id: str, start: int) -> AsyncIterator[bytes]:
        self._live(agent_id)
        async for i, event in self.service.stream(agent_id, start):
            yield sse(i, event)

    async def permission(self, agent_id: str, request_id: str, answer: Answer) -> None:
        if not self._live(agent_id).approver.answer(request_id, answer):
            raise HostError(404, "no pending permission request with that id")

    async def cancel(self, agent_id: str) -> bool:
        self._live(agent_id)
        return self.service.cancel(agent_id)

    async def close(self) -> None:
        for live in self.service.live.values():
            if live.running:
                live.task.cancel()
