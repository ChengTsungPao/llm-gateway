"""Keeps one model on the GPU and decides who may use it next (the model scheduler).

Every model call takes a ticket and waits for its turn: `async with manager.use(profile, ...)`.
The GPU serves one generation at a time; the scheduler picks the next ticket by these rules:

    1. Compatible models are shared, not switched. A profile is compatible with the loaded one
       when it has the same weights and needs no more context (e.g. an 8K chat profile on the
       32K agent profile of the same model). No reload at all.
    2. Same model: chat goes first. A waiting chat request runs before the agent's next call
       (a generation already running is never interrupted).
    3. Different model: wait for the agent to finish. An agent task holds a lease on its model
       from its first call until the task ends, so nothing switches the model under it while
       it runs tools between calls. Requests for other models wait until no lease is left.
    4. Otherwise: chat, then agent, then other API clients, first come first served.

Safety valves: a lease is dropped after `lease_idle_s` without a model call (an agent waiting
for a permission that never comes must not block the GPU forever), and a ticket gives up after
`max_wait_s` in the queue.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .backends import Backend, BackendError
from .config import Config, Profile

PRIORITY = {"chat": 0, "agent": 1, "api": 2, "admin": 3}


class ModelBusy(RuntimeError):
    """The request could not get the model (it waited longer than the queue allows)."""

    def __init__(self, model_id: str | None, reason: str, message: str | None = None):
        self.model_id, self.reason = model_id, reason
        super().__init__(message or f"model '{model_id}' is busy ({reason})")


@dataclass
class Ticket:
    profile: Profile
    source: str                      # chat | agent | api | admin
    lease: str | None = None         # agent task id, if the call belongs to one
    request_id: str | None = None    # set by the caller so a UI can find its own place in the queue
    seq: int = 0
    created: float = field(default_factory=time.time)
    started: float | None = None     # when it got the GPU


@dataclass
class Lease:
    id: str
    profile_id: str                  # the loaded profile the lease holds
    last_used: float = field(default_factory=time.time)


def weights(p: Profile) -> tuple:
    """What has to be on the GPU for a profile; profiles with equal weights can share it."""
    return (p.backend, p.model or p.model_path, tuple(p.args))


class ModelManager:
    def __init__(self, cfg: Config, backend_factory: Callable[[Profile], Backend],
                 max_wait_s: float = 1800, lease_idle_s: float = 600):
        self.cfg = cfg
        self._factory = backend_factory
        self.max_wait_s = max_wait_s
        self.lease_idle_s = lease_idle_s
        self._cond = asyncio.Condition()
        self._waiting: list[Ticket] = []
        self._running: Ticket | None = None
        self._leases: dict[str, Lease] = {}
        self._seq = itertools.count()
        self.backend: Backend | None = None
        self.active_id: str | None = None
        self.status = "idle"          # idle | loading | ready | error
        self.loading_id: str | None = None
        self.error: str | None = None
        self.since = time.time()

    # ── state for the UI ──────────────────────────────────────────────────

    def snapshot(self) -> dict:
        now = time.time()
        order = self._order()
        return {
            "active": self.active_id,
            "status": self.status,
            "loading": self.loading_id,
            "error": self.error,
            "since": self.since,
            "inflight": int(self._running is not None),
            "busy": bool(self._running or self.loading_id or self._waiting),
            "running": ({**self._describe(self._running, now),
                         "running_s": round(now - (self._running.started or now))} if self._running else None),
            "queue": [{**self._describe(t, now), "position": i + 1} for i, t in enumerate(order)],
            "leases": [{"id": l.id, "model": l.profile_id, "idle_s": round(now - l.last_used)}
                       for l in self._live_leases(now)],
        }

    @staticmethod
    def _describe(t: Ticket, now: float) -> dict:
        return {"request_id": t.request_id, "source": t.source, "lease": t.lease, "model": t.profile.id,
                "waiting_s": round(now - t.created)}

    def queue_status(self, request_id: str) -> dict | None:
        """Where a waiting request stands, and what it is waiting for (None once it runs)."""
        now = time.time()
        order = self._order()
        for i, t in enumerate(order):
            if t.request_id == request_id:
                blocker = self._running or (order[0] if i else None)
                holders = [l.id for l in self._live_leases(now)] if not self._compatible(t.profile) else []
                return {"position": i + 1, "waiting_s": round(now - t.created),
                        "behind": self._describe(blocker, now) if blocker else None,
                        "leases": holders, "model": t.profile.id}
        return None

    # ── scheduling ────────────────────────────────────────────────────────

    def _live_leases(self, now: float) -> list[Lease]:
        return [l for l in self._leases.values() if now - l.last_used < self.lease_idle_s]

    def _compatible(self, p: Profile) -> bool:
        """Can p run on what is loaded, without a switch?"""
        loaded = self.cfg.profile(self.active_id) if self.active_id else None
        return (loaded is not None and self.status == "ready" and weights(loaded) == weights(p)
                and loaded.context >= p.context)

    def _order(self) -> list[Ticket]:
        return sorted(self._waiting, key=lambda t: (PRIORITY.get(t.source, 2), t.seq))

    def _eligible(self, t: Ticket, now: float) -> bool:
        if self._compatible(t.profile):
            return True
        live = self._live_leases(now)
        # another model: only when no task holds this one (the lease holder may switch its own)
        return all(l.id == t.lease for l in live)

    def _next(self) -> Ticket | None:
        if self._running is not None or self.loading_id:
            return None
        now = time.time()
        return next((t for t in self._order() if self._eligible(t, now)), None)

    # ── loading ───────────────────────────────────────────────────────────

    async def _load(self, profile: Profile) -> Backend:
        if self._compatible(profile):
            return self.backend
        self.status, self.loading_id, self.error, self.since = "loading", profile.id, None, time.time()
        if self.backend:
            await self.backend.unload()
        self.backend, self.active_id = None, None
        backend = self._factory(profile)
        try:
            await backend.load()
        except Exception as e:  # surface any load failure in the UI instead of crashing the server
            await backend.unload()
            self.status, self.loading_id, self.error, self.since = "error", None, str(e), time.time()
            raise BackendError(str(e)) from e
        self.backend, self.active_id = backend, profile.id
        self.status, self.loading_id, self.since = "ready", None, time.time()
        return backend

    # ── public API ────────────────────────────────────────────────────────

    @asynccontextmanager
    async def use(self, profile_id: str, source: str = "api", lease: str | None = None,
                  request_id: str | None = None) -> AsyncIterator[Backend]:
        """Wait for a turn on the GPU with profile_id (loading it if needed) and hold it until the
        request, including its stream, ends."""
        profile = self.cfg.profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        ticket = Ticket(profile, source, lease, request_id, next(self._seq))
        async with self._cond:
            self._waiting.append(ticket)
            granted = False
            try:
                deadline = time.monotonic() + self.max_wait_s
                while self._next() is not ticket:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ModelBusy(profile_id, "queue_timeout",
                                        f"waited {self.max_wait_s / 60:.0f} minutes for the GPU; still busy")
                    try:   # re-check now and then: leases expire without anyone notifying
                        await asyncio.wait_for(self._cond.wait(), timeout=min(left, 5))
                    except asyncio.TimeoutError:
                        pass
                granted = True
            finally:
                self._waiting.remove(ticket)
                if not granted:            # gave up or disconnected: the next one may go now
                    self._cond.notify_all()
            self._running = ticket
            ticket.started = time.time()
        try:
            backend = await self._load(profile)
            if lease:
                self._leases[lease] = Lease(lease, self.active_id)
            yield backend
        finally:
            async with self._cond:
                self._running = None
                if lease and lease in self._leases:
                    self._leases[lease].last_used = time.time()
                self._cond.notify_all()

    async def activate(self, profile_id: str) -> Backend:
        """Load a model ahead of time (the Models page's Load button). Queues like any request."""
        async with self.use(profile_id, source="admin") as backend:
            return backend

    def release_lease(self, lease: str) -> None:
        """An agent task ended: other models may be loaded again."""
        if self._leases.pop(lease, None) is not None:
            asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    async def unload(self, force: bool = False) -> None:
        """Release the GPU. Waits for running requests; unless forced (shutdown), it also waits
        until no agent task holds the model."""
        if force:
            async with self._cond:
                await self._cond.wait_for(lambda: self._running is None)
                self._running = Ticket(Profile("__unload__", "", "ollama", 1), "admin")
        else:
            async with self._cond:
                deadline = time.monotonic() + self.max_wait_s
                while self._running is not None or self.loading_id or self._live_leases(time.time()):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ModelBusy(self.active_id, "in_use", "the model is still in use by an agent task")
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=min(left, 5))
                    except asyncio.TimeoutError:
                        pass
                self._running = Ticket(Profile("__unload__", "", "ollama", 1), "admin")
        try:
            if self.backend:
                await self.backend.unload()
            self.backend, self.active_id = None, None
            self.status, self.loading_id, self.error, self.since = "idle", None, None, time.time()
        finally:
            async with self._cond:
                self._running = None
                self._cond.notify_all()
