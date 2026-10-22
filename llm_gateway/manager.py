"""Keeps exactly one model loaded. Switching to another model is refused while the current
one is in use (requests in flight) or while a model is loading."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from .backends import Backend, BackendError
from .config import Config, Profile


class ModelBusy(RuntimeError):
    def __init__(self, model_id: str | None, reason: str):
        self.model_id, self.reason = model_id, reason
        what = "is loading" if reason == "loading" else "is in use"
        super().__init__(f"model '{model_id}' {what}; switching models is not allowed until it is idle")


class ModelManager:
    def __init__(self, cfg: Config, backend_factory: Callable[[Profile], Backend]):
        self.cfg = cfg
        self._factory = backend_factory
        # Held while loading/unloading AND while registering a request, so a request
        # can never be counted against a model that is about to be unloaded.
        self._switch_lock = asyncio.Lock()
        self._idle = asyncio.Condition()
        self._inflight = 0
        self.backend: Backend | None = None
        self.active_id: str | None = None
        self.status = "idle"          # idle | loading | ready | error
        self.loading_id: str | None = None
        self.error: str | None = None
        self.since = time.time()

    def snapshot(self) -> dict:
        return {
            "active": self.active_id,
            "status": self.status,
            "loading": self.loading_id,
            "error": self.error,
            "since": self.since,
            "inflight": self._inflight,
            "busy": bool(self._inflight or self.loading_id),
        }

    async def _wait_idle(self) -> None:
        async with self._idle:
            await self._idle.wait_for(lambda: self._inflight == 0)

    async def _unload_current(self) -> None:
        if self.backend:
            await self.backend.unload()
        self.backend, self.active_id = None, None

    async def _activate_locked(self, profile_id: str) -> Backend:
        profile = self.cfg.profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        if self.active_id == profile_id and self.status == "ready" and self.backend:
            return self.backend
        self._check_switch_allowed(profile_id)
        self.status, self.loading_id, self.error, self.since = "loading", profile_id, None, time.time()
        await self._unload_current()
        backend = self._factory(profile)
        try:
            await backend.load()
        except Exception as e:  # surface any load failure in the UI instead of crashing the server
            await backend.unload()
            self.status, self.loading_id, self.error, self.since = "error", None, str(e), time.time()
            raise BackendError(str(e)) from e
        self.backend, self.active_id = backend, profile_id
        self.status, self.loading_id, self.since = "ready", None, time.time()
        return backend

    def _check_switch_allowed(self, target: str | None) -> None:
        """No switching while a model is busy: refuse instead of queueing behind it."""
        if self.loading_id and self.loading_id != target:
            raise ModelBusy(self.loading_id, "loading")
        if self._inflight and self.active_id != target:
            raise ModelBusy(self.active_id, "in_use")

    def check_can_use(self, profile_id: str) -> None:
        """Raise ModelBusy now if a request for profile_id would need a refused switch."""
        if self.active_id != profile_id or self.status != "ready":
            self._check_switch_allowed(profile_id)

    async def activate(self, profile_id: str) -> Backend:
        self._check_switch_allowed(profile_id)  # fail fast instead of waiting for the lock
        async with self._switch_lock:
            return await self._activate_locked(profile_id)

    async def unload(self, force: bool = False) -> None:
        if not force:
            self._check_switch_allowed(None)
        async with self._switch_lock:
            if force:
                await self._wait_idle()
            else:
                self._check_switch_allowed(None)
            await self._unload_current()
            self.status, self.loading_id, self.error, self.since = "idle", None, None, time.time()

    @asynccontextmanager
    async def use(self, profile_id: str) -> AsyncIterator[Backend]:
        """Make profile_id the active model and hold it until the request (and its stream) ends.
        Requests for the model already in use run concurrently; a request for a different
        model is refused (ModelBusy) while the current one is in use or loading."""
        if self.active_id != profile_id or self.status != "ready":
            self._check_switch_allowed(profile_id)
        async with self._switch_lock:
            backend = await self._activate_locked(profile_id)
            async with self._idle:
                self._inflight += 1
        try:
            yield backend
        finally:
            async with self._idle:
                self._inflight -= 1
                self._idle.notify_all()
