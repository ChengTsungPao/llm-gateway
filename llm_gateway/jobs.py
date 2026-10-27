"""Async jobs: a persistent queue for background LLM work (memory extraction and the like).

A client submits a chat request and gets a job id at once (`POST /v1/async/chat/completions`), then
polls `GET /v1/jobs/{id}`. The queue runs ONE job at a time, through the same scheduler as every other
request (manager.py), so a background job never switches the model under a chat or an agent.

    priority   runs as          when
    high       a chat request   as soon as it is first in line
    normal     an API request   after live chats and agents
    idle_only  an API request   only while nothing else is queued or running (no chat, agent or job)

Timing: a job waiting in the queue, or waiting for the GPU or for its model to load, is `queued` and has
no timeout. The execution timeout starts when the model is ready and generation starts (`running`).

Failures: a retryable error (timeout, 429, 5xx, model busy, connection) is retried by the gateway with an
exponential backoff; after `max_attempts` the job is `suspended` (a dead letter). A fatal error (400, 401,
403, 404, 422) is `failed` at once. A suspended or failed job waits for `resume` (run again) or `discard`.
A caller that submits the same `dedup_key` again gets the existing job back instead of a second one.

State lives in SQLite (data/jobs.db), so a restart loses neither the queue, the dead letters nor today's
numbers; a job that was running when the gateway stopped is queued again.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from .backends import BackendError
from .manager import ModelBusy

PRIORITIES = ("high", "normal", "idle_only")
RANK = {p: i for i, p in enumerate(PRIORITIES)}
SOURCE = {"high": "chat", "normal": "api", "idle_only": "api"}   # how the scheduler sees the job
FATAL_STATUS = {400, 401, 403, 404, 422}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, dedup_key TEXT UNIQUE, priority TEXT NOT NULL, model TEXT NOT NULL,
    body TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL, timeout_s REAL NOT NULL, next_run_at REAL NOT NULL,
    error TEXT, error_kind TEXT, result TEXT, created REAL NOT NULL, started REAL, finished REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, priority, created);
CREATE TABLE IF NOT EXISTS daily (day TEXT PRIMARY KEY, requests INTEGER NOT NULL DEFAULT 0,
                                  success INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0);
"""

Runner = Callable[[dict, Callable[[], None]], Awaitable[dict]]


class JobNotFound(KeyError):
    pass


class JobConflict(RuntimeError):
    """The job is in a state that does not allow this action."""


def classify(exc: BaseException) -> tuple[bool, str]:
    """(retryable, kind) for an error raised while a job ran."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True, "timeout"
    if isinstance(exc, ModelBusy):
        return True, "model_busy"
    if isinstance(exc, httpx.HTTPError):
        return True, "connection"
    if isinstance(exc, BackendError):
        m = re.search(r"error (\d{3})", str(exc))
        status = int(m.group(1)) if m else None
        if status in FATAL_STATUS:
            return False, f"http_{status}"
        return True, f"http_{status}" if status else "backend"
    if isinstance(exc, KeyError):          # unknown model: retrying cannot help
        return False, "bad_request"
    return True, "error"


class JobQueue:
    def __init__(self, path: Path, runner: Runner, is_idle: Callable[[], bool], *, max_attempts: int = 5,
                 backoff_base_s: float = 30.0, backoff_cap_s: float = 900.0, timeout_s: float = 900.0,
                 poll_s: float = 2.0, keep_days: float = 7.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.executescript(SCHEMA)
        self.runner, self.is_idle = runner, is_idle
        self.max_attempts, self.backoff_base_s, self.backoff_cap_s = max_attempts, backoff_base_s, backoff_cap_s
        self.timeout_s, self.poll_s, self.keep_days = timeout_s, poll_s, keep_days
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.current: str | None = None
        self._recover()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def _recover(self) -> None:
        """Jobs that were running when the gateway stopped did not finish: queue them again."""
        with self._lock:
            self._db.execute("UPDATE jobs SET status='queued', next_run_at=? WHERE status='running'", (time.time(),))
            self._db.execute("DELETE FROM jobs WHERE status IN ('succeeded','discarded') AND finished < ?",
                             (time.time() - self.keep_days * 86400,))

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        with self._lock:
            self._db.close()

    # ── API used by the HTTP routes ───────────────────────────────────────

    def submit(self, body: dict, *, model: str, priority: str = "normal", dedup_key: str | None = None,
               timeout_s: float | None = None, max_attempts: int | None = None) -> dict:
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        now = time.time()
        with self._lock:
            row = self._row_by_key(dedup_key) if dedup_key else None
            if row is not None:
                if row["status"] not in ("failed", "discarded"):
                    return self._view(row)          # the same work again: hand back the job that exists
                # a failed (fatal) job submitted again is a retry the caller asks for; a discarded one is
                # the user's decision to drop that work, and stays dropped
                if row["status"] == "discarded":
                    return self._view(row)
                self._db.execute("UPDATE jobs SET status='queued', attempts=0, error=NULL, error_kind=NULL, "
                                 "next_run_at=?, body=?, priority=?, model=?, finished=NULL WHERE id=?",
                                 (now, json.dumps(body), priority, model, row["id"]))
                self._bump("requests")
                job_id = row["id"]
            else:
                job_id = uuid.uuid4().hex[:16]
                self._db.execute(
                    "INSERT INTO jobs (id, dedup_key, priority, model, body, status, max_attempts, timeout_s, "
                    "next_run_at, created) VALUES (?,?,?,?,?,'queued',?,?,?,?)",
                    (job_id, dedup_key, priority, model, json.dumps(body), max_attempts or self.max_attempts,
                     timeout_s or self.timeout_s, now, now))
                self._bump("requests")
            view = self._view(self._row(job_id))
        self._wake.set()
        return view

    def get(self, job_id: str, *, with_result: bool = True) -> dict:
        with self._lock:
            return self._view(self._row(job_id), with_result=with_result)

    def list(self, statuses: list[str] | None = None, priority: str | None = None, key_prefix: str | None = None,
             limit: int = 100) -> list[dict]:
        sql, args = "SELECT * FROM jobs WHERE 1=1", []
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            args += statuses
        if priority:
            sql += " AND priority=?"
            args.append(priority)
        if key_prefix:
            sql += " AND dedup_key LIKE ? ESCAPE '\\'"
            args.append(key_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        sql += " ORDER BY created DESC LIMIT ?"
        args.append(max(1, min(limit, 500)))
        with self._lock:
            return [self._view(r, with_result=False) for r in self._db.execute(sql, args)]

    def resume(self, job_id: str) -> dict:
        with self._lock:
            row = self._row(job_id)
            if row["status"] not in ("suspended", "failed"):
                raise JobConflict(f"job is {row['status']}; only a suspended or failed job can be resumed")
            self._db.execute("UPDATE jobs SET status='queued', attempts=0, error=NULL, error_kind=NULL, "
                             "next_run_at=?, finished=NULL WHERE id=?", (time.time(), job_id))
            view = self._view(self._row(job_id))
        self._wake.set()
        return view

    def discard(self, job_id: str) -> dict:
        """Drop a job for good. Its dedup_key stays as a tombstone, so the caller that submitted it learns
        that the work was discarded instead of submitting it again."""
        with self._lock:
            row = self._row(job_id)
            if row["status"] == "running":
                raise JobConflict("the job is running; wait for it to finish")
            self._db.execute("UPDATE jobs SET status='discarded', body='{}', result=NULL, finished=? WHERE id=?",
                             (time.time(), job_id))
            return self._view(self._row(job_id))

    def stats(self) -> dict:
        day = time.strftime("%Y-%m-%d")
        with self._lock:
            d = self._db.execute("SELECT requests, success, failed FROM daily WHERE day=?", (day,)).fetchone()
            counts = {(r["status"], r["priority"]): r["n"] for r in self._db.execute(
                "SELECT status, priority, COUNT(*) AS n FROM jobs GROUP BY status, priority")}
            running = self._db.execute("SELECT * FROM jobs WHERE status='running' LIMIT 1").fetchone()
            waiting_retry = self._db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND attempts>0"
                                             ).fetchone()[0]
            running_view = self._view(running, with_result=False) if running else None
            queued = list(self._db.execute("SELECT id, model, priority, attempts, created, next_run_at FROM jobs "
                                           "WHERE status='queued'"))
        by_status = lambda s: sum(n for (st, _), n in counts.items() if st == s)  # noqa: E731
        now = time.time()
        queued.sort(key=lambda r: (RANK[r["priority"]], r["created"]))      # the order _pick runs them in
        waiting = [{"id": r["id"], "model": r["model"], "priority": r["priority"], "attempts": r["attempts"],
                    "waiting_s": max(0, round(now - r["next_run_at"])),     # since it became runnable (resume too)
                    "retry_in_s": max(0, round(r["next_run_at"] - now)) if r["attempts"] else 0}
                   for r in queued[:20]]
        return {
            "date": day,
            "since": time.mktime(time.strptime(day, "%Y-%m-%d")),   # the counters cover local midnight → now
            "waiting": waiting,
            "today_requests": d["requests"] if d else 0,
            "today_success": d["success"] if d else 0,
            "today_failed": d["failed"] if d else 0,
            "queue": {p: counts.get(("queued", p), 0) for p in PRIORITIES},
            "waiting_retry": waiting_retry,
            "running": running_view,
            "suspended": by_status("suspended"),
            "failed": by_status("failed"),
        }

    # ── the worker ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            row = self._pick()
            if row is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._sleep_s())
                except asyncio.TimeoutError:
                    pass
                continue
            await self._run(row)

    def _sleep_s(self) -> float:
        with self._lock:
            nxt = self._db.execute("SELECT MIN(next_run_at) FROM jobs WHERE status='queued'").fetchone()[0]
        return self.poll_s if nxt is None else max(0.05, min(self.poll_s, nxt - time.time()))

    def _pick(self) -> sqlite3.Row | None:
        now = time.time()
        with self._lock:
            rows = list(self._db.execute("SELECT * FROM jobs WHERE status='queued' AND next_run_at<=?", (now,)))
            foreground_waiting = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='queued' AND priority IN ('high','normal')").fetchone()[0]
        rows.sort(key=lambda r: (RANK[r["priority"]], r["created"]))
        for row in rows:
            if row["priority"] != "idle_only":
                return row
            # background scavenging gives way to everything: no high/normal job anywhere (even one that is
            # only waiting out a backoff), and the GPU is idle
            if foreground_waiting == 0 and self.is_idle():
                return row
        return None

    async def _run(self, row: sqlite3.Row) -> None:
        job_id = row["id"]
        self.current = job_id

        def mark_running() -> None:      # the scheduler granted the GPU: from here the timeout counts
            with self._lock:
                self._db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), job_id))

        job = {"id": job_id, "model": row["model"], "priority": row["priority"], "timeout_s": row["timeout_s"],
               "source": SOURCE[row["priority"]], "body": json.loads(row["body"])}
        try:
            result = await self.runner(job, mark_running)
        except asyncio.CancelledError:
            with self._lock:
                self._db.execute("UPDATE jobs SET status='queued', started=NULL WHERE id=?", (job_id,))
            raise
        except Exception as e:  # noqa: BLE001 - every failure of a job is classified, none may kill the worker
            self._failed(job_id, e)
        else:
            with self._lock:
                self._db.execute("UPDATE jobs SET status='succeeded', result=?, error=NULL, error_kind=NULL, "
                                 "finished=? WHERE id=?", (json.dumps(result), time.time(), job_id))
                self._bump("success")
        finally:
            self.current = None

    def _failed(self, job_id: str, exc: BaseException) -> None:
        retryable, kind = classify(exc)
        message = (str(exc) or type(exc).__name__)[:500]
        with self._lock:
            row = self._row(job_id)
            attempts = row["attempts"] + 1
            if retryable and attempts < row["max_attempts"]:
                delay = min(self.backoff_cap_s, self.backoff_base_s * 2 ** (attempts - 1))
                delay *= 1 + random.random() * 0.2          # a little jitter: many jobs must not retry together
                self._db.execute("UPDATE jobs SET status='queued', attempts=?, error=?, error_kind=?, next_run_at=?, "
                                 "started=NULL WHERE id=?", (attempts, message, kind, time.time() + delay, job_id))
            else:
                status = "suspended" if retryable else "failed"
                self._db.execute("UPDATE jobs SET status=?, attempts=?, error=?, error_kind=?, finished=? WHERE id=?",
                                 (status, attempts, message, kind, time.time(), job_id))
                self._bump("failed")

    # ── helpers ───────────────────────────────────────────────────────────

    def _bump(self, column: str) -> None:      # caller holds the lock
        day = time.strftime("%Y-%m-%d")
        self._db.execute("INSERT OR IGNORE INTO daily (day) VALUES (?)", (day,))
        self._db.execute(f"UPDATE daily SET {column}={column}+1 WHERE day=?", (day,))

    def _row(self, job_id: str) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return row

    def _row_by_key(self, key: str) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM jobs WHERE dedup_key=?", (key,)).fetchone()

    def _view(self, row: sqlite3.Row, with_result: bool = True) -> dict:
        view = {k: row[k] for k in ("id", "dedup_key", "priority", "model", "status", "attempts", "max_attempts",
                                    "timeout_s", "next_run_at", "error", "error_kind", "created", "started",
                                    "finished")}
        view["job_id"] = row["id"]
        if row["status"] == "queued":
            ahead = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='queued' AND (CASE priority WHEN 'high' THEN 0 WHEN 'normal' "
                "THEN 1 ELSE 2 END, created) < (?, ?)", (RANK[row["priority"]], row["created"])).fetchone()[0]
            view["queue_position"] = ahead + 1
        if with_result and row["status"] == "succeeded" and row["result"]:
            view["result"] = json.loads(row["result"])
        return view
