"""The async job queue (jobs.py): priorities, idle-only scavenging, timing, retries, dead letters, persistence."""

import asyncio
import json
import time

import pytest

from llm_gateway.backends import BackendError
from llm_gateway.config import load_config
from llm_gateway.jobs import JobConflict, JobNotFound, JobQueue, classify
from llm_gateway.manager import ModelBusy

OK = {"choices": [{"message": {"content": "done"}}]}


def make(tmp_path, runner, idle=lambda: True, **kw):
    kw.setdefault("poll_s", 0.01)
    kw.setdefault("backoff_base_s", 0.02)
    return JobQueue(tmp_path / "jobs.db", runner, idle, **kw)


async def until(cond, timeout=3.0):
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < end, "condition not reached in time"
        await asyncio.sleep(0.01)


async def finished(q, job_id, statuses=("succeeded", "failed", "suspended")):
    await until(lambda: q.get(job_id)["status"] in statuses)
    return q.get(job_id)


async def test_job_goes_queued_running_succeeded_and_keeps_its_result(tmp_path):
    gpu = asyncio.Event()
    seen = []

    async def runner(job, mark_running):
        await gpu.wait()                  # waiting for the GPU or a model load: still "queued", no timeout
        mark_running()
        seen.append(job["source"])
        return OK

    q = make(tmp_path, runner)
    q.start()
    j = q.submit({"messages": []}, model="m", priority="high")
    assert j["status"] == "queued" and j["job_id"] == j["id"]
    await asyncio.sleep(0.1)
    assert q.get(j["id"])["status"] == "queued"            # started, but has not been given the GPU yet
    gpu.set()
    done = await finished(q, j["id"])
    assert done["status"] == "succeeded" and done["result"] == OK and seen == ["chat"]
    assert q.stats()["today_requests"] == 1 and q.stats()["today_success"] == 1
    await q.close()


async def test_one_job_at_a_time_in_priority_order(tmp_path):
    order, gate = [], asyncio.Event()

    async def runner(job, mark_running):
        mark_running()
        order.append(job["body"]["n"])
        if job["body"]["n"] == "first":
            await gate.wait()
        return OK

    q = make(tmp_path, runner)
    q.start()
    first = q.submit({"n": "first"}, model="m", priority="normal")
    await until(lambda: q.get(first["id"])["status"] == "running")
    ids = [q.submit({"n": n}, model="m", priority=p)["id"]
           for n, p in (("idle", "idle_only"), ("normal", "normal"), ("high", "high"))]
    assert q.stats()["queue"] == {"high": 1, "normal": 1, "idle_only": 1} and q.stats()["running"]["id"] == first["id"]
    s = q.stats()                                                   # the waiting list is in the order they run
    assert [w["id"] for w in s["waiting"]] == ids[::-1] and [w["priority"] for w in s["waiting"]] == ["high", "normal", "idle_only"]
    assert time.localtime(s["since"])[3:6] == (0, 0, 0) and time.strftime("%Y-%m-%d", time.localtime(s["since"])) == s["date"]
    gate.set()
    for i in ids:
        await finished(q, i)
    assert order == ["first", "high", "normal", "idle"]
    await q.close()


async def test_idle_only_runs_only_when_the_gpu_is_idle(tmp_path):
    idle = {"v": False}

    async def runner(job, mark_running):
        mark_running()
        return OK

    q = make(tmp_path, runner, idle=lambda: idle["v"])
    q.start()
    j = q.submit({}, model="m", priority="idle_only")
    await asyncio.sleep(0.15)
    assert q.get(j["id"])["status"] == "queued"
    idle["v"] = True
    assert (await finished(q, j["id"]))["status"] == "succeeded"
    await q.close()


async def test_idle_only_gives_way_even_to_a_normal_job_that_is_waiting_out_a_backoff(tmp_path):
    ran = []

    async def runner(job, mark_running):
        mark_running()
        ran.append(job["priority"])
        if job["priority"] == "normal" and ran.count("normal") == 1:
            raise BackendError("llama-server error 503: busy")      # retryable: backs off
        return OK

    q = make(tmp_path, runner, backoff_base_s=0.3)
    q.start()
    n = q.submit({}, model="m", priority="normal")
    await until(lambda: q.get(n["id"])["attempts"] == 1)
    i = q.submit({}, model="m", priority="idle_only")
    await asyncio.sleep(0.1)
    assert q.get(i["id"])["status"] == "queued"             # the normal job is only waiting, but it is ahead
    await finished(q, n["id"])
    await finished(q, i["id"])
    assert ran == ["normal", "normal", "idle_only"]
    await q.close()


async def test_retryable_errors_back_off_then_suspend_and_resume_runs_again(tmp_path):
    calls = {"n": 0, "fail": True}

    async def runner(job, mark_running):
        mark_running()
        calls["n"] += 1
        if calls["fail"]:
            raise asyncio.TimeoutError
        return OK

    q = make(tmp_path, runner, max_attempts=3)
    q.start()
    j = q.submit({}, model="m")
    dead = await finished(q, j["id"])
    assert dead["status"] == "suspended" and dead["attempts"] == 3 and calls["n"] == 3
    assert dead["error_kind"] == "timeout"
    s = q.stats()
    assert s["suspended"] == 1 and s["today_failed"] == 1 and s["today_success"] == 0
    calls["fail"] = False
    assert q.resume(j["id"])["status"] == "queued"
    assert (await finished(q, j["id"]))["status"] == "succeeded"
    assert q.stats()["suspended"] == 0 and q.stats()["today_success"] == 1
    assert q.stats()["today_requests"] == 2              # the resume is a request too: one per outcome
    await q.close()


async def test_fatal_errors_fail_at_once_without_retrying(tmp_path):
    calls = []

    async def runner(job, mark_running):
        mark_running()
        calls.append(1)
        raise BackendError("Ollama error 400: bad request")

    q = make(tmp_path, runner)
    q.start()
    j = q.submit({}, model="m")
    done = await finished(q, j["id"])
    assert done["status"] == "failed" and done["attempts"] == 1 and len(calls) == 1
    assert done["error_kind"] == "http_400"
    await q.close()


def test_which_errors_are_worth_retrying():
    assert classify(asyncio.TimeoutError()) == (True, "timeout")
    assert classify(BackendError("Ollama error 429: slow down")) == (True, "http_429")
    assert classify(BackendError("llama-server error 503: x")) == (True, "http_503")
    assert classify(BackendError("llama-server error 401: no")) == (False, "http_401")
    assert classify(BackendError("Ollama failed to load qwen: out of memory"))[0] is True
    assert classify(ModelBusy("m", "queue_timeout"))[0] is True
    assert classify(KeyError("unknown"))[0] is False


async def test_same_dedup_key_is_the_same_job_and_a_discarded_one_stays_dropped(tmp_path):
    async def runner(job, mark_running):
        raise BackendError("Ollama error 400: bad")

    q = make(tmp_path, runner)
    a = q.submit({}, model="m", dedup_key="L1:s1:0_10")
    b = q.submit({}, model="m", dedup_key="L1:s1:0_10")
    assert a["id"] == b["id"] and q.stats()["today_requests"] == 1
    q.start()
    await finished(q, a["id"])                                  # fatal: failed
    again = q.submit({}, model="m", dedup_key="L1:s1:0_10")     # the caller asks again: runs again
    assert again["id"] == a["id"] and again["status"] == "queued" and again["attempts"] == 0
    await finished(q, a["id"])
    assert q.discard(a["id"])["status"] == "discarded"
    assert q.submit({}, model="m", dedup_key="L1:s1:0_10")["status"] == "discarded"   # the user dropped it
    await q.close()


async def test_resume_and_discard_only_where_they_make_sense(tmp_path):
    gate = asyncio.Event()

    async def runner(job, mark_running):
        mark_running()
        await gate.wait()
        return OK

    q = make(tmp_path, runner)
    q.start()
    j = q.submit({}, model="m")
    with pytest.raises(JobConflict):
        q.resume(j["id"])                                       # not suspended
    await until(lambda: q.get(j["id"])["status"] == "running")
    with pytest.raises(JobConflict):
        q.discard(j["id"])                                      # running
    with pytest.raises(JobNotFound):
        q.get("nope")
    gate.set()
    await finished(q, j["id"])
    await q.close()


async def test_state_survives_a_restart_and_an_interrupted_job_is_queued_again(tmp_path):
    async def stuck(job, mark_running):
        mark_running()
        await asyncio.sleep(60)

    q = make(tmp_path, stuck, max_attempts=1)
    q.start()
    running = q.submit({"x": 1}, model="m", dedup_key="k-running")
    await until(lambda: q.get(running["id"])["status"] == "running")
    await q.close()                                             # the gateway stops mid-job

    async def fail(job, mark_running):
        mark_running()
        raise asyncio.TimeoutError

    q2 = make(tmp_path, fail, max_attempts=1)
    assert q2.get(running["id"])["status"] == "queued"          # it did not finish: run it again
    q2.start()
    dead = await finished(q2, running["id"])
    assert dead["status"] == "suspended"
    await q2.close()

    q3 = make(tmp_path, fail)                                   # restart: the dead letter and today's numbers stay
    assert q3.get(running["id"])["status"] == "suspended"
    assert q3.stats()["suspended"] == 1 and q3.stats()["today_requests"] == 1 and q3.stats()["today_failed"] == 1
    assert [j["id"] for j in q3.list(["suspended"])] == [running["id"]]
    assert [j["id"] for j in q3.list(key_prefix="k-")] == [running["id"]]
    assert q3.list(key_prefix="other") == []
    await q3.close()


async def test_jobs_are_never_lost_when_the_runner_is_cancelled(tmp_path):
    async def slow(job, mark_running):
        mark_running()
        await asyncio.sleep(60)

    q = make(tmp_path, slow)
    q.start()
    j = q.submit({}, model="m")
    await until(lambda: q.get(j["id"])["status"] == "running")
    await q.close()
    assert make(tmp_path, slow).get(j["id"])["status"] == "queued"


# ── the HTTP routes (the worker is not started: no GPU, no Ollama) ────────


def app_client(tmp_path):
    from fastapi.testclient import TestClient
    from llm_gateway.server import create_app

    cfg = {"llamacpp": {"server_binary": "llama-server"}, "profiles": [
        {"id": "a", "name": "A", "backend": "ollama", "model": "a:1b", "context": 4096}]}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return TestClient(create_app(load_config(tmp_path / "config.json")))


def test_http_submit_poll_list_stats_resume_discard(tmp_path):
    c = app_client(tmp_path)
    r = c.post("/v1/async/chat/completions", json={
        "model": "a", "messages": [{"role": "user", "content": "hi"}], "priority": "idle_only",
        "dedup_key": "L1:s:0_5", "stream": True})
    assert r.status_code == 202
    job = r.json()
    assert job["status"] == "queued" and job["priority"] == "idle_only" and job["dedup_key"] == "L1:s:0_5"
    assert c.post("/v1/async/chat/completions", json={"model": "a", "messages": [], "dedup_key": "L1:s:0_5"}
                  ).json()["id"] == job["id"]                    # idempotent
    assert c.get(f"/v1/jobs/{job['id']}").json()["status"] == "queued"
    assert [j["id"] for j in c.get("/v1/jobs", params={"key_prefix": "L1:s:"}).json()["jobs"]] == [job["id"]]
    s = c.get("/v1/jobs/stats").json()
    assert s["today_requests"] == 1 and s["queue"]["idle_only"] == 1 and s["suspended"] == 0
    assert "jobs" in c.get("/api/status").json()
    assert c.post(f"/v1/jobs/{job['id']}/resume").status_code == 409     # only for suspended or failed
    assert c.delete(f"/v1/jobs/{job['id']}").json()["status"] == "discarded"


def test_http_errors(tmp_path):
    c = app_client(tmp_path)
    assert c.get("/v1/jobs/unknown").status_code == 404                 # the caller must treat this as a lost job
    assert c.post("/v1/jobs/unknown/resume").status_code == 404
    assert c.post("/v1/async/chat/completions", json={"model": "nope", "messages": []}).status_code == 404
    assert c.post("/v1/async/chat/completions",
                  json={"model": "a", "messages": [], "priority": "urgent"}).status_code == 400
