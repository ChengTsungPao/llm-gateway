"""HTTP API: OpenAI-compatible /v1 routes, management /api routes, and the web UI."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psutil
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent.dispatch import Dispatcher
from .agent.hosting import AgentHost, HostError, LocalHost
from .agent.k8s import PodHost
from .agent.permissions import MODES, Answer
from .agent.runtime import AgentRuntime
from .agent.service import AgentService
from .backends import BackendError, evict_ollama, free_gpu, make_backend
from .compaction import Compactor
from .config import Config
from .jobs import PRIORITIES, JobConflict, JobNotFound, JobQueue
from .manager import ModelBusy, ModelManager
from .protocols import anthropic, responses
from .protocols.stream import events as stream_events
from .protocols.stream import prompt_size, sse, with_heartbeat

# uvicorn's own logger, so these lines appear in the gateway log next to the request lines
log = logging.getLogger("uvicorn.error")

STATIC = Path(__file__).parent / "static"


def _gpu_info() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, used, total, util = [x.strip() for x in out.split(",")]
        return {"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization": int(util)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _system_info() -> dict:
    # cpu_percent(None) measures since the previous call (no blocking); the first call returns 0
    mem = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(None), "cpu_count": psutil.cpu_count(),
            "memory_used_mb": mem.used // 2**20, "memory_total_mb": mem.total // 2**20}


def create_app(cfg: Config) -> FastAPI:
    client = httpx.AsyncClient()
    manager = ModelManager(cfg, lambda profile: make_backend(profile, cfg, client))
    compactor = Compactor()
    gpu_cache: dict = {"t": 0.0, "v": None, "sys": None}
    # The agent calls this same server's /v1 API, like any other client would.
    runtime = AgentRuntime(cfg.agent, f"http://127.0.0.1:{cfg.port}/v1", cfg.api_key)
    context_of = {p.id: p.context for p in cfg.profiles}
    if cfg.agent.sandbox == "k8s":
        k = cfg.agent.k8s
        host: AgentHost = PodHost(k, {
            "GATEWAY_URL": k.gateway_url or f"http://host.docker.internal:{cfg.port}/v1",
            "LLM_API_KEY": cfg.api_key,
            "MODEL_CONTEXTS": json.dumps(context_of),
            "AGENT_CONFIG": json.dumps({"default_mode": cfg.agent.default_mode, "max_steps": cfg.agent.max_steps,
                                        "hooks": cfg.agent.hooks, "mcp_servers": cfg.agent.mcp_servers}),
        }, cfg.agent.default_mode)
    else:
        host = LocalHost(runtime, AgentService(runtime, lambda model: context_of.get(model, 8192)))
    dispatcher = Dispatcher(host, cfg.agent.max_running)    # layer 2: how many agent tasks run at once

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if isinstance(host, PodHost):
            await host.start()          # namespace + kubectl proxy; retried on first use if Kubernetes is off
        else:
            await runtime.start()       # MCP servers (in k8s mode each pod starts its own)
        # Models left in Ollama by an earlier run hold GPU memory while the gateway shows
        # "No model loaded"; start from a clean GPU (pinned models stay).
        await evict_ollama(client, cfg, keep=None, required=False)
        dispatcher.start()
        jobs.start()
        yield
        await jobs.close()
        await dispatcher.close()
        await host.close()
        await runtime.close()
        await manager.unload(force=True)  # never leave a llama-server process behind
        await client.aclose()

    app = FastAPI(title="llm-gateway", lifespan=lifespan)

    def require_key(request: Request) -> None:
        if not cfg.api_key:
            return
        auth = request.headers.get("authorization", "")
        # Claude Code sends the key as x-api-key when ANTHROPIC_API_KEY is used
        if auth != f"Bearer {cfg.api_key}" and request.headers.get("x-api-key") != cfg.api_key:
            raise HTTPException(401, "missing or wrong API key (Authorization: Bearer <key>)")

    def caller(request: Request) -> dict:
        """Who is asking, for the scheduler (manager.py): the source (chat | agent | api), an agent
        task's lease, and a request id the UI uses to find its own place in the queue."""
        h = request.headers
        source = h.get("x-llm-gateway-source", "api")
        source = source if source in ("chat", "agent") else "api"
        return {"source": source, "lease": (h.get("x-llm-gateway-lease") or None) if source == "agent" else None,
                "request_id": h.get("x-llm-gateway-request-id") or uuid.uuid4().hex}

    def queue_beat(request_id: str):
        """Heartbeat for streams: while the request waits, say where it stands (an SSE comment
        that OpenAI clients ignore and the UI and agent read); afterwards just keep-alive."""
        def beat() -> bytes:
            status = manager.queue_status(request_id)
            return (f": llm-gateway-queue {json.dumps(status)}\n\n".encode() if status else b": keep-alive\n\n")
        return beat

    def profile_or_404(profile_id: str):
        p = cfg.profile(profile_id)
        if p is None:
            raise HTTPException(404, f"unknown model '{profile_id}'. GET /v1/models lists the available ids.")
        return p

    # ── management ────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/status", dependencies=[Depends(require_key)])
    async def status():
        if time.time() - gpu_cache["t"] > 3:
            gpu_cache["v"], gpu_cache["sys"], gpu_cache["t"] = (await asyncio.to_thread(_gpu_info), _system_info(),
                                                                time.time())
        snap = manager.snapshot()
        # name agent tasks in the GPU queue by their title (leases are "agent-<task id>")
        titles = {f"agent-{a}": t for a, t in dispatcher.titles.items()}
        for item in [snap["running"], *snap["queue"], *snap["leases"]]:
            if item:
                key = item.get("lease") or item.get("id")
                if key in titles:
                    item["title"] = titles[key]
        return {**snap, "tasks": dispatcher.snapshot(), "jobs": jobs.stats(), "gpu": gpu_cache["v"],
                "system": gpu_cache["sys"], "auth_required": bool(cfg.api_key)}

    @app.get("/api/profiles", dependencies=[Depends(require_key)])
    async def profiles():
        return {"profiles": [p.public() for p in cfg.profiles]}

    @app.post("/api/activate", dependencies=[Depends(require_key)])
    async def activate(payload: dict):
        profile = profile_or_404(payload.get("id", ""))
        try:
            await manager.activate(profile.id)
        except ModelBusy as e:
            raise HTTPException(409, str(e)) from e
        except BackendError as e:
            raise HTTPException(502, str(e)) from e
        return manager.snapshot()

    @app.post("/api/unload", dependencies=[Depends(require_key)])
    async def unload(force: bool = False):
        if force:                         # the UI's Release when busy or stuck: now, plus leftovers
            await manager.release_now()
            await free_gpu(client, cfg)
            return manager.snapshot()
        try:
            await manager.unload()        # waits for running requests and agent tasks
        except ModelBusy as e:
            raise HTTPException(409, str(e)) from e
        return manager.snapshot()

    @app.delete("/v1/leases/{lease_id}", dependencies=[Depends(require_key)])
    async def release_lease(lease_id: str):
        """An agent task ended: its model no longer has to stay loaded for it."""
        manager.release_lease(lease_id)
        return {"ok": True}

    # ── OpenAI-compatible ─────────────────────────────────────────────────

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    async def models():
        return {"object": "list", "data": [
            {"id": p.id, "object": "model", "owned_by": p.backend, "created": 0,
             "context_length": p.context, "active": p.id == manager.active_id}
            for p in cfg.profiles]}

    async def fit_context(backend, profile, body: dict) -> tuple[dict, dict | None]:
        # Clients that manage their own context (the agent) send auto_compact=false
        client_allows = body.pop("auto_compact", True)
        if not (cfg.auto_compact and client_allows):
            return body, None

        async def summarize(messages: list[dict], max_tokens: int) -> str:
            d = await backend.chat({"messages": messages, "max_tokens": max_tokens,
                                    "temperature": 0.2, "think": False})
            return d["choices"][0]["message"].get("content") or ""

        return await compactor.compact(body, profile.context, summarize)

    # ── async jobs (jobs.py): submit now, poll later, one at a time ───────

    async def run_job(job: dict, mark_running) -> dict:
        profile = cfg.profile(job["model"])
        if profile is None:
            raise KeyError(job["model"])
        # waiting for the scheduler and for the model to load is not part of the timeout; generating is
        async with manager.use(profile.id, source=job["source"], request_id=job["id"]) as backend:
            mark_running()
            body, _ = await fit_context(backend, profile, dict(job["body"], model=profile.id, stream=False))
            return await asyncio.wait_for(backend.chat(body), job["timeout_s"])

    def gpu_idle() -> bool:
        """Nothing is queued, loading or running, no agent task holds the model: background scavenging may go."""
        snap = manager.snapshot()
        return not snap["busy"] and not snap["leases"] and not dispatcher.snapshot().get("running")

    jobs = JobQueue(cfg.jobs.db, run_job, gpu_idle, max_attempts=cfg.jobs.max_attempts,
                    backoff_base_s=cfg.jobs.backoff_base_s, backoff_cap_s=cfg.jobs.backoff_cap_s,
                    timeout_s=cfg.jobs.timeout_s)

    @app.post("/v1/async/chat/completions", dependencies=[Depends(require_key)], status_code=202)
    async def async_chat(body: dict):
        """A chat request that does not wait: returns a job id; poll GET /v1/jobs/{id}. Extra fields:
        priority (high | normal | idle_only), dedup_key (the same key returns the same job), timeout_s,
        max_attempts."""
        model_id = body.pop("model", None) or manager.active_id
        if not model_id:
            raise HTTPException(400, "no 'model' given and no model is active")
        profile = profile_or_404(model_id)
        options = {k: body.pop(k, None) for k in ("priority", "dedup_key", "timeout_s", "max_attempts")}
        priority = options["priority"] or "normal"
        if priority not in PRIORITIES:
            raise HTTPException(400, f"priority must be one of {list(PRIORITIES)}")
        body["stream"] = False
        return jobs.submit(body, model=profile.id, priority=priority, dedup_key=options["dedup_key"] or None,
                           timeout_s=options["timeout_s"], max_attempts=options["max_attempts"])

    def job_or_404(job_id: str, with_result: bool = True) -> dict:
        try:
            return jobs.get(job_id, with_result=with_result)
        except JobNotFound:
            raise HTTPException(404, f"no job '{job_id}' (the gateway may have been reset)") from None

    @app.get("/v1/jobs/stats", dependencies=[Depends(require_key)])
    async def job_stats():
        return jobs.stats()

    @app.get("/v1/jobs", dependencies=[Depends(require_key)])
    async def job_list(status: str = "", priority: str = "", key_prefix: str = "", limit: int = 100):
        statuses = [s for s in status.split(",") if s]
        return {"jobs": jobs.list(statuses or None, priority or None, key_prefix or None, limit)}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    async def job_get(job_id: str):
        return job_or_404(job_id)

    @app.post("/v1/jobs/{job_id}/resume", dependencies=[Depends(require_key)])
    async def job_resume(job_id: str):
        job_or_404(job_id, with_result=False)
        try:
            return jobs.resume(job_id)
        except JobConflict as e:
            raise HTTPException(409, str(e)) from e

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    async def job_discard(job_id: str):
        job_or_404(job_id, with_result=False)
        try:
            return jobs.discard(job_id)
        except JobConflict as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
    async def chat_completions(body: dict, request: Request):
        model_id = body.get("model") or manager.active_id
        if not model_id:
            raise HTTPException(400, "no 'model' given and no model is active")
        profile = profile_or_404(model_id)
        who = caller(request)

        if not body.get("stream"):
            try:
                async with manager.use(profile.id, **who) as backend:
                    body, info = await fit_context(backend, profile, body)
                    d = await backend.chat(body)
            except ModelBusy as e:
                raise HTTPException(409, str(e)) from e
            except BackendError as e:
                raise HTTPException(502, str(e)) from e
            if info:
                d["llm_gateway"] = info
            return JSONResponse(d, headers={"X-LLM-Gateway-Compacted": str(info["compacted_messages"])} if info else None)

        # Streaming: hold the model for the whole stream. Load errors become an SSE error event.
        # A compaction notice is sent first as an SSE comment line, which OpenAI clients ignore.
        async def answer():
            nonlocal body
            async with manager.use(profile.id, **who) as backend:
                body, info = await fit_context(backend, profile, body)
                if info:
                    yield f": llm-gateway {json.dumps(info)}\n\n".encode()
                async for piece in backend.chat_stream(body):
                    yield piece

        async def gen():
            try:
                async for piece in with_heartbeat(answer(), queue_beat(who["request_id"]), every_s=2):
                    yield piece
            except (BackendError, ModelBusy) as e:
                err = {"message": str(e), "code": "model_busy" if isinstance(e, ModelBusy) else "backend_error"}
                yield f"data: {json.dumps({'error': err}, ensure_ascii=False)}\n\n".encode()
                yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── Anthropic Messages (Claude Code) and OpenAI Responses (Codex) ─────
    # Each request is translated to Chat Completions (protocols/), runs through the same model
    # switching, compaction and backends as /v1/chat/completions, and is translated back.

    def protocol_profile(model: str | None):
        """The named profile; otherwise the loaded model. Claude Code sends side requests (titles,
        summaries) for its own model names; those should not force a model switch."""
        return cfg.profile(model or "") or (cfg.profile(manager.active_id) if manager.active_id else None)

    async def run_chat(profile, chat: dict, who: dict) -> dict:
        async with manager.use(profile.id, **who) as backend:
            chat, _ = await fit_context(backend, profile, chat)
            return await backend.chat(chat)

    async def stream_chat(profile, chat: dict, who: dict) -> AsyncIterator[bytes]:
        async with manager.use(profile.id, **who) as backend:
            chat, _ = await fit_context(backend, profile, dict(chat, stream=True))
            async for piece in backend.chat_stream(chat):
                yield piece

    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    model_list = ", ".join(p.id for p in cfg.profiles if p.tools) or ", ".join(p.id for p in cfg.profiles)

    def anthropic_error(status: int, kind: str, message: str) -> JSONResponse:
        return JSONResponse(anthropic.error_body(kind, message), status_code=status)

    @app.post("/v1/messages", dependencies=[Depends(require_key)])
    async def messages(body: dict, request: Request):
        profile = protocol_profile(body.get("model"))
        if profile is None:
            return anthropic_error(404, "not_found_error", f"unknown model '{body.get('model')}'. "
                                   f"Set ANTHROPIC_MODEL to one of: {model_list}")
        chat = anthropic.to_chat(body, profile.id)
        size = prompt_size(chat)
        log.info("anthropic /v1/messages model=%s ~%d prompt tokens, %d messages, %d tools, stream=%s",
                 profile.id, size, len(chat["messages"]), len(chat.get("tools", [])), bool(body.get("stream")))
        who = caller(request)
        if not body.get("stream"):
            try:
                return anthropic.from_chat(await run_chat(profile, chat, who), profile.id, size)
            except ModelBusy as e:      # waited too long in the queue; Claude Code retries 529
                return anthropic_error(529, "overloaded_error", str(e))
            except BackendError as e:
                return anthropic_error(500, "api_error", str(e))

        async def gen():
            yield anthropic.message_start(profile.id, size)       # at once, even while the model loads
            try:
                frames = anthropic.stream(stream_events(stream_chat(profile, chat, who)), size)
                async for frame in with_heartbeat(frames, anthropic.PING):
                    yield frame
            except (BackendError, ModelBusy) as e:
                kind = "overloaded_error" if isinstance(e, ModelBusy) else "api_error"
                yield sse("error", anthropic.error_body(kind, str(e)))

        return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers)

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(require_key)])
    async def count_tokens(body: dict):
        return {"input_tokens": prompt_size(anthropic.to_chat(body, ""))}

    @app.post("/v1/responses", dependencies=[Depends(require_key)])
    async def responses_api(body: dict, request: Request):
        if body.get("previous_response_id"):
            raise HTTPException(400, "previous_response_id is not supported: send the full input (store=false)")
        profile = protocol_profile(body.get("model"))
        if profile is None:
            raise HTTPException(404, f"unknown model '{body.get('model')}'. Use one of: {model_list}")
        chat = responses.to_chat(body, profile.id)
        kinds = responses.tool_kinds(body)
        size = prompt_size(chat)
        log.info("responses /v1/responses model=%s ~%d prompt tokens, %d items, %d tools, stream=%s",
                 profile.id, size, len(body.get("input") or []), len(chat.get("tools", [])), bool(body.get("stream")))
        who = caller(request)
        if not body.get("stream"):
            try:
                return responses.from_chat(await run_chat(profile, chat, who), profile.id, kinds, size)
            except ModelBusy as e:
                raise HTTPException(409, str(e)) from e
            except BackendError as e:
                raise HTTPException(502, str(e)) from e

        writer = responses.StreamWriter(profile.id, kinds, size)

        async def gen():
            for frame in writer.start():
                yield frame
            try:
                frames = writer.stream(stream_events(stream_chat(profile, chat, who)))
                async for frame in with_heartbeat(frames, queue_beat(who["request_id"])):
                    yield frame
            except (BackendError, ModelBusy) as e:
                yield writer.failed(str(e), "model_busy" if isinstance(e, ModelBusy) else "server_error")

        return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers)

    # ── agent ─────────────────────────────────────────────────────────────
    # The routes only talk to an AgentHost (hosting.py): agents run either in this process
    # (sandbox "local") or each in its own Kubernetes pod (sandbox "k8s").
    # Pods are the sandbox, so with k8s any device may use the agent. With "local" the tools act
    # directly on this computer, so other devices need the API key.

    def agent_access(request: Request) -> None:
        if host.kind == "k8s" or cfg.api_key:
            return    # pods sandbox the agent; with an API key require_key has already checked it
        client_ip = request.client.host if request.client else ""
        try:
            local = ipaddress.ip_address(client_ip).is_loopback
        except ValueError:
            local = False
        if not local:
            raise HTTPException(403, "With agent.sandbox \"local\" the agent runs commands directly on this computer, "
                                     "so other devices need server.api_key. Use agent.sandbox \"k8s\" to run each "
                                     "agent in its own pod instead.")

    agent_deps = [Depends(require_key), Depends(agent_access)]

    def agent_model(model_id: str) -> str:
        p = cfg.profile(model_id)
        if not p or not p.tools:
            raise HTTPException(400, f"'{model_id}' is not a model marked for agent use (\"tools\": true)")
        return p.id

    @app.exception_handler(HostError)
    async def host_error(_req: Request, exc: HostError):
        return JSONResponse({"error": {"message": exc.message}}, status_code=exc.status)

    @app.get("/api/agent/info", dependencies=agent_deps)
    async def agent_info():
        return {
            **await host.info(),
            "modes": MODES, "default_mode": cfg.agent.default_mode,
            "models": [{"id": p.id, "name": p.name, "context": p.context} for p in cfg.profiles if p.tools],
            "tools": [{"name": t.name, "risk": t.risk.value, "description": t.description} for t in runtime.tools()],
            "mcp_errors": runtime.startup_errors,
        }

    @app.get("/api/agent/sessions", dependencies=agent_deps)
    async def agent_sessions():
        return {"sessions": await host.list()}

    @app.post("/api/agent/sessions", dependencies=agent_deps)
    async def agent_create(payload: dict):
        return await host.create(agent_model(payload.get("model", "")), payload.get("mode") or cfg.agent.default_mode,
                                 payload.get("workspace"))

    @app.get("/api/agent/sessions/{session_id}", dependencies=agent_deps)
    async def agent_get(session_id: str):
        view = await host.get(session_id)
        return {**view, "queued": dispatcher.position(session_id), "start_error": dispatcher.errors.get(session_id)}

    @app.delete("/api/agent/sessions/{session_id}", dependencies=agent_deps)
    async def agent_delete(session_id: str):
        await host.delete(session_id)
        return {"ok": True}

    @app.patch("/api/agent/sessions/{session_id}", dependencies=agent_deps)
    async def agent_update(session_id: str, payload: dict):
        changes = {k: payload[k] for k in ("title", "mode", "workspace", "model") if k in payload}
        if "model" in changes:
            changes["model"] = agent_model(changes["model"])
        return await host.update(session_id, changes)

    @app.post("/api/agent/sessions/{session_id}/turn", dependencies=agent_deps)
    async def agent_turn(session_id: str, payload: dict):
        position = await dispatcher.submit(session_id, str(payload.get("prompt", "")))
        return {"ok": True, "position": position}       # 0: starting now; n: n-th in line

    @app.get("/api/agent/sessions/{session_id}/events", dependencies=agent_deps)
    async def agent_events(session_id: str, start: int = 0):
        await host.get(session_id)    # answer 404 before the stream starts
        return StreamingResponse(host.events(session_id, start), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/agent/sessions/{session_id}/permission", dependencies=agent_deps)
    async def agent_permission(session_id: str, payload: dict):
        answer = Answer(bool(payload.get("allow")), bool(payload.get("always")), str(payload.get("reason", "")))
        await host.permission(session_id, str(payload.get("id", "")), answer)
        return {"ok": True}

    @app.post("/api/agent/sessions/{session_id}/cancel", dependencies=agent_deps)
    async def agent_cancel(session_id: str):
        return {"cancelled": await dispatcher.cancel(session_id)}

    @app.exception_handler(HTTPException)
    async def http_error(_req: Request, exc: HTTPException):
        return JSONResponse({"error": {"message": exc.detail}}, status_code=exc.status_code)

    # ── web UI ────────────────────────────────────────────────────────────

    @app.middleware("http")
    async def revalidate_ui(request: Request, call_next):
        # Make browsers (phones especially) check for a newer UI on every load instead of
        # running stale cached JS after an update; unchanged files still come from cache (304).
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    # Fixed root paths that browsers and "web app" wrappers probe for an icon without reading the HTML.
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(STATIC / "icons" / "favicon.ico", media_type="image/x-icon")

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    async def apple_touch_icon():
        return FileResponse(STATIC / "icons" / "apple-touch-icon.png", media_type="image/png")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest():
        return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
