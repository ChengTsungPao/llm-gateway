"""HTTP API: OpenAI-compatible /v1 routes, management /api routes, and the web UI."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psutil
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent.hosting import AgentHost, HostError, LocalHost
from .agent.k8s import PodHost
from .agent.permissions import MODES, Answer
from .agent.runtime import AgentRuntime
from .agent.service import AgentService
from .backends import BackendError, evict_ollama, make_backend
from .compaction import Compactor
from .config import Config
from .manager import ModelBusy, ModelManager

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

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if isinstance(host, PodHost):
            await host.start()          # namespace + kubectl proxy; retried on first use if Kubernetes is off
        else:
            await runtime.start()       # MCP servers (in k8s mode each pod starts its own)
        # Models left in Ollama by an earlier run hold GPU memory while the gateway shows
        # "No model loaded"; start from a clean GPU (pinned models stay).
        await evict_ollama(client, cfg, keep=None, required=False)
        yield
        await host.close()
        await runtime.close()
        await manager.unload(force=True)  # never leave a llama-server process behind
        await client.aclose()

    app = FastAPI(title="llm-gateway", lifespan=lifespan)

    def require_key(request: Request) -> None:
        if not cfg.api_key:
            return
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {cfg.api_key}":
            raise HTTPException(401, "missing or wrong API key (Authorization: Bearer <key>)")

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
        return {**manager.snapshot(), "gpu": gpu_cache["v"], "system": gpu_cache["sys"],
                "auth_required": bool(cfg.api_key)}

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
    async def unload():
        try:
            await manager.unload()
        except ModelBusy as e:
            raise HTTPException(409, str(e)) from e
        return manager.snapshot()

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

    @app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
    async def chat_completions(body: dict):
        model_id = body.get("model") or manager.active_id
        if not model_id:
            raise HTTPException(400, "no 'model' given and no model is active")
        profile = profile_or_404(model_id)
        try:
            manager.check_can_use(profile.id)
        except ModelBusy as e:
            raise HTTPException(409, str(e)) from e

        if not body.get("stream"):
            try:
                async with manager.use(profile.id) as backend:
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
        async def gen():
            nonlocal body
            try:
                async with manager.use(profile.id) as backend:
                    body, info = await fit_context(backend, profile, body)
                    if info:
                        yield f": llm-gateway {json.dumps(info)}\n\n".encode()
                    async for piece in backend.chat_stream(body):
                        yield piece
            except (BackendError, ModelBusy) as e:
                err = {"message": str(e), "code": "model_busy" if isinstance(e, ModelBusy) else "backend_error"}
                yield f"data: {json.dumps({'error': err}, ensure_ascii=False)}\n\n".encode()
                yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
        return await host.get(session_id)

    @app.delete("/api/agent/sessions/{session_id}", dependencies=agent_deps)
    async def agent_delete(session_id: str):
        await host.delete(session_id)
        return {"ok": True}

    @app.patch("/api/agent/sessions/{session_id}", dependencies=agent_deps)
    async def agent_update(session_id: str, payload: dict):
        changes = {k: payload[k] for k in ("mode", "workspace", "model") if k in payload}
        if "model" in changes:
            changes["model"] = agent_model(changes["model"])
        return await host.update(session_id, changes)

    @app.post("/api/agent/sessions/{session_id}/turn", dependencies=agent_deps)
    async def agent_turn(session_id: str, payload: dict):
        await host.start_turn(session_id, str(payload.get("prompt", "")))
        return {"ok": True}

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
        return {"cancelled": await host.cancel(session_id)}

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
