"""HTTP API: OpenAI-compatible /v1 routes, management /api routes, and the web UI."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .backends import BackendError, make_backend
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


def create_app(cfg: Config) -> FastAPI:
    client = httpx.AsyncClient()
    manager = ModelManager(cfg, lambda profile: make_backend(profile, cfg, client))
    compactor = Compactor()
    gpu_cache: dict = {"t": 0.0, "v": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
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
            gpu_cache["v"], gpu_cache["t"] = await asyncio.to_thread(_gpu_info), time.time()
        return {**manager.snapshot(), "gpu": gpu_cache["v"], "auth_required": bool(cfg.api_key)}

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
        if not cfg.auto_compact:
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
