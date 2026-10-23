"""The program inside an agent pod: one agent, served over HTTP to the gateway.

The pod sees only its own folder, mounted at /agent:
    /agent/workspace            the agent's files (its tools cannot leave this folder; bash runs here)
    /agent/state/sessions/      the session checkpoint, so a restarted pod continues where it stopped

Inside the pod the agent is the same LocalHost the gateway uses without Kubernetes; the pod is
the sandbox. Model calls go back to the gateway (GATEWAY_URL), which owns the GPU.

Environment (set by k8s.py when it creates the pod):
    AGENT_ID, GATEWAY_URL, LLM_API_KEY, MODEL_CONTEXTS (JSON {model: tokens}), AGENT_CONFIG (JSON)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import AgentConfig
from .hosting import HostError, LocalHost
from .permissions import Answer
from .runtime import AgentRuntime
from .service import AgentService

ROOT = Path(os.environ.get("AGENT_ROOT", "/agent"))
SHARED = Path(os.environ.get("AGENT_SHARED", "/shared"))   # data/shared of the host, read-only: skills/ and AGENTS.md


def create_pod_app(env: dict[str, str] | None = None) -> FastAPI:
    env = dict(os.environ if env is None else env)
    agent_id = env["AGENT_ID"]
    extra = json.loads(env.get("AGENT_CONFIG", "{}"))
    cfg = AgentConfig(data_dir=ROOT / "state", default_workspace=ROOT / "workspace", scratch_dir=Path("/tmp"), shared_dir=SHARED,
                      default_mode=extra.get("default_mode", "default"), max_steps=int(extra.get("max_steps", 40)),
                      hooks=extra.get("hooks", {}), mcp_servers=extra.get("mcp_servers", {}))
    contexts = json.loads(env.get("MODEL_CONTEXTS", "{}"))
    runtime = AgentRuntime(cfg, env["GATEWAY_URL"], env.get("LLM_API_KEY", ""))
    host = LocalHost(runtime, AgentService(runtime, lambda model: contexts.get(model, 8192)))

    @asynccontextmanager
    async def lifespan(_app):
        await runtime.start()
        yield
        await host.close()
        await runtime.close()

    app = FastAPI(title=f"agent {agent_id}", lifespan=lifespan)

    @app.exception_handler(HostError)
    async def host_error(_req: Request, exc: HostError):
        return JSONResponse({"error": {"message": exc.message}}, status_code=exc.status)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "agent": agent_id}

    @app.get("/session")
    async def get_session():
        return await host.get(agent_id)

    @app.patch("/session")
    async def update_session(payload: dict):
        return await host.update(agent_id, payload)

    @app.post("/turn")
    async def turn(payload: dict):
        await host.start_turn(agent_id, str(payload.get("prompt", "")))
        return {"ok": True}

    @app.get("/events")
    async def events(start: int = 0):
        await host.get(agent_id)    # 404 before streaming starts, not in the middle of it
        return StreamingResponse(host.events(agent_id, start), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/permission")
    async def permission(payload: dict):
        answer = Answer(bool(payload.get("allow")), bool(payload.get("always")), str(payload.get("reason", "")))
        await host.permission(agent_id, str(payload.get("id", "")), answer)
        return {"ok": True}

    @app.post("/cancel")
    async def cancel():
        return {"cancelled": await host.cancel(agent_id)}

    @app.exception_handler(HTTPException)
    async def http_error(_req: Request, exc: HTTPException):
        return JSONResponse({"error": {"message": exc.detail}}, status_code=exc.status_code)

    return app


def main() -> None:
    import uvicorn
    uvicorn.run(create_pod_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
