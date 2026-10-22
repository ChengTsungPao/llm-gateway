"""One Kubernetes pod per agent (PodHost), managed with kubectl.

    host folder (agent.k8s.agents_dir)          pod llm-agents/agent-<id>
    data/agents/<id>/workspace   ──hostPath──>   /agent/workspace   the agent's files
    data/agents/<id>/state       ──hostPath──>   /agent/state       its session checkpoint

How the pieces talk:

    browser ──> gateway /api/agent/...  ──kubectl proxy──> K8s API server ──> pod :8000 (podserver.py)
    pod ──> http://host.docker.internal:8790/v1  (model calls go back to the gateway, which owns the GPU)

The pod is the sandbox. It sees only its own folder, it has no Kubernetes credentials
(automountServiceAccountToken: false), and CPU and memory are capped. That is why agents in pods may be
used from any device: the worst a runaway command can do is wreck its own folder.

kubectl is used as a subprocess on purpose: every action below is a command you can also type by
hand to see what happens (`kubectl -n llm-agents get pods`). The official Python client would work too.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path, PureWindowsPath

import httpx

from ..config import K8sConfig
from .hosting import HostError, check_mode
from .permissions import Answer
from .session import Session, SessionStore

POD_PORT = 8000


def host_path(path: Path, prefix: str) -> str:
    """Windows path -> the path Docker Desktop's Kubernetes node sees it under.
    D:\\a\\b -> /run/desktop/mnt/host/d/a/b. Other systems: unchanged."""
    if sys.platform != "win32":
        return str(path)
    p = PureWindowsPath(path)
    return f"{prefix.rstrip('/')}/{p.drive.rstrip(':').lower()}/" + "/".join(p.parts[1:])


def pod_name(agent_id: str) -> str:
    return "agent-" + re.sub(r"[^a-z0-9-]", "-", agent_id.lower())


class Kubectl:
    def __init__(self, context: str, namespace: str):
        self.base = ["kubectl", "--context", context, "-n", namespace]

    async def run(self, *args: str, stdin: str | None = None, check: bool = True) -> str:
        proc = await asyncio.create_subprocess_exec(
            *self.base, *args, stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate(stdin.encode() if stdin else None)
        if check and proc.returncode != 0:
            raise HostError(502, f"kubectl {' '.join(args[:3])} failed: {err.decode(errors='replace').strip()[:400]}")
        return out.decode(errors="replace")


class PodManager:
    """Creates, finds and deletes agent pods."""

    def __init__(self, cfg: K8sConfig, env: dict[str, str]):
        self.cfg = cfg
        self.env = env                  # GATEWAY_URL, MODEL_CONTEXTS, ... for every pod
        self.kubectl = Kubectl(cfg.kube_context, cfg.namespace)
        self._locks: dict[str, asyncio.Lock] = {}

    def manifest(self, agent_id: str) -> dict:
        folder = host_path(self.cfg.agents_dir / agent_id, self.cfg.host_mount_prefix)
        env = [{"name": k, "value": v} for k, v in {**self.env, "AGENT_ID": agent_id}.items()]
        return {
            "apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": pod_name(agent_id), "namespace": self.cfg.namespace,
                         "labels": {"app": "llm-agent", "agent-id": pod_name(agent_id)[6:]}},
            "spec": {
                "automountServiceAccountToken": False,      # no Kubernetes credentials inside the sandbox
                "restartPolicy": "Always",
                "containers": [{
                    "name": "agent",
                    "image": self.cfg.image,
                    "imagePullPolicy": "IfNotPresent",      # the image is built locally, not pulled
                    "env": env,
                    "ports": [{"containerPort": POD_PORT}],
                    "workingDir": "/agent/workspace",
                    "volumeMounts": [{"name": "home", "mountPath": "/agent"}],
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"},
                                  "limits": {"cpu": self.cfg.cpu, "memory": self.cfg.memory}},
                    "readinessProbe": {"httpGet": {"path": "/healthz", "port": POD_PORT},
                                       "periodSeconds": 2, "failureThreshold": 3},
                    "securityContext": {"allowPrivilegeEscalation": False},
                }],
                "volumes": [{"name": "home", "hostPath": {"path": folder, "type": "DirectoryOrCreate"}}],
            },
        }

    async def ensure_namespace(self) -> None:
        ns = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": self.cfg.namespace}}
        await Kubectl(self.cfg.kube_context, "default").run("apply", "-f", "-", stdin=json.dumps(ns))

    async def status(self, agent_id: str) -> str:
        """'ready', 'starting', 'failed: <reason>' or 'none'."""
        out = await self.kubectl.run("get", "pod", pod_name(agent_id), "-o", "json", "--ignore-not-found")
        if not out.strip():
            return "none"
        pod = json.loads(out)
        for c in pod.get("status", {}).get("containerStatuses", []):
            waiting = c.get("state", {}).get("waiting", {})
            if waiting.get("reason") in ("ErrImageNeverPull", "ErrImagePull", "ImagePullBackOff", "CrashLoopBackOff",
                                         "CreateContainerConfigError"):
                return f"failed: {waiting['reason']} {waiting.get('message', '')[:200]}"
            if c.get("ready"):
                return "ready"
        return "starting"

    async def ensure(self, agent_id: str) -> None:
        """Make sure the agent's pod exists and is ready (creating it if needed)."""
        lock = self._locks.setdefault(agent_id, asyncio.Lock())
        async with lock:   # two requests for a stopped agent must not create it twice
            state = await self.status(agent_id)
            if state == "ready":
                return
            if state == "none":
                await self.kubectl.run("apply", "-f", "-", stdin=json.dumps(self.manifest(agent_id)))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.cfg.startup_timeout_s
            while loop.time() < deadline:
                state = await self.status(agent_id)
                if state == "ready":
                    return
                if state.startswith("failed"):
                    raise HostError(502, f"agent pod {pod_name(agent_id)} {state}")
                await asyncio.sleep(1)
            raise HostError(504, f"agent pod {pod_name(agent_id)} was not ready within {self.cfg.startup_timeout_s:.0f}s")

    async def delete(self, agent_id: str) -> None:
        await self.kubectl.run("delete", "pod", pod_name(agent_id), "--ignore-not-found", "--wait=true",
                               "--grace-period=5")


class KubectlProxy:
    """`kubectl proxy` on a local port: the gateway reaches pods through the API server
    (/api/v1/namespaces/<ns>/pods/<pod>:8000/proxy/...), so pods need no exposed ports at all."""

    def __init__(self, context: str):
        self.context = context
        self.proc: subprocess.Popen | None = None
        self.url = ""

    def start(self) -> None:
        from ..backends import _die_with_gateway
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(["kubectl", "--context", self.context, "proxy", "--port", "0"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags)
        _die_with_gateway(self.proc)
        line = self.proc.stdout.readline()          # "Starting to serve on 127.0.0.1:53817"
        m = re.search(r"(\d+\.\d+\.\d+\.\d+:\d+)", line)
        if not m:
            self.stop()
            raise RuntimeError(f"kubectl proxy did not start: {line.strip()}")
        self.url = f"http://{m.group(1)}"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class PodHost:
    """AgentHost that runs each agent in its own pod."""
    kind = "k8s"

    def __init__(self, cfg: K8sConfig, pod_env: dict[str, str], default_mode: str):
        self.cfg = cfg
        self.default_mode = default_mode
        self.pods = PodManager(cfg, pod_env)
        self.proxy = KubectlProxy(cfg.kube_context)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))
        self.started = False
        self.error = ""

    async def start(self) -> None:
        try:
            self.cfg.agents_dir.mkdir(parents=True, exist_ok=True)
            await self.pods.ensure_namespace()
            self.proxy.start()
            self.started, self.error = True, ""
        except (HostError, RuntimeError, OSError) as e:
            self.error = f"Kubernetes is not available ({getattr(e, 'message', e)}). Is Kubernetes enabled in Docker Desktop?"

    async def close(self) -> None:
        self.proxy.stop()
        await self.client.aclose()

    # ── helpers ───────────────────────────────────────────────────────────

    async def _ready(self) -> None:
        if not self.started:
            await self.start()      # Kubernetes may have been switched on since the gateway started
        if not self.started:
            raise HostError(503, self.error)

    def _store(self, agent_id: str) -> SessionStore:
        return SessionStore(self.cfg.agents_dir / agent_id / "state" / "sessions")

    def _url(self, agent_id: str, path: str) -> str:
        return (f"{self.proxy.url}/api/v1/namespaces/{self.cfg.namespace}/pods/"
                f"{pod_name(agent_id)}:{POD_PORT}/proxy{path}")

    def _exists(self, agent_id: str) -> None:
        if not re.fullmatch(r"[\w-]+", agent_id) or not (self.cfg.agents_dir / agent_id / "state").is_dir():
            raise HostError(404, f"no agent {agent_id}")

    def _with_folder(self, view: dict, agent_id: str) -> dict:
        return {**view, "host_folder": str(self.cfg.agents_dir / agent_id / "workspace"), "pod": pod_name(agent_id)}

    async def _call(self, method: str, agent_id: str, path: str, body: dict | None = None) -> dict:
        await self._ready()
        self._exists(agent_id)
        await self.pods.ensure(agent_id)
        try:
            r = await self.client.request(method, self._url(agent_id, path), json=body)
        except httpx.HTTPError as e:
            raise HostError(502, f"cannot reach agent pod: {type(e).__name__}") from e
        if r.status_code >= 400:
            try:
                msg = r.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                msg = r.text[:300]
            raise HostError(r.status_code, msg)
        return r.json()

    # ── AgentHost ─────────────────────────────────────────────────────────

    async def info(self) -> dict:
        return {"sandbox": self.kind, "agents_dir": str(self.cfg.agents_dir), "namespace": self.cfg.namespace,
                "error": self.error}

    async def list(self) -> list[dict]:
        out = []
        for d in self.cfg.agents_dir.iterdir() if self.cfg.agents_dir.is_dir() else []:
            out += self._store(d.name).list()
        return sorted(out, key=lambda m: m["updated"], reverse=True)

    async def create(self, model: str, mode: str, workspace: str | None) -> dict:
        await self._ready()
        check_mode(mode)
        session = Session(workspace="/agent/workspace", model=model, mode=mode)
        (self.cfg.agents_dir / session.id / "workspace").mkdir(parents=True)
        self._store(session.id).save(session)      # the pod loads this checkpoint when it starts
        return self._with_folder(await self._call("GET", session.id, "/session"), session.id)

    async def get(self, agent_id: str) -> dict:
        self._exists(agent_id)
        if self.started and await self.pods.status(agent_id) == "ready":
            return self._with_folder(await self._call("GET", agent_id, "/session"), agent_id)
        # pod not running: show the checkpoint; the pod is started again on the next action
        s = self._store(agent_id).load(agent_id)
        return self._with_folder({**s.meta(), "messages": s.messages, "todos": s.todos, "summary": s.summary,
                                  "running": False, "pending_permissions": []}, agent_id)

    async def delete(self, agent_id: str) -> None:
        self._exists(agent_id)
        if self.started:
            await self.pods.delete(agent_id)
        shutil.rmtree(self.cfg.agents_dir / agent_id)

    async def update(self, agent_id: str, changes: dict) -> dict:
        if "workspace" in changes:
            raise HostError(400, "In Kubernetes mode each agent works in its own folder; it cannot be changed.")
        return self._with_folder(await self._call("PATCH", agent_id, "/session", changes), agent_id)

    async def start_turn(self, agent_id: str, prompt: str) -> None:
        await self._call("POST", agent_id, "/turn", {"prompt": prompt})

    async def events(self, agent_id: str, start: int) -> AsyncIterator[bytes]:
        await self._ready()
        self._exists(agent_id)
        await self.pods.ensure(agent_id)
        async with self.client.stream("GET", self._url(agent_id, f"/events?start={start}")) as r:
            async for chunk in r.aiter_raw():       # pass the pod's SSE stream through untouched
                yield chunk

    async def permission(self, agent_id: str, request_id: str, answer: Answer) -> None:
        await self._call("POST", agent_id, "/permission",
                         {"id": request_id, "allow": answer.allow, "always": answer.always, "reason": answer.reason})

    async def cancel(self, agent_id: str) -> bool:
        return (await self._call("POST", agent_id, "/cancel"))["cancelled"]
