"""Model backends. Each one can load, unload and answer OpenAI-style chat requests.

Ollama: translated to the native /api/chat so the profile's context size
(num_ctx) applies — Ollama's own OpenAI endpoint cannot set it.
llama.cpp: a llama-server process started with the profile's tuned args;
it already speaks the OpenAI protocol, so requests pass straight through.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import psutil

from .config import Config, Profile


class BackendError(RuntimeError):
    pass


_job = None


def _die_with_gateway(proc: subprocess.Popen) -> None:
    """Windows: put the child in a job object that is killed when the gateway process exits,
    even if the gateway is killed outright. Otherwise an orphaned llama-server keeps holding GPU
    memory and the next model silently spills into shared memory."""
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    if _job is None:
        class LimitInfo(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        job = k32.CreateJobObjectW(None, None)
        info = LimitInfo(LimitFlags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return
        _job = job
    handle = k32.OpenProcess(0x0101, False, proc.pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if handle:
        k32.AssignProcessToJobObject(wintypes.HANDLE(_job), wintypes.HANDLE(handle))
        k32.CloseHandle(wintypes.HANDLE(handle))


def _sse(obj: dict | str) -> bytes:
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


class Backend:
    def __init__(self, profile: Profile, cfg: Config, client: httpx.AsyncClient):
        self.profile = profile
        self.cfg = cfg
        self.client = client

    async def load(self) -> None: ...
    async def unload(self) -> None: ...
    async def chat(self, body: dict) -> dict: ...
    def chat_stream(self, body: dict) -> AsyncIterator[bytes]: ...


# ── Ollama ────────────────────────────────────────────────────────────────


def _is_pinned(name: str, pinned: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ":") for p in pinned)


def _same_model(a: str, b: str) -> bool:
    """Ollama lists `foo:latest` for a model pulled or created as `foo`."""
    norm = lambda s: s if ":" in s else s + ":latest"  # noqa: E731
    return norm(a) == norm(b)


async def unload_ollama_model(client: httpx.AsyncClient, url: str, name: str) -> None:
    await client.post(f"{url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=60)
    # Ollama answers before its runner has released GPU memory. Loading the next model into a
    # still-occupied GPU spills it into shared memory (~8x slower), so wait until it is gone.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            ps = (await client.get(f"{url}/api/ps", timeout=5)).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            break
        if not any(_same_model(m.get("name", ""), name) for m in ps):
            break
        await asyncio.sleep(0.5)
    await asyncio.sleep(1.5)  # driver-level release lags the ps listing slightly


async def evict_ollama(client: httpx.AsyncClient, cfg: Config, keep: str | None, required: bool = True) -> None:
    """Unload every model Ollama holds except `keep` and pinned ones (e.g. an embedding model
    another service depends on). Also clears models left behind if the gateway was killed."""
    try:
        ps = (await client.get(f"{cfg.ollama_url}/api/ps", timeout=10)).json().get("models", [])
    except (httpx.HTTPError, ValueError) as e:
        if required:
            raise BackendError(f"Ollama is not reachable at {cfg.ollama_url}: {e}") from e
        return
    for m in ps:
        name = m.get("name", "")
        if (keep and _same_model(name, keep)) or _is_pinned(name, cfg.ollama_pinned):
            continue
        await unload_ollama_model(client, cfg.ollama_url, name)


def _json_object(raw) -> dict:
    """Ollama wants tool-call arguments as an object. A model's earlier call may have had broken
    JSON (the agent sent an error back for it); keep the history usable instead of failing."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _ollama_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        msg = {"role": m.get("role", "user")}
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts → text only
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        msg["content"] = content or ""
        if msg["role"] == "tool" and m.get("name"):
            msg["tool_name"] = m["name"]   # tells the model which call this result answers
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {"function": {"name": tc["function"]["name"], "arguments": _json_object(tc["function"].get("arguments"))}}
                for tc in m["tool_calls"]
            ]
        out.append(msg)
    return out


def _openai_tool_calls(calls: list[dict] | None) -> list[dict] | None:
    if not calls:
        return None
    return [
        {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
         "function": {"name": c["function"]["name"],
                      "arguments": json.dumps(c["function"].get("arguments", {}), ensure_ascii=False)}}
        for c in calls
    ]


class OllamaBackend(Backend):
    @property
    def url(self) -> str:
        return self.cfg.ollama_url

    async def load(self) -> None:
        await evict_ollama(self.client, self.cfg, keep=self.profile.model)
        r = await self.client.post(
            f"{self.url}/api/generate",
            json={"model": self.profile.model, "keep_alive": -1, "options": {"num_ctx": self.profile.context}},
            timeout=900,
        )
        if r.status_code != 200:
            raise BackendError(f"Ollama failed to load {self.profile.model}: {r.text[:300]}")

    async def unload(self) -> None:
        try:
            await unload_ollama_model(self.client, self.cfg.ollama_url, self.profile.model)
        except httpx.HTTPError:
            pass

    def _request(self, body: dict, stream: bool) -> dict:
        options = {"num_ctx": self.profile.context}
        for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "num_predict"),
                         ("max_completion_tokens", "num_predict"), ("stop", "stop"), ("seed", "seed")):
            if body.get(src) is not None:
                options[dst] = body[src]
        req = {"model": self.profile.model, "messages": _ollama_messages(body.get("messages", [])),
               "stream": stream, "options": options, "keep_alive": -1}
        if body.get("tools"):
            req["tools"] = body["tools"]
        if "think" in body:
            req["think"] = body["think"]
        return req

    @staticmethod
    def _no_thinking_support(status: int, text: str) -> bool:
        return status == 400 and "does not support thinking" in text

    async def chat(self, body: dict) -> dict:
        r = await self.client.post(f"{self.url}/api/chat", json=self._request(body, False), timeout=None)
        if self._no_thinking_support(r.status_code, r.text) and "think" in body:
            body = {k: v for k, v in body.items() if k != "think"}
            r = await self.client.post(f"{self.url}/api/chat", json=self._request(body, False), timeout=None)
        if r.status_code != 200:
            raise BackendError(f"Ollama error {r.status_code}: {r.text[:300]}")
        d = r.json()
        msg = d.get("message", {})
        out_msg = {"role": "assistant", "content": msg.get("content", "")}
        if msg.get("thinking"):
            out_msg["reasoning_content"] = msg["thinking"]
        tool_calls = _openai_tool_calls(msg.get("tool_calls"))
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        n_gen = d.get("eval_count", 0)
        gen_s = d.get("eval_duration", 0) / 1e9
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(time.time()),
            "model": self.profile.id,
            "choices": [{"index": 0, "message": out_msg,
                         "finish_reason": "tool_calls" if tool_calls else ("length" if d.get("done_reason") == "length" else "stop")}],
            "usage": {"prompt_tokens": d.get("prompt_eval_count", 0), "completion_tokens": n_gen,
                      "total_tokens": d.get("prompt_eval_count", 0) + n_gen},
            "timings": {"predicted_n": n_gen, "predicted_per_second": n_gen / gen_s if gen_s else None},
        }

    async def chat_stream(self, body: dict) -> AsyncIterator[bytes]:
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        def chunk(delta: dict, finish: str | None = None, extra: dict | None = None) -> bytes:
            c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.profile.id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                c.update(extra)
            return _sse(c)

        yield chunk({"role": "assistant", "content": ""})
        if body.get("think"):  # models without thinking reject think=true; check once before streaming
            probe = await self.client.post(f"{self.url}/api/show", json={"model": self.profile.model}, timeout=30)
            if probe.status_code == 200 and "thinking" not in probe.json().get("capabilities", []):
                body = {k: v for k, v in body.items() if k != "think"}
        async with self.client.stream("POST", f"{self.url}/api/chat", json=self._request(body, True), timeout=None) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors="replace")
                yield _sse({"error": {"message": f"Ollama error {r.status_code}: {text[:300]}"}})
                yield _sse("[DONE]")
                return
            n_calls = 0  # Ollama sends each tool call whole, possibly in separate chunks: number them all
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("error"):
                    yield _sse({"error": {"message": d["error"]}})
                    break
                msg = d.get("message", {})
                delta = {}
                if msg.get("thinking"):
                    delta["reasoning_content"] = msg["thinking"]
                if msg.get("content"):
                    delta["content"] = msg["content"]
                tool_calls = _openai_tool_calls(msg.get("tool_calls"))
                if tool_calls:
                    delta["tool_calls"] = [dict(tc, index=n_calls + i) for i, tc in enumerate(tool_calls)]
                    n_calls += len(tool_calls)
                if delta:
                    yield chunk(delta)
                if d.get("done"):
                    n_gen = d.get("eval_count", 0)
                    gen_s = d.get("eval_duration", 0) / 1e9
                    finish = "tool_calls" if n_calls else "length" if d.get("done_reason") == "length" else "stop"
                    yield chunk({}, finish, {
                        "usage": {"prompt_tokens": d.get("prompt_eval_count", 0), "completion_tokens": n_gen,
                                  "total_tokens": d.get("prompt_eval_count", 0) + n_gen},
                        "timings": {"predicted_n": n_gen, "predicted_per_second": n_gen / gen_s if gen_s else None},
                    })
        yield _sse("[DONE]")


# ── llama.cpp ─────────────────────────────────────────────────────────────


class LlamaCppBackend(Backend):
    def __init__(self, profile: Profile, cfg: Config, client: httpx.AsyncClient):
        super().__init__(profile, cfg, client)
        self.proc: subprocess.Popen | None = None
        self.log_path = cfg.log_dir / f"llamacpp-{profile.id}.log"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.cfg.llamacpp_port}"

    def command(self) -> list[str]:
        return [self.cfg.llamacpp_binary, "-m", self.profile.model_path, "-c", str(self.profile.context),
                "--host", "127.0.0.1", "--port", str(self.cfg.llamacpp_port), "--jinja", "-fa", "on",
                *self.profile.args]

    def _log_tail(self, n: int = 6) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        hits = [l for l in lines if any(k in l.lower() for k in ("error", "failed", "out of memory"))]
        return " | ".join((hits or lines)[-n:])

    async def load(self) -> None:
        for path, what in ((self.cfg.llamacpp_binary, "llama-server binary"), (self.profile.model_path, "model file")):
            if not Path(path).exists():
                raise BackendError(f"{what} not found: {path}")
        # Free the GPU of any Ollama model first (Ollama may be absent: that's fine here).
        await evict_ollama(self.client, self.cfg, keep=None, required=False)
        try:  # a server left over from a gateway that was killed would still hold GPU memory
            await self.client.get(f"{self.url}/health", timeout=2)
            raise BackendError(f"port {self.cfg.llamacpp_port} is already in use, probably by a leftover "
                               "llama-server from an earlier run; stop it and try again")
        except httpx.HTTPError:
            pass
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        log.close()
        _die_with_gateway(self.proc)
        deadline = time.monotonic() + self.cfg.llamacpp_startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BackendError(f"llama-server exited during startup: {self._log_tail()}")
            try:
                r = await self.client.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(1)
        await self.unload()
        raise BackendError(f"llama-server did not become ready within {self.cfg.llamacpp_startup_timeout_s:.0f}s")

    async def unload(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                await asyncio.to_thread(self.proc.wait, 15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                await asyncio.to_thread(self.proc.wait, 10)
        self.proc = None

    def _body(self, body: dict, stream: bool) -> dict:
        out = dict(body, model=self.profile.id, stream=stream)
        if "think" in out:  # Ollama-style flag → llama.cpp chat-template switch
            think = bool(out.pop("think"))
            kwargs = {"enable_thinking": think}
            if not think:  # templates without an off switch: lowest reasoning level (gpt-oss, Muse)
                kwargs |= {"reasoning_effort": "low", "reasoning_strength": "low"}
            out["chat_template_kwargs"] = kwargs | (out.get("chat_template_kwargs") or {})
        return out

    async def chat(self, body: dict) -> dict:
        r = await self.client.post(f"{self.url}/v1/chat/completions", json=self._body(body, False), timeout=None)
        if r.status_code != 200:
            raise BackendError(f"llama-server error {r.status_code}: {r.text[:300]}")
        d = r.json()
        d["model"] = self.profile.id
        return d

    async def chat_stream(self, body: dict) -> AsyncIterator[bytes]:
        async with self.client.stream("POST", f"{self.url}/v1/chat/completions",
                                      json=self._body(body, True), timeout=None) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors="replace")
                yield _sse({"error": {"message": f"llama-server error {r.status_code}: {text[:300]}"}})
                yield _sse("[DONE]")
                return
            async for raw in r.aiter_raw():
                yield raw


async def free_gpu(client: httpx.AsyncClient, cfg: Config) -> None:
    """Clear what the gateway may have lost track of: non-pinned Ollama models and a llama-server
    still listening on the gateway's port."""
    await evict_ollama(client, cfg, keep=None, required=False)
    for c in psutil.net_connections("tcp"):
        if c.laddr and c.laddr.port == cfg.llamacpp_port and c.status == psutil.CONN_LISTEN and c.pid:
            with contextlib.suppress(psutil.Error):
                proc = psutil.Process(c.pid)
                proc.kill()
                await asyncio.to_thread(proc.wait, 10)


def make_backend(profile: Profile, cfg: Config, client: httpx.AsyncClient) -> Backend:
    return {"ollama": OllamaBackend, "llamacpp": LlamaCppBackend}[profile.backend](profile, cfg, client)
