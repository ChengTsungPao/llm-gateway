import asyncio
import json
from pathlib import Path

import pytest

from llm_gateway.backends import Backend, BackendError, LlamaCppBackend, _is_pinned, _same_model
from llm_gateway.config import ConfigError, load_config, read_env_file
from llm_gateway.manager import ModelManager


def write_config(tmp_path: Path, profiles: list[dict], **extra) -> Path:
    cfg = {"llamacpp": {"server_binary": "llama-server"}, "profiles": profiles, **extra}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


OLLAMA = {"id": "a", "name": "A", "backend": "ollama", "model": "a:1b", "context": 4096}
LLAMA = {"id": "b", "name": "B", "backend": "llamacpp", "model_path": "b.gguf", "context": 8192, "args": ["-ngl", "99"]}


def test_example_config_is_valid(monkeypatch):
    root = Path(__file__).parent.parent
    for k, v in read_env_file(root / ".env.example").items():
        monkeypatch.setenv(k, v)
    cfg = load_config(root / "config.example.json")
    assert cfg.profiles and all(p.context > 0 for p in cfg.profiles)
    assert cfg.ollama_url == "http://127.0.0.1:11434"

    monkeypatch.setenv("OLLAMA_URL", "http://host.docker.internal:11434")
    cfg_docker = load_config(root / "config.example.json")
    assert cfg_docker.ollama_url == "http://host.docker.internal:11434"


def test_rejects_bad_profiles(tmp_path):
    with pytest.raises(ConfigError, match="backend"):
        load_config(write_config(tmp_path, [dict(OLLAMA, backend="vllm")]))
    with pytest.raises(ConfigError, match="model_path"):
        load_config(write_config(tmp_path, [{k: v for k, v in LLAMA.items() if k != "model_path"}]))
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write_config(tmp_path, [OLLAMA, OLLAMA]))


def test_pinned_matching():
    assert _is_pinned("bge-m3:latest", ("bge-m3",))
    assert not _is_pinned("bge-m3-large:latest", ("bge-m3",))


def test_same_model_handles_implicit_latest_tag():
    assert _same_model("my-model:latest", "my-model")
    assert _same_model("qwen3:8b", "qwen3:8b")
    assert not _same_model("qwen3:8b", "qwen3:14b")


def test_ollama_history_survives_broken_tool_arguments():
    from llm_gateway.backends import _ollama_messages
    msgs = _ollama_messages([
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": '{"a": '}}]},
        {"role": "tool", "tool_call_id": "c", "name": "f", "content": "Error (bad_json)"},
    ])
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert msgs[1]["tool_name"] == "f"


def test_llamacpp_think_flag_maps_to_template_kwargs(tmp_path):
    cfg = load_config(write_config(tmp_path, [LLAMA]))
    b = LlamaCppBackend(cfg.profiles[0], cfg, client=None)
    body = b._body({"messages": [], "think": False}, stream=True)
    assert "think" not in body and body["chat_template_kwargs"] == {
        "enable_thinking": False, "reasoning_effort": "low", "reasoning_strength": "low"}
    assert body["model"] == "b" and body["stream"] is True
    on = b._body({"messages": [], "think": True, "chat_template_kwargs": {"x": 1}}, stream=False)
    assert on["chat_template_kwargs"] == {"enable_thinking": True, "x": 1}


async def test_llamacpp_refuses_port_held_by_leftover_server(tmp_path):
    import httpx

    (tmp_path / "llama-server").write_text("")
    (tmp_path / "b.gguf").write_text("")
    cfg = load_config(write_config(tmp_path, [dict(LLAMA, model_path=str(tmp_path / "b.gguf"))],
                                   llamacpp={"server_binary": str(tmp_path / "llama-server")}))
    alive = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok", "models": []}))
    async with httpx.AsyncClient(transport=alive) as client:
        b = LlamaCppBackend(cfg.profiles[0], cfg, client)
        with pytest.raises(BackendError, match="already in use"):
            await b.load()
        assert b.proc is None  # nothing was started


def test_placement_is_exposed(tmp_path):
    cfg = load_config(write_config(tmp_path, [dict(LLAMA, placement="gpu+cpu")]))
    assert cfg.profiles[0].public()["placement"] == "gpu+cpu"


class FakeBackend(Backend):
    log: list[str] = []

    def __init__(self, profile, fail=False):
        self.profile, self.fail = profile, fail

    async def load(self):
        await asyncio.sleep(0.01)
        if self.fail:
            raise BackendError("boom")
        FakeBackend.log.append(f"load {self.profile.id}")

    async def unload(self):
        FakeBackend.log.append(f"unload {self.profile.id}")


@pytest.fixture
def manager(tmp_path):
    FakeBackend.log = []
    cfg = load_config(write_config(tmp_path, [OLLAMA, LLAMA]))
    return ModelManager(cfg, FakeBackend)


async def test_only_one_model_loaded(manager):
    await manager.activate("a")
    await manager.activate("a")  # already active: no reload
    await manager.activate("b")
    assert FakeBackend.log == ["load a", "unload a", "load b"]
    assert manager.active_id == "b" and manager.status == "ready"


async def test_forced_unload_waits_instead_of_refusing(manager):
    done = []

    async def request():
        async with manager.use("a"):
            await asyncio.sleep(0.03)
            done.append("request")

    async def shutdown():
        await asyncio.sleep(0.01)
        await manager.unload(force=True)
        done.append("unloaded")

    await asyncio.gather(request(), shutdown())
    assert done == ["request", "unloaded"] and manager.active_id is None


async def test_load_failure_is_reported(tmp_path):
    cfg = load_config(write_config(tmp_path, [OLLAMA]))
    m = ModelManager(cfg, lambda p: FakeBackend(p, fail=True))
    with pytest.raises(BackendError):
        await m.activate("a")
    assert m.status == "error" and m.error == "boom" and m.active_id is None


def test_app_icons_and_manifest_are_served(tmp_path):
    from fastapi.testclient import TestClient
    from llm_gateway.server import create_app

    client = TestClient(create_app(load_config(write_config(tmp_path, [OLLAMA]))))
    for path, ctype in (("/favicon.ico", "image/x-icon"), ("/apple-touch-icon.png", "image/png"),
                        ("/manifest.webmanifest", "application/manifest+json")):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype), path
    for icon in client.get("/manifest.webmanifest").json()["icons"]:
        assert client.get(icon["src"]).status_code == 200, icon["src"]
    html = client.get("/").text
    assert 'rel="manifest"' in html and 'rel="apple-touch-icon"' in html


async def test_unknown_profile(manager):
    with pytest.raises(KeyError):
        await manager.activate("nope")
