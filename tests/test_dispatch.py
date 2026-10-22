"""Layer 2 (agent/dispatch.py): at most max_running agent turns at once; the rest wait in line.
Also: ${VAR} expansion from .env in config.json."""

import asyncio
import json

import pytest

from llm_gateway.agent.dispatch import Dispatcher
from llm_gateway.agent.hosting import HostError
from llm_gateway.config import ConfigError, load_config


class FakeHost:
    """An AgentHost whose turns run until finish() is called."""

    def __init__(self):
        self.running: set[str] = set()
        self.started: list[str] = []

    async def get(self, agent_id):
        if agent_id == "missing":
            raise HostError(404, "no agent")
        return {"id": agent_id, "title": f"task {agent_id}", "running": agent_id in self.running}

    async def start_turn(self, agent_id, prompt):
        self.running.add(agent_id)
        self.started.append(agent_id)

    async def cancel(self, agent_id):
        self.running.discard(agent_id)
        return True

    def finish(self, agent_id):
        self.running.discard(agent_id)


async def settle(s=0.05):
    await asyncio.sleep(s)


async def test_one_task_at_a_time_the_rest_wait():
    host = FakeHost()
    d = Dispatcher(host, max_running=1, poll_s=0.01)
    d.start()
    assert await d.submit("a", "first") == 0
    await settle()
    assert await d.submit("b", "second") == 1          # waits behind a
    await settle()
    assert host.started == ["a"] and d.position("b") == 1
    assert d.snapshot()["waiting"][0]["title"] == "task b"
    host.finish("a")
    await settle()
    assert host.started == ["a", "b"] and d.position("b") is None
    await d.close()


async def test_cancel_a_waiting_task_and_refuse_duplicates():
    host = FakeHost()
    d = Dispatcher(host, max_running=1, poll_s=0.01)
    d.start()
    await d.submit("a", "x")
    await settle()
    with pytest.raises(HostError):
        await d.submit("a", "again")                    # already running
    await d.submit("b", "y")
    assert await d.cancel("b") and d.position("b") is None
    with pytest.raises(HostError):
        await d.submit("missing", "z")
    await d.close()


def test_config_reads_variables_from_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLAMA_SERVER=/opt/llama-server\nPROJECTS='/home/me/code'\n# comment\n")
    (tmp_path / "config.json").write_text(json.dumps({
        "server": {"host": "${GATEWAY_HOST:-127.0.0.1}"},
        "llamacpp": {"server_binary": "${LLAMA_SERVER}"},
        "agent": {"workspaces": ["${PROJECTS}"]},
        "profiles": [{"id": "a", "name": "A", "backend": "llamacpp", "model_path": "${LLAMA_SERVER}.gguf", "context": 8192}]}))
    monkeypatch.delenv("GATEWAY_HOST", raising=False)
    cfg = load_config(tmp_path / "config.json")
    assert cfg.host == "127.0.0.1" and cfg.llamacpp_binary == "/opt/llama-server"
    assert cfg.profiles[0].model_path == "/opt/llama-server.gguf"
    monkeypatch.setenv("LLAMA_SERVER", "/from/env")                # real environment wins over .env
    assert load_config(tmp_path / "config.json").llamacpp_binary == "/from/env"
    (tmp_path / ".env").write_text("")
    monkeypatch.delenv("LLAMA_SERVER")
    with pytest.raises(ConfigError, match="LLAMA_SERVER"):
        load_config(tmp_path / "config.json")
