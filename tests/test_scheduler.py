"""The model scheduler (manager.py): queue instead of refusing, chat first on the same model,
agent tasks keep their model until they finish, compatible profiles share the loaded weights."""

import asyncio
import json

import pytest

from llm_gateway.backends import Backend
from llm_gateway.config import load_config
from llm_gateway.manager import ModelBusy, ModelManager

PROFILES = [
    {"id": "small", "name": "S", "backend": "ollama", "model": "qwen3:8b", "context": 8192},
    {"id": "small-32k", "name": "S32", "backend": "ollama", "model": "qwen3:8b", "context": 32768},
    {"id": "other", "name": "O", "backend": "ollama", "model": "gpt-oss:20b", "context": 8192},
]


class Fake(Backend):
    log: list[str] = []

    def __init__(self, profile):
        self.profile = profile

    async def load(self):
        Fake.log.append(f"load {self.profile.id}")

    async def unload(self):
        Fake.log.append(f"unload {self.profile.id}")


@pytest.fixture
def m(tmp_path):
    Fake.log = []
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"profiles": PROFILES}))
    return ModelManager(load_config(p), Fake)


async def tick(n=5):
    for _ in range(n):
        await asyncio.sleep(0)


async def call(m, model, order, name, hold=0.0, **who):
    async with m.use(model, **who):
        order.append(name)
        await asyncio.sleep(hold)


async def test_other_model_waits_instead_of_failing(m):
    order = []
    first = asyncio.create_task(call(m, "small", order, "a", hold=0.05))
    await tick()
    second = asyncio.create_task(call(m, "other", order, "b"))
    await tick()
    assert m.snapshot()["queue"][0]["model"] == "other"        # waiting, not refused
    await asyncio.gather(first, second)
    assert order == ["a", "b"] and Fake.log == ["load small", "unload small", "load other"]


async def test_same_model_chat_goes_before_the_agents_next_call(m):
    order = []
    running = asyncio.create_task(call(m, "small", order, "agent-1", hold=0.05, source="agent", lease="t1"))
    await tick()
    agent_next = asyncio.create_task(call(m, "small", order, "agent-2", source="agent", lease="t1"))
    await tick()
    chat = asyncio.create_task(call(m, "small", order, "chat", source="chat"))
    await asyncio.gather(running, agent_next, chat)
    assert order == ["agent-1", "chat", "agent-2"]               # the running call was not interrupted


async def test_other_model_waits_for_the_whole_agent_task(m):
    order = []
    await call(m, "small", order, "agent-1", source="agent", lease="t1")
    chat = asyncio.create_task(call(m, "other", order, "chat", source="chat"))
    await tick()
    # between agent steps (running a tool): the chat must not take the GPU away
    await call(m, "small", order, "agent-2", source="agent", lease="t1")
    assert order == ["agent-1", "agent-2"] and not chat.done()
    m.release_lease("t1")                                        # the task ended
    await asyncio.wait_for(chat, 2)
    assert order == ["agent-1", "agent-2", "chat"] and m.active_id == "other"


async def test_compatible_profiles_share_the_loaded_weights(m):
    order = []
    await call(m, "small-32k", order, "agent", source="agent")
    await call(m, "small", order, "chat", source="chat")          # 8K profile on the 32K load
    assert Fake.log == ["load small-32k"] and m.active_id == "small-32k"
    await call(m, "small-32k", order, "agent")
    assert Fake.log == ["load small-32k"]


async def test_idle_lease_is_dropped(m):
    m.lease_idle_s = 0.05
    order = []
    await call(m, "small", order, "agent", source="agent", lease="t1")
    await asyncio.sleep(0.06)                                    # e.g. stuck waiting for a permission
    await asyncio.wait_for(call(m, "other", order, "chat", source="chat"), 6)
    assert m.active_id == "other"


async def test_queue_timeout_and_cancellation(m):
    m.max_wait_s = 0.05
    order = []
    holder = asyncio.create_task(call(m, "small", order, "agent", hold=0.3, source="agent", lease="t1"))
    await tick()
    with pytest.raises(ModelBusy):
        await call(m, "other", order, "too-late", source="chat")
    m.max_wait_s = 60
    waiting = asyncio.create_task(call(m, "other", order, "gone", request_id="r1"))
    await tick()
    assert m.queue_status("r1")["position"] == 1
    waiting.cancel()                                              # the client disconnected
    await tick()
    assert m.queue_status("r1") is None and not m.snapshot()["queue"]
    await holder


async def test_status_lists_running_queue_and_leases(m):
    order = []
    holder = asyncio.create_task(call(m, "small", order, "agent", hold=0.05, source="agent", lease="t1"))
    await tick()
    asyncio.create_task(call(m, "other", order, "api"))
    asyncio.create_task(call(m, "small", order, "chat", source="chat"))
    await tick()
    snap = m.snapshot()
    assert snap["running"]["source"] == "agent"
    assert [q["source"] for q in snap["queue"]] == ["chat", "api"]        # chat ahead in the order
    assert snap["leases"][0]["id"] == "t1"
    await holder


class SlowLoad(Fake):
    async def load(self):
        Fake.log.append(f"load {self.profile.id}")
        await asyncio.sleep(10)


async def test_a_cancelled_load_does_not_stay_loading(tmp_path):
    """A caller that disconnects mid-load must not leave "loading" behind (it blocked the queue
    and Release) nor a half-started server nobody tracks."""
    Fake.log = []
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"profiles": PROFILES}))
    m = ModelManager(load_config(p), SlowLoad)
    task = asyncio.create_task(call(m, "small", [], "a"))
    await tick()
    assert m.snapshot()["status"] == "loading"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snap = m.snapshot()
    assert (snap["status"], snap["loading"], snap["busy"]) == ("idle", None, False)
    assert Fake.log == ["load small", "unload small"]


async def test_release_now_frees_the_gpu_while_a_lease_holds_it(m):
    async with m.use("small", source="agent", lease="t1"):
        pass
    await m.release_now()                       # unload() would wait for the lease
    snap = m.snapshot()
    assert (snap["active"], snap["status"], snap["leases"]) == (None, "idle", [])
    assert Fake.log == ["load small", "unload small"]
