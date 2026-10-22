"""Agent harness tests. The model is replaced by FakeLLM, which plays back a script of answers,
so every branch of the loop can be tested without a GPU."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from llm_gateway.agent import errors
from llm_gateway.agent.context import CLEARED, ContextManager
from llm_gateway.agent.events import Notice, PermissionRequest, TodoUpdate, ToolCall, ToolResult, TurnEnd
from llm_gateway.agent.hooks import Hooks
from llm_gateway.agent.llm import Completion, LLMClient
from llm_gateway.agent.loop import Agent, AgentSettings
from llm_gateway.agent.mcp import connect_servers
from llm_gateway.agent.permissions import Answer, Decision, PermissionPolicy
from llm_gateway.agent.session import Session, SessionStore
from llm_gateway.agent.skills import discover_skills
from llm_gateway.agent.tools import ToolRegistry, builtin_tools
from llm_gateway.agent.tools.base import Risk, ToolContext, validate
from llm_gateway.agent.tools.files import EditFile, ReadFile
from llm_gateway.agent.tools.git import Git
from llm_gateway.agent.tools.shell import detect_shell

SHELL = detect_shell()
PY = sys.executable.replace("\\", "/")


def call(name: str, args, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class FakeLLM:
    """Plays back scripted turns. Each entry is a Completion; `seen` records what the model was sent."""

    def __init__(self, *script: Completion):
        self.script = list(script)
        self.seen: list[list[dict]] = []
        self.lease = None
        self.released: list[str] = []

    async def release_lease(self):
        self.released.append(self.lease)

    async def stream(self, messages, tools, **kw):
        self.seen.append(messages)
        done = self.script.pop(0)
        if done.content:
            from llm_gateway.agent.events import TextDelta
            yield TextDelta(done.content)
        yield done

    async def complete(self, messages, max_tokens=1024):
        return "- summary of earlier work"


class Approve:
    def __init__(self, *answers: Answer):
        self.answers = list(answers)
        self.asked: list[str] = []

    async def ask(self, request_id, tool, arguments, summary, risk):
        self.asked.append(summary)
        return self.answers.pop(0)


def make_agent(tmp_path: Path, llm, mode="default", approver=None, hooks=None, max_steps=10) -> Agent:
    session = Session(workspace=str(tmp_path), model="fake", mode=mode)
    return Agent(session=session, llm=llm, tools=ToolRegistry(builtin_tools(SHELL)), context=ContextManager(32768),
                 approver=approver or Approve(), store=SessionStore(tmp_path / ".sessions"), hooks=hooks,
                 shell=SHELL, settings=AgentSettings(max_steps=max_steps))


async def collect(agent: Agent, prompt: str) -> list:
    return [e async for e in agent.run(prompt)]


# ── the loop ───────────────────────────────────────────────────────────────

async def test_tool_call_then_answer(tmp_path):
    (tmp_path / "pyproject.toml").write_text('version = "1.2.3"\n')
    llm = FakeLLM(Completion(tool_calls=[call("read_file", {"path": "pyproject.toml"})]),
                  Completion(content="Version 1.2.3."))
    agent = make_agent(tmp_path, llm)
    events = await collect(agent, "what version?")
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.ok and "1.2.3" in result.content
    assert isinstance(events[-1], TurnEnd) and events[-1].reason == "done" and events[-1].steps == 2
    assert llm.released == [f"agent-{agent.session.id}"]      # the turn gave its model lease back
    roles = [m["role"] for m in agent.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # the second model call saw the tool result, paired with its call id
    assert llm.seen[1][-1] == {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": result.content}
    # checkpointed: the session can be loaded back
    assert SessionStore(tmp_path / ".sessions").load(agent.session.id).messages == agent.session.messages


async def test_write_asks_permission_and_denial_reaches_the_model(tmp_path):
    approver = Approve(Answer(False, reason="not now"))
    llm = FakeLLM(Completion(tool_calls=[call("write_file", {"path": "a.txt", "content": "x"})]),
                  Completion(content="ok, I won't."))
    events = await collect(make_agent(tmp_path, llm, approver=approver), "write a file")
    assert any(isinstance(e, PermissionRequest) for e in events)
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.error_code == "denied" and "not now" in result.content
    assert not (tmp_path / "a.txt").exists()


async def test_always_allow_is_remembered(tmp_path):
    approver = Approve(Answer(True, always=True))
    llm = FakeLLM(Completion(tool_calls=[call("write_file", {"path": "a.txt", "content": "1"}, "c1")]),
                  Completion(tool_calls=[call("write_file", {"path": "b.txt", "content": "2"}, "c2")]),
                  Completion(content="done"))
    await collect(make_agent(tmp_path, llm, approver=approver), "write two files")
    assert len(approver.asked) == 1 and (tmp_path / "b.txt").read_text() == "2"


async def test_plan_mode_denies_changes_without_asking(tmp_path):
    approver = Approve()
    llm = FakeLLM(Completion(tool_calls=[call("bash", {"command": "echo hi"})]), Completion(content="plan: ..."))
    events = await collect(make_agent(tmp_path, llm, mode="plan", approver=approver), "do it")
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.error_code == "denied" and approver.asked == []
    assert "PLAN MODE" in llm.seen[0][0]["content"]


async def test_bad_arguments_are_fed_back_with_the_schema(tmp_path):
    llm = FakeLLM(Completion(tool_calls=[call("read_file", '{"path": "a.txt"')]),          # broken JSON
                  Completion(tool_calls=[call("read_file", {"file": "a.txt"}, "c2")]),      # wrong field
                  Completion(tool_calls=[call("no_such_tool", {}, "c3")]),
                  Completion(content="giving up"))
    events = await collect(make_agent(tmp_path, llm), "read a.txt")
    codes = [e.error_code for e in events if isinstance(e, ToolResult)]
    assert codes == ["bad_json", "invalid_args", "unknown_tool"]
    second = [e for e in events if isinstance(e, ToolResult)][1].content
    assert "missing required field 'path'" in second and '"properties"' in second   # schema included


async def test_repeated_identical_calls_stop_the_turn(tmp_path):
    same = lambda i: Completion(tool_calls=[call("glob", {"pattern": "*.md"}, f"c{i}")])  # noqa: E731
    llm = FakeLLM(same(1), same(2), same(3), Completion(content="never reached"))
    events = await collect(make_agent(tmp_path, llm), "find docs")
    results = [e for e in events if isinstance(e, ToolResult)]
    assert "Do something different" in results[1].content
    assert results[2].error_code == "loop" and events[-1].reason == "loop"


async def test_announced_but_not_done_gets_a_nudge(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    llm = FakeLLM(Completion(content="The bug is on line 6. Let me fix this bug and run the tests."),
                  Completion(tool_calls=[call("read_file", {"path": "a.txt"})]),
                  Completion(content="Fixed and verified."))
    events = await collect(make_agent(tmp_path, llm), "fix it")
    assert any(isinstance(e, Notice) and "announced" in e.text for e in events)
    assert events[-1].reason == "done" and events[-1].steps == 3
    assert errors.announces_unfinished_work("I'll run the tests now.")
    assert not errors.announces_unfinished_work("The tests pass. Let me know if you need anything else")


async def test_step_budget(tmp_path):
    llm = FakeLLM(*[Completion(tool_calls=[call("glob", {"pattern": f"*.{i}"}, f"c{i}")]) for i in range(5)])
    events = await collect(make_agent(tmp_path, llm, max_steps=3), "loop forever")
    assert events[-1].reason == "max_steps" and events[-1].steps == 3


async def test_todo_write_updates_ui_and_prompt(tmp_path):
    todos = [{"content": "read code", "status": "in_progress"}, {"content": "fix", "status": "pending"}]
    llm = FakeLLM(Completion(tool_calls=[call("todo_write", {"todos": todos})]), Completion(content="working"))
    agent = make_agent(tmp_path, llm)
    events = await collect(agent, "big task")
    assert next(e for e in events if isinstance(e, TodoUpdate)).todos == todos
    assert "[in_progress] read code" in llm.seen[1][0]["content"]   # working memory in the system prompt


# ── tools ──────────────────────────────────────────────────────────────────

def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace=tmp_path.resolve(), shell=SHELL)


async def test_edit_requires_read_and_a_unique_match(tmp_path):
    c = ctx(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\ny = 1\n")
    with pytest.raises(errors.ToolError, match="before editing"):
        await EditFile().run({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}, c)
    assert "1\tx = 1" in await ReadFile().run({"path": "a.py"}, c)
    with pytest.raises(errors.ToolError) as e:
        await EditFile().run({"path": "a.py", "old_string": "= 1", "new_string": "= 2"}, c)
    assert e.value.code == "not_unique"
    await EditFile().run({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}, c)
    assert (tmp_path / "a.py").read_text() == "x = 2\ny = 1\n"


async def test_glob_states_empty_and_missing_folders_plainly(tmp_path):
    from llm_gateway.agent.tools.files import Glob, Grep
    assert "is empty" in await Glob().run({"pattern": "*"}, ctx(tmp_path))
    assert "is empty" in await Grep().run({"pattern": "x"}, ctx(tmp_path))
    with pytest.raises(errors.ToolError) as e:
        await Glob().run({"pattern": "*", "path": "nope"}, ctx(tmp_path))
    assert e.value.code == "not_found"


async def test_scratch_folder_is_allowed_too(tmp_path):
    ws, scratch = tmp_path / "ws", tmp_path / "scratch"
    ws.mkdir(), scratch.mkdir()
    (scratch / "big.txt").write_text("cloned")
    c = ToolContext(workspace=ws.resolve(), shell=SHELL, scratch=scratch.resolve())
    assert "cloned" in await ReadFile().run({"path": str(scratch / "big.txt")}, c)
    with pytest.raises(errors.ToolError):
        await ReadFile().run({"path": str(tmp_path / "elsewhere.txt")}, c)


def test_no_nudge_when_the_model_asks_the_user():
    assert not errors.announces_unfinished_work("Should I delete the folder? Let me know and I'll do it.")
    assert errors.announces_unfinished_work("The folder exists. Let me remove it and clone again.")


async def test_paths_cannot_leave_the_workspace(tmp_path):
    with pytest.raises(errors.ToolError) as e:
        await ReadFile().run({"path": "../outside.txt"}, ctx(tmp_path))
    assert e.value.code == "outside_workspace"


def test_git_risk_depends_on_the_command():
    g = Git()
    assert g.risk_for({"args": "status"}) is Risk.READ
    assert g.risk_for({"args": "log --oneline -5"}) is Risk.READ
    assert g.risk_for({"args": "branch"}) is Risk.READ
    assert g.risk_for({"args": "commit -m 'x y'"}) is Risk.WRITE
    assert g.risk_for({"args": "push origin main"}) is Risk.EXEC
    assert g.risk_for({"args": "reset --hard HEAD~1"}) is Risk.EXEC


async def test_bash_and_git_run(tmp_path):
    c = ctx(tmp_path)
    from llm_gateway.agent.tools.shell import Bash
    out = await Bash(SHELL).run({"command": "echo hello"}, c)
    assert out.ok and "hello" in out.text and "[exit code 0]" in out.text
    failed = await Bash(SHELL).run({"command": "exit 3"}, c)
    assert not failed.ok and "[exit code 3]" in failed.text      # ran, but shown as a failure
    await Git().run({"args": "init -q"}, c)
    assert "No commits yet" in await Git().run({"args": "status"}, c)


def test_validate():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}, "m": {"type": "string", "enum": ["a"]}},
              "required": ["n"]}
    assert validate(schema, {"n": 1}) == []
    assert validate(schema, {"n": True}) == ["field 'n' must be integer, got bool"]
    assert "missing required field 'n'" in validate(schema, {"m": "b"})[0]


# ── errors, permissions, context ───────────────────────────────────────────

async def test_transient_errors_are_retried_with_backoff(monkeypatch):
    sleeps = []

    async def no_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(errors.asyncio, "sleep", no_sleep)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise errors.ToolError("network", "reset", retryable=True)
        return "ok"
    assert await errors.retry_transient(flaky) == "ok"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]    # exponential

    async def side_effect():
        raise errors.ToolError("timeout", "half done", retryable=True, side_effect=True)
    with pytest.raises(errors.ToolError):
        await errors.retry_transient(side_effect)
    assert len(sleeps) == 2                               # never retried


def test_circuit_breaker_and_classification():
    b = errors.CircuitBreaker(threshold=2)
    b.record("t", False)
    b.record("t", False)
    assert b.is_open("t")
    assert errors.classify(errors.ToolError("x", "y", retryable=True, side_effect=True)) == "side_effect"


def test_permission_table():
    p = PermissionPolicy("accept_edits")
    assert p.decide("edit_file", Risk.WRITE) is Decision.ALLOW
    assert p.decide("bash", Risk.EXEC) is Decision.ASK
    p.remember("bash", Risk.EXEC)
    assert p.decide("bash", Risk.EXEC) is Decision.ALLOW
    assert PermissionPolicy("plan").decide("bash", Risk.EXEC) is Decision.DENY


async def test_context_clears_old_results_then_summarizes_on_step_boundaries():
    session = Session(workspace=".", model="m")
    for i in range(8):
        session.messages += [{"role": "user", "content": f"q{i}"},
                             {"role": "assistant", "content": "", "tool_calls": [call("read_file", {"path": "x"}, f"c{i}")]},
                             {"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": "z" * 3000}]
    cm = ContextManager(4096, reply_tokens=512)
    notes = await cm.fit(session, "system", [], FakeLLM())
    assert any("summarized" in n for n in notes) and session.summary
    assert session.messages[0]["role"] == "user"
    ids = {m.get("tool_call_id") for m in session.messages if m["role"] == "tool"}
    calls = {c["id"] for m in session.messages for c in m.get("tool_calls", [])}
    assert ids == calls   # no orphaned tool results


def test_old_tool_results_are_cleared_but_the_latest_kept():
    msgs = [{"role": "tool", "content": f"r{i}"} for i in range(6)]
    assert ContextManager(4096, keep_tool_results=4).clear_old_tool_results(msgs) == 2
    assert [m["content"] for m in msgs] == [CLEARED, CLEARED, "r2", "r3", "r4", "r5"]


# ── hooks, skills, MCP, LLM stream ─────────────────────────────────────────

async def test_pre_tool_hook_can_block(tmp_path):
    hooks = Hooks({"PreToolUse": [{"matcher": "bash", "command": f'"{PY}" -c "import sys; print(\'no shell today\'); sys.exit(2)"'}]},
                  SHELL, tmp_path)
    llm = FakeLLM(Completion(tool_calls=[call("bash", {"command": "echo hi"})]), Completion(content="ok"))
    events = await collect(make_agent(tmp_path, llm, mode="bypass", hooks=hooks), "run")
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.error_code == "blocked_by_hook" and "no shell today" in result.content


async def test_stop_hook_sends_the_model_back(tmp_path):
    flag = tmp_path / "ran"
    script = f'"{PY}" -c "import pathlib,sys; p=pathlib.Path(r\'{flag}\'); done=p.exists(); p.touch(); sys.exit(0 if done else 2)"'
    hooks = Hooks({"Stop": [{"command": script}]}, SHELL, tmp_path)
    llm = FakeLLM(Completion(content="done?"), Completion(content="now really done"))
    events = await collect(make_agent(tmp_path, llm, hooks=hooks), "task")
    assert any(isinstance(e, Notice) and "Stop hook" in e.text for e in events)
    assert events[-1].reason == "done" and events[-1].steps == 2


def test_skills_are_discovered(tmp_path):
    d = tmp_path / ".agents" / "skills" / "release"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: release\ndescription: Write release notes\n---\nSteps...")
    assert discover_skills(tmp_path)["release"].description == "Write release notes"


async def test_mcp_tools_are_adapted(tmp_path):
    server = str(Path(__file__).parent / "fake_mcp_server.py")
    clients, tools, errs = await connect_servers({"fake": {"command": sys.executable, "args": [server]}})
    try:
        assert errs == [] and {t.name for t in tools} == {"mcp__fake__add", "mcp__fake__explode"}
        add = next(t for t in tools if t.name == "mcp__fake__add")
        assert add.risk is Risk.READ and await add.run({"a": 2, "b": 3}, ctx(tmp_path)) == "5"
        with pytest.raises(errors.ToolError, match="boom"):
            await next(t for t in tools if t.name == "mcp__fake__explode").run({}, ctx(tmp_path))
    finally:
        for c in clients:
            await c.close()


async def test_llm_client_assembles_streamed_tool_calls():
    chunks = [
        {"choices": [{"delta": {"content": "Let me look."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "read_file", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "REA'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'DME.md"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    llm = LLMClient("http://gw/v1", "m", transport=httpx.MockTransport(handler))
    items = [i async for i in llm.stream([{"role": "user", "content": "hi"}], [{"type": "function"}])]
    done = items[-1]
    assert done.content == "Let me look." and done.finish_reason == "tool_calls"
    assert done.tool_calls == [call("read_file", '{"path": "README.md"}', "call_a")]
    assert seen["body"]["auto_compact"] is False and seen["body"]["stream"] is True


# ── web API ────────────────────────────────────────────────────────────────

def test_agent_api_requires_local_or_key(tmp_path):
    from fastapi.testclient import TestClient

    from llm_gateway.config import load_config
    from llm_gateway.server import create_app
    (tmp_path / "ws").mkdir()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "llamacpp": {"server_binary": "x"},
        "profiles": [{"id": "a", "name": "A", "backend": "ollama", "model": "a", "context": 8192, "tools": True}],
        "agent": {"workspaces": [str(tmp_path / "ws")], "data_dir": str(tmp_path / "data")}}))
    app = create_app(load_config(cfg_path))
    remote = TestClient(app, client=("203.0.113.5", 5000))
    assert remote.get("/api/agent/info").status_code == 403
    local = TestClient(app, client=("127.0.0.1", 5000))
    assert local.get("/api/agent/info").json()["models"][0]["id"] == "a"
    r = local.post("/api/agent/sessions", json={"workspace": str(tmp_path), "model": "a"})
    assert r.status_code == 400 and "allowed workspace" in r.json()["error"]["message"]
    r = local.post("/api/agent/sessions", json={"workspace": str(tmp_path / "ws"), "model": "a", "mode": "plan"})
    assert r.status_code == 200 and r.json()["mode"] == "plan"
    sid = r.json()["id"]
    assert local.get("/api/agent/sessions").json()["sessions"][0]["id"] == sid
    assert local.patch(f"/api/agent/sessions/{sid}", json={"mode": "accept_edits"}).json()["mode"] == "accept_edits"
    assert local.post(f"/api/agent/sessions/{sid}/permission", json={"id": "nope", "allow": True}).status_code == 404

    # no workspace given: the empty default folder (data/workspace), which is always allowed
    r = local.post("/api/agent/sessions", json={"model": "a"})
    default = tmp_path / "data" / "workspace"
    assert r.status_code == 200 and Path(r.json()["workspace"]) == default and default.is_dir()
    assert not any(default.iterdir())
    # the folder of an existing task can be changed; outside the allowed folders is refused
    sid2 = r.json()["id"]
    r = local.patch(f"/api/agent/sessions/{sid2}", json={"workspace": str(tmp_path / "ws")})
    assert r.status_code == 200 and Path(r.json()["workspace"]) == tmp_path / "ws"
    assert local.patch(f"/api/agent/sessions/{sid2}", json={"workspace": str(tmp_path)}).status_code == 400
