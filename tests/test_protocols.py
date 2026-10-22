"""/v1/messages (Claude Code) and /v1/responses (Codex): translation and streaming, end to end
through the gateway with a scripted backend instead of a GPU."""

import json

import pytest
from fastapi.testclient import TestClient

from llm_gateway import server
from llm_gateway.config import load_config
from llm_gateway.protocols import anthropic, responses


def chunk(delta=None, finish=None, usage=None) -> bytes:
    c = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return f"data: {json.dumps(c)}\n\n".encode()


# A streamed turn: some text, then one tool call whose JSON arguments arrive in two pieces.
# The second piece is split in the middle of a UTF-8 character on purpose.
TOOL_TURN = [
    chunk({"role": "assistant", "content": ""}),
    chunk({"content": "Let me look."}),
    chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "Read", "arguments": ""}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"file_path": "a'}}]}),
    *[bytes([b]) for b in chunk({"tool_calls": [{"index": 0, "function": {"arguments": '.py", "note": "é"}'}}]})],
    chunk({}, "tool_calls", {"prompt_tokens": 120, "completion_tokens": 9}),
    b"data: [DONE]\n\n",
]


class ScriptedBackend:
    seen: list[dict] = []

    def __init__(self, profile, cfg, client):
        self.profile = profile

    async def load(self):
        pass

    async def unload(self):
        pass

    async def chat(self, body):
        ScriptedBackend.seen.append(body)
        return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "ok", "tool_calls": [
                {"id": "call_9", "type": "function", "function": {"name": "apply_patch", "arguments": '{"input": "*** Begin Patch"}'}}]}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5}}

    async def chat_stream(self, body):
        ScriptedBackend.seen.append(body)
        for piece in TOOL_TURN:
            yield piece


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "make_backend", lambda p, cfg, c: ScriptedBackend(p, cfg, c))
    ScriptedBackend.seen = []
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"llamacpp": {"server_binary": "x"}, "server": {"auto_compact": False},
                                    "profiles": [{"id": "local", "name": "L", "backend": "ollama", "model": "l",
                                                  "context": 32768, "tools": True}]}))
    return TestClient(server.create_app(load_config(cfg_path)))


def sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        name = next((l[7:] for l in block.splitlines() if l.startswith("event: ")), None)
        data = next((l[6:] for l in block.splitlines() if l.startswith("data: ")), None)
        if name and data:
            out.append((name, json.loads(data)))
    return out


# ── Anthropic / Claude Code ────────────────────────────────────────────────

def test_anthropic_request_translation():
    chat = anthropic.to_chat({
        "system": [{"type": "text", "text": "You are Claude Code."}],
        "thinking": {"type": "adaptive"},
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}, {"type": "web_search_20250305", "name": "web"}],
        "messages": [
            {"role": "user", "content": "read a.py"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "x"},
                                              {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "x = 1"}]},
                                         {"type": "text", "text": "now explain"}]},
            {"role": "system", "content": "<system-reminder>be brief</system-reminder>"},
        ]}, "local")
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]       # mid-conversation system merged into user
    assert chat["messages"][2]["tool_calls"][0]["function"] == {"name": "Read", "arguments": '{"file_path": "a.py"}'}
    assert chat["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "x = 1"}
    assert "be brief" in chat["messages"][4]["content"] and "now explain" in chat["messages"][4]["content"]
    assert [t["function"]["name"] for t in chat["tools"]] == ["Read"]      # server tools skipped
    assert chat["think"] is False                                          # adaptive -> fast


def test_anthropic_stream_is_a_valid_event_sequence(client):
    client.post("/api/activate", json={"id": "local"})
    r = client.post("/v1/messages", json={"model": "claude-sonnet-5-5", "max_tokens": 100, "stream": True,
                                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    evs = sse_events(r.text)
    names = [n for n, _ in evs]
    assert names == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                     "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"]
    assert evs[1][1]["content_block"]["type"] == "text" and evs[2][1]["delta"]["text"] == "Let me look."
    tool = evs[4][1]["content_block"]
    assert tool == {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}}
    args = "".join(d["delta"]["partial_json"] for n, d in evs if d.get("delta", {}).get("type") == "input_json_delta")
    assert json.loads(args) == {"file_path": "a.py", "note": "é"}           # survived the split UTF-8 byte
    assert evs[-2][1]["delta"]["stop_reason"] == "tool_use" and evs[-2][1]["usage"]["input_tokens"] == 120
    # an unknown model name (Claude Code's own) falls back to the loaded model
    assert ScriptedBackend.seen[-1]["model"] == "local"


def test_anthropic_errors_and_token_count(client):
    # nothing loaded and an unknown name: a clear 404 in Anthropic's error envelope
    r = client.post("/v1/messages", json={"model": "claude-x", "max_tokens": 10, "messages": []})
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
    r = client.post("/v1/messages/count_tokens", json={"model": "local", "messages": [{"role": "user", "content": "hello"}]})
    assert r.json()["input_tokens"] > 0


# ── Responses / Codex ──────────────────────────────────────────────────────

def test_responses_request_translation():
    body = {"instructions": "You are Codex.", "reasoning": {"effort": "medium"},
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}},
                      {"type": "custom", "name": "apply_patch", "description": "Edit files",
                       "format": {"type": "grammar", "syntax": "lark", "definition": "start: ..."}},
                      {"type": "web_search"}],
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "env info"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]},
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": '{"command": ["ls"]}'},
                {"type": "function_call_output", "call_id": "c1", "output": "a.py"},
                {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "*** Begin Patch"},
                {"type": "custom_tool_call_output", "call_id": "c2", "output": "Done"},
            ]}
    chat = responses.to_chat(body, "local")
    assert [m["role"] for m in chat["messages"]] == ["system", "system", "user", "assistant", "tool", "assistant", "tool"]
    assert chat["messages"][5]["tool_calls"][0]["function"]["arguments"] == '{"input": "*** Begin Patch"}'
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names == ["shell", "apply_patch"]                                # web_search dropped
    assert "start: ..." in chat["tools"][1]["function"]["description"]      # the grammar reaches the model
    assert responses.tool_kinds(body) == {"shell": "function", "apply_patch": "custom"}


def test_responses_stream_ends_with_completed_items(client):
    r = client.post("/v1/responses", json={"model": "local", "stream": True, "input": "hi",
                                           "tools": [{"type": "function", "name": "Read", "parameters": {"type": "object"}}]})
    evs = sse_events(r.text)
    names = [n for n, _ in evs]
    assert names[:2] == ["response.created", "response.in_progress"] and names[-1] == "response.completed"
    seqs = [d["sequence_number"] for _, d in evs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    done = [d["item"] for n, d in evs if n == "response.output_item.done"]
    assert done[0]["type"] == "message" and done[0]["content"][0]["text"] == "Let me look."
    assert done[1]["type"] == "function_call" and json.loads(done[1]["arguments"])["file_path"] == "a.py"
    final = evs[-1][1]["response"]
    assert final["status"] == "completed" and len(final["output"]) == 2 and final["usage"]["input_tokens"] == 120


def test_responses_custom_tool_comes_back_as_custom_call(client):
    r = client.post("/v1/responses", json={"model": "local", "input": "patch it",
                                           "tools": [{"type": "custom", "name": "apply_patch"}]})
    out = r.json()["output"]
    assert out[1] == {**out[1], "type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch",
                      "call_id": "call_9"}
