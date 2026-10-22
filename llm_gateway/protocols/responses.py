"""OpenAI Responses API (/v1/responses) <-> Chat Completions. This is what Codex speaks.

    Responses                                   Chat Completions
    instructions                            ->  first "system" message
    input: message (user/developer/assistant) ->  user / system / assistant messages
    input: function_call, custom_tool_call  ->  assistant tool_calls
    input: *_call_output                    ->  "tool" messages
    input: reasoning                        ->  dropped (the model re-reasons)
    tools: function                         ->  function tools as they are
    tools: custom (free-form, e.g. apply_patch) -> a function with one string argument "input"
    tools: local_shell                      ->  a function "local_shell" taking the command as a list
    reasoning.effort high / xhigh           ->  think: true   (lower effort -> off: faster)

The answer is rebuilt as output items (message, function_call, custom_tool_call, local_shell_call)
and, when streaming, as the response.* event sequence Codex reads, ending in response.completed.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from ..compaction import estimate_text
from .stream import Finish, Reasoning, Text, ToolArgs, ToolStart, merge_user_runs, sse, text_of

HEARTBEAT = b": keep-alive\n\n"      # an SSE comment; clients ignore it, but the connection stays alive
LOCAL_SHELL_PARAMS = {"type": "object", "properties": {
    "command": {"type": "array", "items": {"type": "string"}, "description": "Program and arguments"},
    "workdir": {"type": "string"}, "timeout_ms": {"type": "integer"}}, "required": ["command"]}


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def tool_kinds(body: dict) -> dict[str, str]:
    """name -> "function" | "custom" | "local_shell", to turn calls back into the right item type."""
    kinds = {}
    for t in body.get("tools") or []:
        if t.get("type") == "function":
            kinds[t["name"]] = "function"
        elif t.get("type") == "custom":
            kinds[t["name"]] = "custom"
        elif t.get("type") == "local_shell":
            kinds["local_shell"] = "local_shell"
    return kinds


def to_chat(body: dict, model: str) -> dict:
    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input") or []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]

    def assistant_for_call() -> dict:
        # calls attach to the assistant message just before them, or start a new one
        if messages and messages[-1]["role"] == "assistant":
            return messages[-1]
        messages.append({"role": "assistant", "content": ""})
        return messages[-1]

    for item in items:
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind == "message":
            role = item.get("role", "user")
            text = text_of(item.get("content"))
            if role in ("system", "developer"):
                # only a leading system message is safe in most chat templates
                role = "system" if all(m["role"] == "system" for m in messages) else "user"
            messages.append({"role": role if role in ("system", "assistant") else "user", "content": text})
        elif kind in ("function_call", "custom_tool_call", "local_shell_call"):
            if kind == "function_call":
                name, args = item["name"], item.get("arguments") or "{}"
            elif kind == "custom_tool_call":
                name, args = item["name"], json.dumps({"input": item.get("input", "")}, ensure_ascii=False)
            else:
                name, args = "local_shell", json.dumps(item.get("action") or {}, ensure_ascii=False)
            msg = assistant_for_call()
            msg.setdefault("tool_calls", []).append(
                {"id": item.get("call_id") or item.get("id"), "type": "function", "function": {"name": name, "arguments": args}})
        elif kind in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            out = item.get("output")
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"),
                             "content": out if isinstance(out, str) else text_of(out)})
        # reasoning and other item types carry nothing a local model can use
    chat = {"model": model, "messages": merge_user_runs(messages),
            "think": (body.get("reasoning") or {}).get("effort") in ("high", "xhigh")}
    for src, dst in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(src) is not None:
            chat[dst] = body[src]
    tools = []
    for t in body.get("tools") or []:
        if t.get("type") == "function":
            tools.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                           "parameters": t.get("parameters") or {"type": "object"}}})
        elif t.get("type") == "custom":
            fmt = t.get("format") or {}
            hint = f"\nThe input must follow this {fmt.get('syntax', 'format')}:\n{fmt['definition']}" \
                if fmt.get("definition") else ""
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description", "") + hint,
                "parameters": {"type": "object", "properties": {"input": {"type": "string",
                               "description": "The complete raw input for this tool"}}, "required": ["input"]}}})
        elif t.get("type") == "local_shell":
            tools.append({"type": "function", "function": {"name": "local_shell", "description":
                          "Run a command on the user's machine", "parameters": LOCAL_SHELL_PARAMS}})
    if tools:
        chat["tools"] = tools
    return chat



def call_item(kind: str, call_id: str, name: str, args: str) -> dict:
    """A finished tool call as the output item type Codex expects for that tool."""
    if kind == "custom":
        try:
            raw = json.loads(args).get("input", args)
        except (json.JSONDecodeError, AttributeError):
            raw = args
        return {"type": "custom_tool_call", "id": _id("ctc"), "call_id": call_id, "name": name, "input": raw,
                "status": "completed"}
    if kind == "local_shell":
        try:
            action = json.loads(args)
        except json.JSONDecodeError:
            action = {}
        return {"type": "local_shell_call", "id": _id("lsh"), "call_id": call_id, "status": "completed",
                "action": {"type": "exec", "command": action.get("command") or [], "workdir": action.get("workdir"),
                           "timeout_ms": action.get("timeout_ms"), "env": {}}}
    return {"type": "function_call", "id": _id("fc"), "call_id": call_id, "name": name, "arguments": args,
            "status": "completed"}


def _usage(usage: dict, input_estimate: int, out_estimate: int) -> dict:
    i = usage.get("prompt_tokens") or input_estimate
    o = usage.get("completion_tokens") or out_estimate
    return {"input_tokens": i, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": o,
            "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": i + o}


def response_object(rid: str, model: str, status: str, output: list, usage: dict | None) -> dict:
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model,
            "output": output, "usage": usage, "error": None, "incomplete_details": None}


def from_chat(resp: dict, model: str, kinds: dict[str, str], input_estimate: int) -> dict:
    msg = resp["choices"][0]["message"]
    output = []
    if msg.get("reasoning_content"):
        output.append({"type": "reasoning", "id": _id("rs"),
                       "summary": [{"type": "summary_text", "text": msg["reasoning_content"]}]})
    if msg.get("content"):
        output.append({"type": "message", "id": _id("msg"), "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": msg["content"], "annotations": []}]})
    for tc in msg.get("tool_calls") or []:
        name = tc["function"]["name"]
        output.append(call_item(kinds.get(name, "function"), tc["id"], name, tc["function"].get("arguments") or "{}"))
    out_est = estimate_text(msg.get("content") or "")
    return response_object(_id("resp"), model, "completed", output, _usage(resp.get("usage") or {}, input_estimate, out_est))


class StreamWriter:
    """Builds the response.* event sequence; every event carries a growing sequence_number."""

    def __init__(self, model: str, kinds: dict[str, str], input_estimate: int):
        self.rid, self.model, self.kinds, self.input_estimate = _id("resp"), model, kinds, input_estimate
        self.seq = 0
        self.output: list[dict] = []

    def ev(self, kind: str, **data) -> bytes:
        self.seq += 1
        return sse(kind, {"type": kind, "sequence_number": self.seq, **data})

    def start(self) -> list[bytes]:
        r = response_object(self.rid, self.model, "in_progress", [], None)
        return [self.ev("response.created", response=r), self.ev("response.in_progress", response=r)]

    async def stream(self, events: AsyncIterator) -> AsyncIterator[bytes]:
        open_item: dict | None = None      # the item currently receiving deltas
        text = ""
        args: dict[int, str] = {}
        calls: dict[int, dict] = {}

        def finish_item() -> list[bytes]:
            nonlocal open_item, text
            if open_item is None:
                return []
            item, out, pos = open_item, [], len(self.output)
            if item["type"] == "message":
                part = {"type": "output_text", "text": text, "annotations": []}
                out += [self.ev("response.output_text.done", item_id=item["id"], output_index=pos, content_index=0, text=text),
                        self.ev("response.content_part.done", item_id=item["id"], output_index=pos, content_index=0, part=part)]
                done = {**item, "status": "completed", "content": [part]}
            elif item["type"] == "reasoning":
                done = {**item, "summary": [{"type": "summary_text", "text": text}]}
            else:   # a tool call: emit it as the item type Codex expects for that tool
                c = calls[item["_index"]]
                full = args.get(item["_index"], "") or "{}"
                out.append(self.ev("response.function_call_arguments.done", item_id=item["id"], output_index=pos,
                                   arguments=full))
                done = call_item(self.kinds.get(c["name"], "function"), c["id"], c["name"], full)
                done["id"] = item["id"]
            out.append(self.ev("response.output_item.done", output_index=pos, item=done))
            self.output.append(done)
            open_item, text = None, ""
            return out

        def open_new(item: dict) -> list[bytes]:
            nonlocal open_item
            out = finish_item()
            open_item = item
            shown = {k: v for k, v in item.items() if not k.startswith("_")}
            out.append(self.ev("response.output_item.added", output_index=len(self.output), item=shown))
            if item["type"] == "message":
                out.append(self.ev("response.content_part.added", item_id=item["id"], output_index=len(self.output),
                                   content_index=0, part={"type": "output_text", "text": "", "annotations": []}))
            return out

        out_chars = 0
        async for ev in events:
            frames: list[bytes] = []
            if isinstance(ev, Reasoning):
                if not open_item or open_item["type"] != "reasoning":
                    frames += open_new({"type": "reasoning", "id": _id("rs"), "summary": []})
                text += ev.text
                frames.append(self.ev("response.reasoning_summary_text.delta", item_id=open_item["id"],
                                      output_index=len(self.output), summary_index=0, delta=ev.text))
            elif isinstance(ev, Text):
                if not open_item or open_item["type"] != "message":
                    frames += open_new({"type": "message", "id": _id("msg"), "role": "assistant",
                                        "status": "in_progress", "content": []})
                text += ev.text
                out_chars += len(ev.text)
                frames.append(self.ev("response.output_text.delta", item_id=open_item["id"],
                                      output_index=len(self.output), content_index=0, delta=ev.text))
            elif isinstance(ev, ToolStart):
                calls[ev.index] = {"id": ev.id, "name": ev.name}
                frames += open_new({"type": "function_call", "id": _id("fc"), "call_id": ev.id, "name": ev.name,
                                    "arguments": "", "status": "in_progress", "_index": ev.index})
            elif isinstance(ev, ToolArgs):
                args[ev.index] = args.get(ev.index, "") + ev.json
                out_chars += len(ev.json)
                if open_item and open_item.get("_index") == ev.index:
                    frames.append(self.ev("response.function_call_arguments.delta", item_id=open_item["id"],
                                          output_index=len(self.output), delta=ev.json))
            elif isinstance(ev, Finish):
                frames += finish_item()
                usage = _usage(ev.usage, self.input_estimate, max(1, out_chars // 4))
                status = "incomplete" if ev.reason == "length" else "completed"
                r = response_object(self.rid, self.model, status, self.output, usage)
                if status == "incomplete":
                    r["incomplete_details"] = {"reason": "max_output_tokens"}
                frames.append(self.ev("response.completed", response=r))
            for f in frames:
                yield f

    def failed(self, message: str, code: str = "server_error") -> bytes:
        r = response_object(self.rid, self.model, "failed", self.output, None)
        r["error"] = {"code": code, "message": message}
        return self.ev("response.failed", response=r)
