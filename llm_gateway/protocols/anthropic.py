"""Anthropic Messages API (/v1/messages) <-> Chat Completions. This is what Claude Code speaks.

    Anthropic                                   Chat Completions
    system (string or text blocks)          ->  first "system" message
    user [text, tool_result, ...]           ->  "tool" messages (one per tool_result) + a "user" message
    assistant [text, tool_use, thinking]    ->  "assistant" with content + tool_calls (thinking dropped)
    role "system" in the middle             ->  user text (most chat templates allow system only first)
    tools [{name, input_schema}]            ->  tools [{type: function, function: {name, parameters}}]
    thinking {type: enabled}                ->  think: true   (adaptive / absent -> off: faster)

The streamed answer is rebuilt as Anthropic events, in the exact order Claude Code requires:
message_start, then per block content_block_start / _delta / _stop, then message_delta, message_stop,
with `ping` events while the model loads or reads a long prompt.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from .stream import Finish, Reasoning, Text, ToolArgs, ToolStart, merge_user_runs, sse, text_of

STOP_REASON = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
PING = sse("ping", {"type": "ping"})


def to_chat(body: dict, model: str) -> dict:
    messages: list[dict] = []
    if body.get("system"):
        messages.append({"role": "system", "content": text_of(body["system"])})
    for m in body.get("messages", []):
        role, content = m.get("role"), m.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        if role == "assistant":
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text_of([b for b in blocks if b.get("type") == "text"])}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
            continue
        for b in blocks:                      # tool results answer the previous assistant's calls
            if b.get("type") == "tool_result":
                text = text_of(b.get("content"))
                messages.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                 "content": f"[error] {text}" if b.get("is_error") else text})
        text = text_of([b for b in blocks if b.get("type") != "tool_result"])
        if text:
            messages.append({"role": "user", "content": text})
    chat = {"model": model, "messages": merge_user_runs(messages),
            "think": (body.get("thinking") or {}).get("type") == "enabled"}
    for src, dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"),
                     ("stop_sequences", "stop")):
        if body.get(src) is not None:
            chat[dst] = body[src]
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema") or {"type": "object"}}}
             for t in body.get("tools") or [] if t.get("type") in (None, "custom")]   # server tools are skipped
    if tools:
        chat["tools"] = tools
    return chat



def _usage(usage: dict, fallback_in: int, fallback_out: int = 0) -> dict:
    return {"input_tokens": usage.get("prompt_tokens") or fallback_in,
            "output_tokens": usage.get("completion_tokens") or fallback_out,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def from_chat(resp: dict, model: str, input_estimate: int) -> dict:
    choice = resp["choices"][0]
    msg = choice["message"]
    content = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": args})
    reason = "tool_calls" if msg.get("tool_calls") else choice.get("finish_reason") or "stop"
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": STOP_REASON.get(reason, "end_turn"), "stop_sequence": None,
            "usage": _usage(resp.get("usage") or {}, input_estimate)}


def message_start(model: str, input_estimate: int) -> bytes:
    return sse("message_start", {"type": "message_start", "message": {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": _usage({}, input_estimate)}})


async def stream(events: AsyncIterator, input_estimate: int) -> AsyncIterator[bytes]:
    """Chat events -> Anthropic content-block events (message_start is sent by the caller first,
    so the client sees a response immediately even while the model is still loading)."""
    index = -1
    open_kind: str | None = None          # "thinking" | "text" | "tool:<n>"
    out_chars = 0

    def close() -> list[bytes]:
        nonlocal open_kind
        if open_kind is None:
            return []
        out = []
        if open_kind == "thinking":   # Claude Code expects a signature before a thinking block ends
            out.append(sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "signature_delta", "signature": ""}}))
        out.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        open_kind = None
        return out

    def open_block(kind: str, block: dict) -> list[bytes]:
        nonlocal index, open_kind
        out = close()
        index += 1
        open_kind = kind
        out.append(sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": block}))
        return out

    async for ev in events:
        frames: list[bytes] = []
        if isinstance(ev, Reasoning):
            if open_kind != "thinking":
                frames += open_block("thinking", {"type": "thinking", "thinking": "", "signature": ""})
            frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "thinking_delta", "thinking": ev.text}}))
        elif isinstance(ev, Text):
            if open_kind != "text":
                frames += open_block("text", {"type": "text", "text": ""})
            frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "text_delta", "text": ev.text}}))
            out_chars += len(ev.text)
        elif isinstance(ev, ToolStart):
            frames += open_block(f"tool:{ev.index}", {"type": "tool_use", "id": ev.id, "name": ev.name, "input": {}})
        elif isinstance(ev, ToolArgs):
            frames.append(sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "input_json_delta", "partial_json": ev.json}}))
            out_chars += len(ev.json)
        elif isinstance(ev, Finish):
            frames += close()
            usage = _usage(ev.usage, input_estimate, max(1, out_chars // 4))
            frames.append(sse("message_delta", {"type": "message_delta",
                                                "delta": {"stop_reason": STOP_REASON.get(ev.reason, "end_turn"),
                                                          "stop_sequence": None},
                                                "usage": {"output_tokens": usage["output_tokens"],
                                                          "input_tokens": usage["input_tokens"]}}))
            frames.append(sse("message_stop", {"type": "message_stop"}))
        for f in frames:
            yield f


def error_body(kind: str, message: str) -> dict:
    """Anthropic error envelope. Claude Code retries overloaded_error (529) and api_error (5xx)."""
    return {"type": "error", "error": {"type": kind, "message": message}}
