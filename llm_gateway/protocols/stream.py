"""Turn the internal Chat Completions SSE stream into a few simple events.

Backends stream OpenAI chunks (`data: {...}` lines). Tool calls arrive in pieces keyed by `index`,
text and reasoning arrive as deltas. The adapters only want to know what happened, in order:

    Reasoning(text)            a piece of the model's reasoning
    Text(text)                 a piece of the visible answer
    ToolStart(index, id, name) a tool call begins
    ToolArgs(index, json)      a piece of its JSON arguments
    Finish(reason, usage)      the end: "stop" | "tool_calls" | "length", and token counts
"""

from __future__ import annotations

import asyncio
import codecs
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from ..backends import BackendError
from ..compaction import estimate, estimate_text


@dataclass
class Reasoning:
    text: str


@dataclass
class Text:
    text: str


@dataclass
class ToolStart:
    index: int
    id: str
    name: str


@dataclass
class ToolArgs:
    index: int
    json: str


@dataclass
class Finish:
    reason: str = "stop"
    usage: dict = field(default_factory=dict)


Event = Reasoning | Text | ToolStart | ToolArgs | Finish


async def chunks(pieces: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    """Parse SSE bytes (which may split anywhere, even inside a UTF-8 character) into chunk dicts."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    async for piece in pieces:
        buf += decoder.decode(piece)
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            if "error" in chunk:
                err = chunk["error"]
                raise BackendError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
            yield chunk


async def events(pieces: AsyncIterator[bytes]) -> AsyncIterator[Event]:
    started: dict[int, bool] = {}
    finish = Finish()
    async for chunk in chunks(pieces):
        if chunk.get("usage"):
            finish.usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                yield Reasoning(delta["reasoning_content"])
            if delta.get("content"):
                yield Text(delta["content"])
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", len(started))
                fn = tc.get("function") or {}
                if not started.get(i):
                    started[i] = True
                    yield ToolStart(i, tc.get("id") or f"call_{uuid.uuid4().hex[:24]}", fn.get("name") or "")
                args = fn.get("arguments")
                if args:
                    yield ToolArgs(i, args if isinstance(args, str) else json.dumps(args))
            if choice.get("finish_reason"):
                finish.reason = choice["finish_reason"]
    if started and finish.reason == "stop":
        finish.reason = "tool_calls"     # some servers say "stop" even when the turn ends in tool calls
    yield finish


async def with_heartbeat(source: AsyncIterator[bytes], beat: bytes | Callable[[], bytes],
                         every_s: float = 10) -> AsyncIterator[bytes]:
    """Pass `source` through, inserting `beat` (bytes, or a function returning them) whenever it is
    silent for `every_s` seconds. Waiting in the queue, loading a model or reading a long prompt can
    take minutes; clients drop silent streams."""
    it = source.__aiter__()
    pending = asyncio.ensure_future(it.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=every_s)
            if not done:
                yield beat() if callable(beat) else beat
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            yield item
            pending = asyncio.ensure_future(it.__anext__())
    finally:
        if not pending.done():
            pending.cancel()


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def text_of(content) -> str:
    """Text of a content value that may be a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            out.append(part["text"])
        elif part.get("type") in ("image", "input_image", "document", "input_file"):
            out.append("[attachment omitted: this model cannot read images or files]")
    return "\n".join(out)


def merge_user_runs(messages: list[dict]) -> list[dict]:
    """Join consecutive plain user messages: several chat templates require user/assistant turns
    to alternate, and both protocols can produce two user messages in a row."""
    out: list[dict] = []
    for m in messages:
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            out[-1] = {"role": "user", "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(m)
    return out


def prompt_size(chat: dict) -> int:
    """Estimated prompt tokens of a Chat Completions request, tool definitions included."""
    return estimate(chat["messages"]) + estimate_text(json.dumps(chat.get("tools", []), ensure_ascii=False))