"""The model client: one streaming call to an OpenAI-compatible /v1/chat/completions.

It points at the gateway, but any OpenAI-compatible server works; the loop only sees
`LLMClient.stream()`. The interesting part is assembling tool calls from a stream: a call
arrives in pieces, keyed by `index`, and its JSON arguments arrive as string fragments:

    delta.tool_calls = [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]
    delta.tool_calls = [{"index": 0, "function": {"arguments": "{\"path\": \"READ"}}]
    delta.tool_calls = [{"index": 0, "function": {"arguments": "ME.md\"}"}}]
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .events import ReasoningDelta, TextDelta


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Completion:
    """The assembled assistant turn."""
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # OpenAI format, arguments as JSON strings
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    compacted: int = 0     # messages the gateway compacted (should stay 0: the agent manages its context)

    def message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 900,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.transport = transport      # tests pass a mock transport

    async def stream(self, messages: list[dict], tools: list[dict], *, think: bool = False,
                     max_tokens: int | None = None, temperature: float | None = None,
                     ) -> AsyncIterator[TextDelta | ReasoningDelta | Completion]:
        """Yield text/reasoning deltas as they arrive, then one final Completion.
        Retries (with backoff) only when nothing has been streamed yet and the error is transient."""
        body = {"model": self.model, "messages": messages, "stream": True, "think": think,
                "auto_compact": False}
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        for attempt in range(4):
            streamed = False
            try:
                async for item in self._stream_once(body):
                    streamed = True
                    yield item
                return
            except LLMError as e:
                if not e.retryable or streamed or attempt == 3:   # a retry after output would repeat it
                    raise
                await asyncio.sleep(1.5 * 2 ** attempt + random.uniform(0, 1))

    async def _stream_once(self, body: dict) -> AsyncIterator[TextDelta | ReasoningDelta | Completion]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        done = Completion()
        calls: dict[int, dict] = {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s, connect=10),
                                         transport=self.transport) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=body,
                                         headers=headers) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode(errors="replace")[:500]
                        # 409: another model is in use (the gateway's one-model rule); worth waiting for
                        raise LLMError(f"model server answered {r.status_code}: {text}",
                                       retryable=r.status_code in (409, 429, 502, 503))
                    async for line in r.aiter_lines():
                        if line.startswith(": llm-gateway "):
                            done.compacted = json.loads(line[14:]).get("compacted_messages", 0)
                            continue
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        if "error" in chunk:
                            err = chunk["error"]
                            raise LLMError(err.get("message", str(err)), retryable=err.get("code") == "model_busy")
                        if chunk.get("usage"):
                            done.usage = chunk["usage"]
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                done.reasoning += delta["reasoning_content"]
                                yield ReasoningDelta(delta["reasoning_content"])
                            if delta.get("content"):
                                done.content += delta["content"]
                                yield TextDelta(delta["content"])
                            for tc in delta.get("tool_calls") or []:
                                slot = calls.setdefault(tc.get("index", len(calls)),
                                                        {"id": "", "name": "", "arguments": ""})
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                slot["name"] += fn.get("name") or ""
                                args = fn.get("arguments")
                                # most servers stream argument fragments; some send a whole dict at once
                                slot["arguments"] += json.dumps(args) if isinstance(args, dict) else (args or "")
                            if choice.get("finish_reason"):
                                done.finish_reason = choice["finish_reason"]
        except httpx.HTTPError as e:
            raise LLMError(f"cannot reach the model server at {self.base_url}: {type(e).__name__}",
                           retryable=True) from e
        done.tool_calls = [
            {"id": c["id"] or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for _, c in sorted(calls.items())
        ]
        yield done

    async def complete(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Non-streaming helper (used for summaries): returns the text of the answer."""
        text = ""
        async for item in self.stream(messages, [], max_tokens=max_tokens, temperature=0.2):
            if isinstance(item, Completion):
                text = item.content
        return text
