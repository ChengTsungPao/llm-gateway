"""Failure handling: classify first, then route.

Every tool reports failure the same way, as a ToolError with four fields (the "unified adapter"):

    code         short machine-readable reason, e.g. "not_found", "timeout"
    message      what went wrong, written for the model to read
    retryable    True only for transient failures (network blips, a busy server)
    side_effect  True if the tool may already have changed something before it failed

From those fields the loop decides what to do. The question is "would retrying 100 times help?":

    kind           example                         what the loop does
    -------------  ------------------------------  ------------------------------------------
    model_error    arguments are not valid JSON    tell the model, it fixes its own call
    invalid_args   missing field, wrong type       tell the model + show the tool schema
    transient      timeout, 502, connection reset  retry here with exponential backoff + jitter
    side_effect    command timed out half-way      never retry blindly; tell the model to check state
    silent         "success" with an empty result  not an error; add a hint so the model double-checks
    failed         anything else                   tell the model

Guards around the whole loop (also here): a step budget, a loop detector for repeated identical
calls, and a per-tool circuit breaker so a broken tool stops being called.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


class ToolError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False, side_effect: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.side_effect = side_effect


def classify(err: ToolError) -> str:
    if err.code in ("bad_json", "unknown_tool"):
        return "model_error"
    if err.code == "invalid_args":
        return "invalid_args"
    if err.side_effect:
        return "side_effect"  # checked before retryable: never repeat something that may have happened
    if err.retryable:
        return "transient"
    return "failed"


def feedback(err: ToolError, schema: dict | None = None) -> str:
    """The text the model sees for a failed call: the error plus how to recover."""
    kind = classify(err)
    text = f"Error ({err.code}): {err.message}"
    if kind in ("model_error", "invalid_args") and schema:
        text += "\nFix the call. The tool's parameters are:\n" + json.dumps(schema, ensure_ascii=False)
    elif kind == "side_effect":
        text += "\nThe action may have partly happened. Check the current state before trying again."
    elif kind == "transient":
        text += "\nThis kept failing after several retries. Try a different approach."
    return text


async def retry_transient(call: Callable[[], Awaitable[str]], *, attempts: int = 3, base_s: float = 0.5,
                          on_retry: Callable[[int, ToolError, float], None] | None = None) -> str:
    """Run `call`; retry only errors marked retryable, waiting base * 2^n + jitter between tries."""
    for n in range(attempts):
        try:
            return await call()
        except ToolError as e:
            if not e.retryable or e.side_effect or n == attempts - 1:
                raise
            delay = base_s * 2 ** n + random.uniform(0, base_s)  # jitter spreads out simultaneous retries
            if on_retry:
                on_retry(n + 1, e, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


@dataclass
class CircuitBreaker:
    """After `threshold` consecutive failures a tool is switched off for the rest of the session."""
    threshold: int = 3
    failures: dict[str, int] = field(default_factory=dict)

    def is_open(self, tool: str) -> bool:
        return self.failures.get(tool, 0) >= self.threshold

    def record(self, tool: str, ok: bool) -> None:
        self.failures[tool] = 0 if ok else self.failures.get(tool, 0) + 1


_INTENT = re.compile(r"\b(let me(?! know)|let's|i will|i'll|i am going to|i'm going to|now i|next,? i|我(來|會|將|現在))\b"
                     r"[^.!?。\n]*[.:!。]?\s*$", re.IGNORECASE)


def announces_unfinished_work(text: str) -> bool:
    """Small models often end a reply with "Let me fix this now." and no tool call. Detect a
    final sentence that announces an action, so the loop can tell the model to go ahead."""
    tail = text.strip()[-300:]
    if "?" in tail or "？" in tail:     # it asked the user something: wait for the answer
        return False
    return bool(tail) and bool(_INTENT.search(tail))


@dataclass
class LoopDetector:
    """Small models sometimes repeat the exact same call forever. The first repeat gets a warning;
    calling it a third time in a row stops the turn."""
    last: str | None = None
    repeats: int = 0

    def check(self, name: str, args: dict) -> int:
        key = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
        self.repeats = self.repeats + 1 if key == self.last else 0
        self.last = key
        return self.repeats

    def reset(self) -> None:
        self.last, self.repeats = None, 0
