"""Auto-compaction: keep a conversation inside the model's context window.

When a request would not fit, older messages are summarized by the model that is
already loaded (no model switch), and the summary is folded into the system prompt.
Without this, Ollama silently drops the oldest messages and llama.cpp rejects the
request outright.

Token counts are estimated (no tokenizer round-trip): CJK characters count as one
token each, other text as ~3.5 characters per token. The trigger leaves headroom
for that imprecision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable

CJK = re.compile(r"[　-鿿가-힯豈-﫿＀-￯]")
PER_MESSAGE = 6          # role markers / template overhead per message
DEFAULT_REPLY = 1024     # room reserved for the answer when max_tokens is not given
TRIGGER = 0.85           # compact when the prompt would use more than this share of the budget
KEEP_RECENT = 0.40       # share of the budget kept as verbatim recent messages

SUMMARY_PROMPT = (
    "You compress conversations. Summarize the conversation below so it can replace the "
    "original messages. Keep every fact, decision, name, number, code identifier, file path, "
    "user preference, instruction and open question exactly; drop pleasantries and filler. "
    "Write in the same language and script the conversation uses (keep Traditional Chinese "
    "as Traditional Chinese). Output only the summary, as short bullet points."
)
MERGE_PROMPT = (
    "Merge these partial summaries of one conversation into a single shorter list. Do not drop "
    "any fact, number, name, instruction or decision; remove only duplicates and filler. Keep "
    "the original language and script. Output only the bullet points."
)


def text_of(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def estimate_text(text: str) -> int:
    cjk = len(CJK.findall(text))
    return cjk + round((len(text) - cjk) / 3.5)


def estimate(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += PER_MESSAGE + estimate_text(text_of(m.get("content")))
        if m.get("tool_calls"):
            total += estimate_text(json.dumps(m["tool_calls"], ensure_ascii=False))
    return total


def _transcript(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m.get('role', 'user')}]\n{text_of(m.get('content'))}" for m in messages)


def _chunks(text: str, max_tokens: int) -> list[str]:
    """Split text into pieces of at most ~max_tokens, on paragraph boundaries when possible."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        while estimate_text(para) > max_tokens:  # a single huge paragraph: cut by characters
            cut = max(1, int(len(para) * max_tokens / estimate_text(para)))
            if cur:
                out.append(cur)
                cur = ""
            out.append(para[:cut])
            para = para[cut:]
        if cur and estimate_text(cur) + estimate_text(para) > max_tokens:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


Summarizer = Callable[[list[dict], int], Awaitable[str]]


class Compactor:
    def __init__(self, cache_size: int = 64):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._size = cache_size

    def _cached(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _store(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._size:
            self._cache.popitem(last=False)

    async def compact(self, body: dict, context: int, summarize: Summarizer) -> tuple[dict, dict | None]:
        """Return (body, info). `info` is None when nothing had to change."""
        messages = body.get("messages") or []
        reply = int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_REPLY)
        budget = max(512, context - reply)
        before = estimate(messages)
        if before <= budget * TRIGGER:
            return body, None

        # Leading system messages are always kept; then keep recent messages from the end.
        n_sys = 0
        while n_sys < len(messages) and messages[n_sys].get("role") == "system":
            n_sys += 1
        system, rest = messages[:n_sys], messages[n_sys:]
        keep_budget = int(budget * KEEP_RECENT)
        recent: list[dict] = []
        used = 0
        for m in reversed(rest):
            cost = estimate([m])
            if recent and used + cost > keep_budget:
                break
            recent.insert(0, m)
            used += cost
        old = rest[: len(rest) - len(recent)]

        # The newest message alone is too big: summarize its body too, keep it as the question.
        if not old and recent and estimate(recent) > budget * TRIGGER:
            big = recent[0]
            old = [dict(big)]
            recent = [{"role": big.get("role", "user"),
                       "content": "(The long content of this message is summarized above.) "
                                  "Answer based on the summary."}]

        if not old:
            return body, None

        key = hashlib.sha256(json.dumps(old, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        summary = self._cached(key)
        cached = summary is not None
        if summary is None:
            # Summarize each piece on its own and concatenate. Rewriting one rolling summary
            # per piece loses early facts: a fact from piece 1 vanished once piece 2 was mostly noise.
            piece_budget = max(256, int(budget * 0.6))
            out_tokens = min(1024, max(256, budget // 4))
            parts = []
            for piece in _chunks(_transcript(old), piece_budget):
                parts.append((await summarize([{"role": "system", "content": SUMMARY_PROMPT},
                                               {"role": "user", "content": piece}], out_tokens)).strip())
            summary = "\n".join(p for p in parts if p)
            if estimate_text(summary) > budget * 0.45:  # rare: many pieces; merge once, keeping all facts
                summary = (await summarize([{"role": "system", "content": MERGE_PROMPT},
                                            {"role": "user", "content": summary}], out_tokens)).strip()
            self._store(key, summary)

        note = f"Summary of the earlier conversation ({len(old)} messages were compacted):\n{summary}"
        if system:
            first = dict(system[0], content=f"{text_of(system[0].get('content'))}\n\n{note}")
            new_messages = [first, *system[1:], *recent]
        else:
            new_messages = [{"role": "system", "content": note}, *recent]
        info = {"compacted_messages": len(old), "tokens_before": before,
                "tokens_after": estimate(new_messages), "cached": cached}
        return dict(body, messages=new_messages), info
