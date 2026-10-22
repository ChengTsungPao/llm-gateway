from llm_gateway.compaction import Compactor, estimate, estimate_text


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    async def __call__(self, messages, max_tokens):
        self.calls.append(messages[-1]["content"])
        return f"summary#{len(self.calls)}"


def chat(n_pairs: int, filler: str = "lorem ipsum dolor sit amet " * 40) -> list[dict]:
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_pairs):
        msgs += [{"role": "user", "content": f"q{i} {filler}"}, {"role": "assistant", "content": f"a{i} {filler}"}]
    msgs.append({"role": "user", "content": "final question"})
    return msgs


def test_estimate_counts_cjk_per_character():
    assert estimate_text("你好世界") == 4
    assert 6 <= estimate_text("hello world, how are you") <= 8  # 24 chars at ~3.5 chars/token


async def test_short_conversation_untouched():
    s = FakeSummarizer()
    body = {"messages": chat(1)}
    out, info = await Compactor().compact(body, context=8192, summarize=s)
    assert out is body and info is None and not s.calls


async def test_long_conversation_compacts_and_keeps_recent():
    s = FakeSummarizer()
    msgs = chat(30)
    out, info = await Compactor().compact({"messages": msgs, "max_tokens": 256}, context=2048, summarize=s)
    new = out["messages"]
    assert info and info["compacted_messages"] > 0 and info["tokens_after"] < info["tokens_before"]
    assert new[0]["role"] == "system" and new[0]["content"].startswith("You are helpful.")
    assert "summary#" in new[0]["content"]
    assert new[-1] == msgs[-1]                       # newest message kept verbatim
    assert sum(1 for m in new if m["role"] == "system") == 1  # summary merged, not a 2nd system msg
    assert estimate(new) <= 2048 - 256


async def test_summary_is_cached_for_the_same_history():
    s = FakeSummarizer()
    c = Compactor()
    body = {"messages": chat(30), "max_tokens": 256}
    await c.compact(body, 2048, s)
    calls = len(s.calls)
    _, info = await c.compact(body, 2048, s)
    assert len(s.calls) == calls and info["cached"] is True


async def test_single_huge_message_is_summarized_in_pieces():
    s = FakeSummarizer()
    huge = "這是很長的內容。" * 3000                    # ~24k tokens, far above the context
    body = {"messages": [{"role": "user", "content": huge}], "max_tokens": 256}
    out, info = await Compactor().compact(body, 2048, s)
    assert info and len(s.calls) > 1                 # split into several summarization passes
    assert estimate(out["messages"]) <= 2048 - 256
