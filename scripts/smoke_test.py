"""End-to-end check against a running gateway: chat, streaming, and switching models.

usage: uv run python scripts/smoke_test.py [base_url] [profile_a] [profile_b]
Loads real models, so it takes as long as the loads do.
"""

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790"
A = sys.argv[2] if len(sys.argv) > 2 else "qwen3-8b"
B = sys.argv[3] if len(sys.argv) > 3 else None
c = httpx.Client(base_url=BASE, timeout=900)


def status() -> dict:
    return c.get("/api/status").json()


def chat(model: str, text: str, stream: bool) -> tuple[str, float | None]:
    body = {"model": model, "think": False, "max_tokens": 80, "stream": stream,
            "messages": [{"role": "user", "content": text}]}
    if not stream:
        r = c.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], (d.get("timings") or {}).get("predicted_per_second")
    out, tps = "", None
    with c.stream("POST", "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            if "error" in j:
                raise RuntimeError(j["error"]["message"])
            out += j["choices"][0]["delta"].get("content") or ""
            tps = (j.get("timings") or {}).get("predicted_per_second") or tps
    return out, tps


def step(label: str, model: str, stream: bool) -> None:
    t0 = time.time()
    text, tps = chat(model, "用一句話說明什麼是 API。", stream)
    s = status()
    tps_s = f"{tps:.1f} tok/s" if tps else "tok/s n/a"
    print(f"[ok] {label}: {time.time() - t0:.1f}s, {tps_s}, active={s['active']} status={s['status']}")
    print(f"     -> {text.strip()[:90]!r}")
    assert s["active"] == model and s["status"] == "ready", s


step(f"{A} non-stream", A, False)
step(f"{A} stream", A, True)
if B:
    step(f"switch to {B} (stream)", B, True)
    step(f"switch back to {A}", A, False)
print("all checks passed")
