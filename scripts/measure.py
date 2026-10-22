"""Measure generation speed of profiles through a running gateway, in two languages.

usage: uv run python scripts/measure.py [--base URL] [--write config.json] [profile ids...]
Prints one line per profile. With --write, stores the results as `speed_tps` (English
coding prompt) and `speed_zh_tps` (Chinese chat prompt) in the config file.

Speeds depend heavily on the prompt: speculative decoding (draft models) can be much
faster on code and slower on Chinese, which is why both are measured.
"""

import argparse
import json
from pathlib import Path

import httpx

PROMPTS = {
    "speed_tps": ("Write a Python function that parses a CSV file of orders (id, customer, amount, date), "
                  "returns total amount per customer sorted descending, and include a short docstring and two unit tests."),
    "speed_zh_tps": "請用繁體中文說明資料庫索引的用途、優缺點,以及什麼時候不該建立索引,大約三段。",
}


def run(client: httpx.Client, model: str, prompt: str) -> float | None:
    body = {"model": model, "think": False, "max_tokens": 300, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    t = r.json().get("timings") or {}
    return round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8790")
    ap.add_argument("--write", help="config.json to update with the measured speeds")
    ap.add_argument("ids", nargs="*", help="profile ids (default: all)")
    args = ap.parse_args()
    client = httpx.Client(base_url=args.base, timeout=1200)
    ids = args.ids or [m["id"] for m in client.get("/v1/models").json()["data"]]
    results = {}
    for pid in ids:
        try:
            run(client, pid, "hi")  # load + warm up
            results[pid] = {k: run(client, pid, p) for k, p in PROMPTS.items()}
            print(f"{pid}: en/code {results[pid]['speed_tps']} tok/s | zh {results[pid]['speed_zh_tps']} tok/s", flush=True)
        except httpx.HTTPError as e:
            print(f"{pid}: failed: {e}", flush=True)
            continue
        if args.write:  # write after every profile so an interrupted run keeps what it measured
            path = Path(args.write)
            cfg = json.loads(path.read_text(encoding="utf-8"))
            for p in cfg["profiles"]:
                if p["id"] == pid:
                    p.update({k: v for k, v in results[pid].items() if v is not None})
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
