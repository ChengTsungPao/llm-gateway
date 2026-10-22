"""Evaluate models as agents: real harness, real model (through the running gateway), fresh folder.

    uv run --group dev python scripts/agent_eval.py --models qwen3-8b-agent gemma4-12b-agent --runs 3

Six tasks, each with an automatic check. For every run it records whether the task succeeded,
how many model calls it took, invalid tool calls (broken JSON, unknown tools, wrong arguments),
whether the loop detector had to stop it, and the time. Results go to a JSON-lines file and a
summary table. Runs use bypass mode in a temporary folder: the agent's commands run on this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_gateway.agent.context import ContextManager  # noqa: E402
from llm_gateway.agent.events import ToolResult, TurnEnd  # noqa: E402
from llm_gateway.agent.llm import LLMClient  # noqa: E402
from llm_gateway.agent.loop import Agent, AgentSettings  # noqa: E402
from llm_gateway.agent.permissions import DenyAll  # noqa: E402
from llm_gateway.agent.session import Session  # noqa: E402
from llm_gateway.agent.tools import ToolRegistry, builtin_tools  # noqa: E402
from llm_gateway.agent.tools.shell import detect_shell  # noqa: E402

INVALID = {"bad_json", "unknown_tool", "invalid_args"}


def pytest_passes(d: Path) -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=d,
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def py(d: Path, code: str) -> str:
    r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


STATS = '''def average(numbers):
    """Return the mean of a list of numbers; 0 for an empty list."""
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers) - 1


def median(numbers):
    s = sorted(numbers)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
'''
STATS_TEST = '''from stats import average, median


def test_average():
    assert average([1, 2, 3]) == 2
    assert average([]) == 0


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 2, 3]) == 2.5
'''

TASKS = {
    "lookup": {
        "files": {"settings.toml": "[server]\nhost = \"0.0.0.0\"\nport = 8421\nworkers = 4\n\n[db]\nport = 5432\n",
                  "app.py": "import tomllib\n\nwith open('settings.toml', 'rb') as f:\n    cfg = tomllib.load(f)\n"
                            "print('serving on', cfg['server']['port'])\n",
                  "README.md": "# Demo service\nSee settings.toml for configuration.\n"},
        "prompt": "What port does the server (not the database) use according to this project's configuration? "
                  "Reply with just the number.",
        "check": lambda d, answer: "8421" in answer,
    },
    "fix_bug": {
        "files": {"stats.py": STATS, "test_stats.py": STATS_TEST},
        "prompt": "The tests in this folder fail. Find the bug in stats.py, fix it, and run `python -m pytest` "
                  "to confirm they pass.",
        "check": lambda d, answer: pytest_passes(d),
    },
    "build": {
        "files": {},
        "prompt": "Create a Python module greet.py with a function hello(name) that returns 'Hello, <name>!' "
                  "(for example hello('Ann') returns 'Hello, Ann!'). Add test_greet.py with a pytest test for it, "
                  "and run `python -m pytest` to check it passes.",
        "check": lambda d, answer: (py(d, "import greet; print(greet.hello('Ann'))") == "Hello, Ann!"
                                    and (d / "test_greet.py").exists() and pytest_passes(d)),
    },
    "search": {
        "files": {"mailer.py": "def send_email(to, subject, body):\n    print('mail to', to)\n",
                  "alerts.py": "from mailer import send_email\n\n\ndef notify_user(user, msg):\n"
                               "    send_email(user.email, 'Alert', msg)\n",
                  "reports.py": "def build_report(rows):\n    return '\\n'.join(map(str, rows))\n",
                  "utils.py": "def slug(s):\n    return s.lower().replace(' ', '-')\n"},
        "prompt": "In the Python files in this folder, which function calls send_email()? Answer with the file "
                  "name and the function name.",
        "check": lambda d, answer: "notify_user" in answer and "alerts" in answer,
    },
    "recover": {
        "files": {"app/main.py": 'GREETING = "Hi"\n\n\ndef greet(name):\n    return f"{GREETING}, {name}"\n',
                  "app/__init__.py": ""},
        # the path in the request is wrong on purpose: the agent has to look around and recover
        "prompt": "In src/app.py change the greeting from 'Hi' to 'Hello'.",
        "check": lambda d, answer: 'GREETING = "Hello"' in (d / "app/main.py").read_text(encoding="utf-8"),
    },
    "rename": {
        "files": {"shop/__init__.py": "",
                  "shop/cart.py": "def calc_total(items):\n    return sum(price * qty for price, qty in items)\n",
                  "shop/checkout.py": "from .cart import calc_total\n\n\ndef checkout(items, tax=0.1):\n"
                                      "    return round(calc_total(items) * (1 + tax), 2)\n",
                  "tests/test_shop.py": "from shop.cart import calc_total\nfrom shop.checkout import checkout\n\n\n"
                                        "def test_total():\n    assert calc_total([(2, 3)]) == 6\n\n\n"
                                        "def test_checkout():\n    assert checkout([(10, 1)]) == 11.0\n"},
        "prompt": "Rename the function calc_total to compute_total everywhere in this project (definition, imports "
                  "and calls, tests included), then run `python -m pytest` to check.",
        "check": lambda d, answer: (not any("calc_total" in p.read_text(encoding="utf-8") for p in d.rglob("*.py"))
                                    and pytest_passes(d)),
    },
}


async def run_once(gateway: str, model: str, context: int, task: str, root: Path, max_steps: int,
                   timeout_s: float) -> dict:
    spec = TASKS[task]
    d = Path(tempfile.mkdtemp(prefix=f"{task}-", dir=root))
    for rel, text in spec["files"].items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")
    shell = detect_shell()
    session = Session(workspace=str(d), model=model, mode="bypass")
    agent = Agent(session=session, llm=LLMClient(gateway, model), tools=ToolRegistry(builtin_tools(shell)),
                  context=ContextManager(context), approver=DenyAll(), shell=shell,
                  settings=AgentSettings(max_steps=max_steps, max_output_chars=8000))
    invalid = steps = 0
    reason = "timeout"
    t0 = time.time()

    async def go():
        nonlocal invalid, steps, reason
        async for ev in agent.run(spec["prompt"]):
            if isinstance(ev, ToolResult) and ev.error_code in INVALID:
                invalid += 1
            if isinstance(ev, TurnEnd):
                steps, reason = ev.steps, ev.reason
    try:
        await asyncio.wait_for(go(), timeout_s)
    except asyncio.TimeoutError:
        pass
    answer = next((m.get("content") or "" for m in reversed(session.messages) if m["role"] == "assistant"), "")
    try:
        ok = bool(spec["check"](d, answer))
    except Exception:
        ok = False
    shutil.rmtree(d, ignore_errors=True)
    return {"model": model, "task": task, "ok": ok, "steps": steps, "invalid_calls": invalid,
            "loop_stop": reason == "loop", "reason": reason, "seconds": round(time.time() - t0, 1),
            "answer": answer[:300]}


def summarize(rows: list[dict]) -> str:
    models = list(dict.fromkeys(r["model"] for r in rows))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    head = "| Model | " + " | ".join(tasks) + " | Success | Avg steps | Invalid calls | Loop stops | Avg time |"
    out = [head, "|" + "---|" * (len(tasks) + 6)]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        cells = []
        for t in tasks:
            tr = [r for r in rs if r["task"] == t]
            cells.append(f"{sum(r['ok'] for r in tr)}/{len(tr)}")
        ok = sum(r["ok"] for r in rs)
        out.append(f"| {m} | " + " | ".join(cells) +
                   f" | **{100 * ok / len(rs):.0f}%** | {sum(r['steps'] for r in rs) / len(rs):.1f} | "
                   f"{sum(r['invalid_calls'] for r in rs)} | {sum(r['loop_stop'] for r in rs)} | "
                   f"{sum(r['seconds'] for r in rs) / len(rs):.0f}s |")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gateway", default="http://127.0.0.1:8790/v1")
    ap.add_argument("--workdir", default=None, help="where the temporary task folders go")
    ap.add_argument("--out", default="agent_eval.jsonl")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    contexts = {m["id"]: m.get("context_length", 8192)
                for m in httpx.get(f"{args.gateway}/models", timeout=10).json()["data"]}
    root = Path(args.workdir or tempfile.gettempdir()) / "agent-eval"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "a", encoding="utf-8") as f:
        for model in args.models:                 # grouped by model: one load per model
            for task in args.tasks:
                for i in range(args.runs):
                    r = await run_once(args.gateway, model, contexts.get(model, 8192), task, root,
                                       args.max_steps, args.timeout)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{model:24} {task:9} run {i + 1}: {'OK  ' if r['ok'] else 'FAIL'} "
                          f"{r['steps']:>2} steps {r['seconds']:>6.1f}s {r['reason']}", flush=True)
    print("\n" + summarize(rows))


if __name__ == "__main__":
    asyncio.run(main())
