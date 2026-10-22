# Agent model evaluation

How the models in the Agent menu were chosen. Script: `scripts/agent_eval.py`.

Each candidate ran the real agent harness with its 32K profile, on **6 tasks × 3 runs**, each checked
automatically (`scripts/agent_eval.py`):

| Task | What it checks |
|---|---|
| lookup | Find a value in a config file and answer it |
| fix_bug | Fix a bug so `pytest` passes |
| build | Create a module and its test, run the tests |
| search | Find which function calls another one |
| recover | Edit a file given a wrong path: look around and recover |
| rename | Rename a function in all files, tests pass |

A model is offered in the Agent menu when it succeeds in **at least 60%** of the runs:

| Model (32K) | lookup | fix_bug | build | search | recover | rename | Success | Invalid tool calls |
|---|---|---|---|---|---|---|---:|---:|
| Gemma 4 12B QAT | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | **78%** | 0 |
| Qwen3.5 9B | 2/3 | 3/3 | 1/3 | 3/3 | 1/3 | 3/3 | **72%** | 0 |
| Qwen3 8B | 0/3 | 2/3 | 3/3 | 0/3 | 3/3 | 3/3 | **61%** | 0 |
| Llama 3.1 8B | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 | 0/3 | 6% | 32 |
| Mistral Nemo 12B | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 | 0/3 | 6% | 0 |

Even the best small local model fails some tasks (e.g. Qwen3 8B searched only `.py` files and never
opened the config file). Expect a helper that needs checking, not Claude.


## Method details

- **Harness:** the real agent harness, the real model through the running gateway, bypass mode.
- **Workspace:** a fresh temporary folder per run.
- **Limits:** at most 20 model calls and 5 minutes per run.
- **Counted per run:**
  - success (an automatic check, e.g. `pytest` passes or the answer contains the right value);
  - model calls;
  - invalid tool calls (broken JSON, unknown tool, wrong arguments);
  - loop stops;
  - time.
- **Run on:** RTX 3060 12 GB, 32K profiles. Results vary between runs, which is why each task runs 3 times.
- **Not evaluated:** Gemma 4 26B and Qwen3.6 35B keep experts in system RAM (up to ~14 GB). The run was stopped for lack of memory. Run them with RAM free:

```bash
uv run --group dev python scripts/agent_eval.py --models gemma4-26b-agent qwen3.6-35b-agent --runs 3
```
