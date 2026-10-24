# AGENTS.md — setup and developer reference for llm-gateway

Canonical reference for AI coding agents (Claude Code, Antigravity, Codex, etc.)
and human contributors working on `llm-gateway`.

## 1. Project Overview

`llm-gateway` runs local LLM models behind an OpenAI/Anthropic-compatible API with a web UI,
a built-in coding agent harness, and scheduler support for single consumer GPUs (e.g. RTX 3060 12GB).

Key capabilities:
- **FastAPI gateway** (`:8790`): OpenAI Chat (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`), OpenAI Responses (`/v1/responses`).
- **Scheduler (ModelManager)**: Queues concurrent calls and ensures only one heavy model is loaded at a time.
- **Agent Pod Host**: Integrates with local Kubernetes pods for sandboxed web agent execution.
- **Terminal Agent**: `llm-agent` CLI loop with tools and session compaction.

## 2. Environment & Setup

Prerequisites:
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama / llama.cpp (with CUDA build if running local GPU models)

Setup commands:
```bash
cp .env.example .env                 # Machine-specific paths and binaries
cp config.example.json config.json   # Model profile configurations
uv sync --group dev                  # Install dependencies and dev tools
```

## 3. Running & Testing

```bash
# Run the gateway server
uv run llm-gateway

# Run the terminal coding agent
uv run llm-agent

# Run test suite
uv run --group dev pytest

# Measure model speeds
uv run python scripts/measure.py --write config.json [ids]
```

## 4. Code Structure

- `llm_gateway/`: Core application package
  - `__main__.py`: Gateway entrypoint
  - `server.py`: FastAPI routes (`/v1/*`, `/api/*`)
  - `models/`: ModelManager, profile loading, and process lifecycle
  - `agent/`: Coding agent loop, tools, session management, compaction
- `deploy/`: Agent Dockerfile and Kubernetes manifests
- `docs/`: In-depth architecture, configuration, and agent evaluation guides
- `scripts/`: Evaluation benchmarks and smoke tests
- `tests/`: Automated unit and integration tests

## 5. AI Agent Git Commit Attribution

When contributing or committing changes to this repository, AI coding agents (Claude Code, Antigravity, etc.) must append the corresponding official `Co-authored-by` trailer at the bottom of the commit message (preceded by an empty line) so that contributions and avatars render properly on GitHub:

- **Claude Code**:
  ```text
  Co-authored-by: Claude <noreply@anthropic.com>
  ```
- **Antigravity**:
  ```text
  Co-authored-by: Antigravity <347312419+antigravity-code@users.noreply.github.com>
  ```

Rules:
1. Ensure there is a blank line separating the commit message body from the trailers.
2. Maintain standard Git trailer format: `Co-authored-by: Name <email>`.
