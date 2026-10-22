# Configuration and tuning

## `.env`: values that belong to this machine

Model folders, binaries, the project folder, the Kubernetes context and the API key live in `.env`
(not committed; `.env.example` lists every variable). `config.json` refers to them as `${NAME}` or
`${NAME:-default}`, e.g. `"server_binary": "${LLAMA_SERVER}"`. Real environment variables win over
`.env`; a missing variable without a default stops the gateway with a message naming it.

## `config.json`

Not committed either; `config.example.json` is the template.

| Key | Meaning |
|---|---|
| `server.host` / `server.port` | Bind address. `127.0.0.1` = this machine only; `0.0.0.0` = reachable from your LAN / Tailscale (e.g. a phone). |
| `server.auto_compact` | Summarize older messages when a conversation outgrows the context (default `true`). |
| `server.api_key` | If set, every `/api` and `/v1` call needs `Authorization: Bearer <key>` (or `x-api-key`). Set one before exposing the gateway beyond your own machine. |
| `ollama.url` | Ollama server. |
| `ollama.pinned` | Ollama models that are never unloaded (e.g. an embedding model another service uses). |
| `llamacpp.server_binary` / `port` / `startup_timeout_s` / `log_dir` | The managed `llama-server`: binary, local port, load timeout, log folder. |
| `agent.sandbox` | `local`: agents run in the gateway and act on this computer. `k8s`: each agent runs in its own Kubernetes pod and only sees its own folder ([details](agent.md#pods-agentsandbox-k8s)). |
| `agent.k8s.*` | `agents_dir` (host folder; agent `<id>` gets `<agents_dir>/<id>/` at `/agent`), `namespace`, `image`, `kube_context`, `cpu` / `memory` per pod, `gateway_url`, `host_mount_prefix`. |
| `agent.workspaces` / `agent.default_workspace` | Local sandbox: folders the web agent may use, and the empty folder new tasks start in. |
| `agent.data_dir` | Saved sessions, user-wide `AGENTS.md` and `skills/` (default `data`). |
| `agent.default_mode` | `plan`, `default`, `accept_edits` or `bypass` ([safety model](agent.md#6-safety-model)). |
| `agent.max_steps` | Model calls per agent turn before it stops and asks you to continue. |
| `agent.max_running` | Agent tasks running at the same time (default 1). |
| `agent.hooks`, `agent.mcp_servers` | Shell hooks and MCP servers for the agent ([examples](agent.md#7-extending-it)). |

Each entry in `profiles`:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Used as `model` in API calls. |
| `name` | yes | Display name. |
| `backend` | yes | `ollama` or `llamacpp`. |
| `context` | yes | Context window to load the model with (tokens). |
| `model` | ollama | Ollama tag, e.g. `qwen3:8b`. |
| `model_path` | llamacpp | GGUF file. |
| `args` | no | Extra `llama-server` flags, e.g. `["-ngl","99","-ncmoe","12"]`. |
| `tools` | no | `true`: offered in the Agent menu (see the README for how models are chosen). |
| `placement` | no | `gpu` or `gpu+cpu`, shown as a tag. |
| `max_context`, `type`, `size_gb`, `notes`, `quant`, `bits`, `params_b` | no | Shown in the UI. `bits` = file size × 8 ÷ parameters (includes unquantized parts, so above the nominal bits). |
| `speed_tps`, `speed_zh_tps` | no | Measured tok/s on an English coding and a Chinese chat prompt (`scripts/measure.py --write config.json`). |

## Where the models come from

- Ollama profiles use the Ollama tag shown in the UI.
- llama.cpp profiles load GGUF files directly. Most are the blobs of Ollama tags (`gemma4:26b-a4b-it-qat`, `gpt-oss:20b`, `qwen3.6:35b-a3b`, `qwen3:14b`, `qwen3:0.6b` as a draft model). Muse is an Unsloth UD-IQ2_XXS GGUF.
- `gemma4-12b-qat-mtp` is a local Ollama model: Gemma 4 12B QAT weights plus Gemma's MTP draft model.

## What changes the speed

- **Free GPU memory.** Desktop apps use it too, and a pinned embedding model takes ~0.6 GB. A model that no longer fits spills into shared memory and gets 5–10× slower without any error.
- **Free system RAM**, for GPU+CPU models: `qwen3.6-35b-moe` keeps ~14 GB of experts in RAM.
- **Context size.** More context needs more GPU memory: MoE profiles need more expert layers on the CPU (`-ncmoe`), dense models may spill.
- **Language.** A draft model speeds up English code but can slow Chinese down.
- **Thinking.** Reasoning tokens come before the answer; with a low `max_tokens` a thinking model can return nothing.
- **Disk.** Models on a HDD load in seconds when still in the OS file cache, minutes when not.

Test machine: RTX 3060 12 GB (driver 591.59), Core i5-11400, 32 GB RAM, models on a HDD, Windows 11,
Ollama 0.34.2, llama.cpp build 11756 (CUDA).

## Troubleshooting

- **A model is suddenly several times slower.** Something else holds GPU memory and the model spilled. Check `nvidia-smi` for GPU-heavy apps or leftover `llama-server` processes (the gateway refuses to start one while an old one holds its port).
- **Empty answer.** A thinking model spent `max_tokens` on reasoning: raise it or send `"think": false`.
- **Ollama fails to load a model with a draft (MTP).** Not enough GPU memory for the draft: lower the profile's `context`.
- **Agent pods: `git clone` times out.** The mounted Windows folder is slow for many small files; the agent is told to clone big repositories into `/tmp` ([details](agent.md#pods-agentsandbox-k8s)).
