"""Load and validate config.json."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

BACKENDS = ("ollama", "llamacpp")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    backend: str
    context: int
    model: str = ""          # ollama model tag
    model_path: str = ""     # llamacpp GGUF path
    args: tuple[str, ...] = ()
    max_context: int | None = None
    type: str = ""
    size_gb: float | None = None
    speed_tps: float | None = None
    speed_zh_tps: float | None = None
    quant: str = ""
    bits: float | None = None
    params_b: float | None = None
    tools: bool = False      # tool calling works well enough for the agent
    placement: str = ""     # "gpu" or "gpu+cpu" (part of the model offloaded to system RAM)
    notes: str = ""

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "backend": self.backend,
            "model": self.model or Path(self.model_path).name,
            "context": self.context,
            "max_context": self.max_context,
            "type": self.type,
            "size_gb": self.size_gb,
            "speed_tps": self.speed_tps,
            "speed_zh_tps": self.speed_zh_tps,
            "quant": self.quant,
            "bits": self.bits,
            "params_b": self.params_b,
            "placement": self.placement,
            "tools": self.tools,
            "args": list(self.args),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class K8sConfig:
    """agent.k8s: run every agent in its own Kubernetes pod (see docs/agent.md, "Pods")."""
    agents_dir: Path = Path("data/agents")    # host folder; each agent gets <agents_dir>/<id>/ mounted at /agent
    namespace: str = "llm-agents"
    image: str = "llm-gateway-agent:latest"
    kube_context: str = "docker-desktop"
    host_mount_prefix: str = "/run/desktop/mnt/host"   # how Docker Desktop's node sees Windows drives
    gateway_url: str = ""                     # how pods reach the gateway; default http://host.docker.internal:<port>/v1
    cpu: str = "2"
    memory: str = "2Gi"
    startup_timeout_s: float = 180


@dataclass(frozen=True)
class AgentConfig:
    """The `agent` section: where the agent may work and how it is set up (see docs/agent.md)."""
    sandbox: str = "local"                 # "local": tools run on this machine; "k8s": one pod per agent
    k8s: K8sConfig = field(default_factory=K8sConfig)
    workspaces: tuple[Path, ...] = ()     # local sandbox: the agent may only open workspaces inside these
    data_dir: Path = Path("data")          # sessions, user-wide AGENTS.md and skills
    default_workspace: Path = Path("data/workspace")   # an empty folder; always allowed
    scratch_dir: Path | None = None        # fast temporary folder the tools may also use (pods: /tmp)
    default_mode: str = "default"
    max_steps: int = 40
    shared_dir: Path | None = None         # read-only folder with skills/ and AGENTS.md for every agent (pods: /shared)
    max_running: int = 1                   # agent tasks running at once (agent/dispatch.py)
    shell: tuple[str, ...] = ()            # empty = detect (Git Bash on Windows, else PowerShell)
    hooks: dict = field(default_factory=dict)
    mcp_servers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    api_key: str
    ollama_url: str
    ollama_pinned: tuple[str, ...]
    llamacpp_binary: str
    llamacpp_port: int
    llamacpp_startup_timeout_s: float
    log_dir: Path
    profiles: tuple[Profile, ...] = field(default_factory=tuple)
    auto_compact: bool = True
    agent: AgentConfig = field(default_factory=AgentConfig)

    def profile(self, profile_id: str) -> Profile | None:
        return next((p for p in self.profiles if p.id == profile_id), None)


def _profile(raw: dict, index: int) -> Profile:
    where = f"profiles[{index}]"
    for key in ("id", "name", "backend", "context"):
        if key not in raw:
            raise ConfigError(f"{where}: missing '{key}'")
    if raw["backend"] not in BACKENDS:
        raise ConfigError(f"{where}: backend must be one of {BACKENDS}, got {raw['backend']!r}")
    if raw["backend"] == "ollama" and not raw.get("model"):
        raise ConfigError(f"{where}: ollama profiles need 'model' (an Ollama tag)")
    if raw["backend"] == "llamacpp" and not raw.get("model_path"):
        raise ConfigError(f"{where}: llamacpp profiles need 'model_path' (a GGUF file)")
    if not isinstance(raw["context"], int) or raw["context"] <= 0:
        raise ConfigError(f"{where}: 'context' must be a positive integer")
    return Profile(
        id=str(raw["id"]),
        name=str(raw["name"]),
        backend=raw["backend"],
        context=raw["context"],
        model=raw.get("model", ""),
        model_path=raw.get("model_path", ""),
        args=tuple(str(a) for a in raw.get("args", [])),
        max_context=raw.get("max_context"),
        type=raw.get("type", ""),
        size_gb=raw.get("size_gb"),
        speed_tps=raw.get("speed_tps"),
        speed_zh_tps=raw.get("speed_zh_tps"),
        quant=raw.get("quant", ""),
        bits=raw.get("bits"),
        params_b=raw.get("params_b"),
        placement=raw.get("placement", ""),
        tools=bool(raw.get("tools", False)),
        notes=raw.get("notes", ""),
    )


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def read_env_file(path: Path) -> dict[str, str]:
    """A minimal .env reader: KEY=value lines, # comments, optional quotes around the value."""
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def expand_vars(value, env: dict[str, str]):
    """Replace ${VAR} and ${VAR:-default} in every string of the config. Machine-specific values
    (model folders, binaries, project folders, the API key) live in .env, not in config.json."""
    if isinstance(value, dict):
        return {k: expand_vars(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_vars(v, env) for v in value]
    if not isinstance(value, str):
        return value

    def sub(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        raise ConfigError(f"${{{name}}} is used in the config but not set (add it to .env; see .env.example)")
    return _VAR.sub(sub, value)


def load_config(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"{path} not found — copy config.example.json to config.json and edit it")
    # .env next to the config file; real environment variables win over it
    env = {**read_env_file(path.parent / ".env"), **os.environ}
    raw = expand_vars(json.loads(path.read_text(encoding="utf-8")), env)
    server = raw.get("server", {})
    ollama = raw.get("ollama", {})
    llamacpp = raw.get("llamacpp", {})
    profiles = tuple(_profile(p, i) for i, p in enumerate(raw.get("profiles", [])))
    ids = [p.id for p in profiles]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ConfigError(f"duplicate profile ids: {sorted(dupes)}")
    if any(p.backend == "llamacpp" for p in profiles) and not llamacpp.get("server_binary"):
        raise ConfigError("llamacpp.server_binary is required when any profile uses the llamacpp backend")
    log_dir = Path(llamacpp.get("log_dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = path.parent / log_dir
    agent = _agent(raw.get("agent", {}), path.parent)
    return Config(
        host=server.get("host", "127.0.0.1"),
        port=int(server.get("port", 8790)),
        api_key=server.get("api_key", ""),
        ollama_url=ollama.get("url", "http://127.0.0.1:11434").rstrip("/"),
        ollama_pinned=tuple(ollama.get("pinned", [])),
        llamacpp_binary=llamacpp.get("server_binary", ""),
        llamacpp_port=int(llamacpp.get("port", 8791)),
        llamacpp_startup_timeout_s=float(llamacpp.get("startup_timeout_s", 300)),
        log_dir=log_dir,
        profiles=profiles,
        auto_compact=bool(server.get("auto_compact", True)),
        agent=agent,
    )



def _agent(raw: dict, base: Path) -> AgentConfig:
    from .agent.hooks import EVENTS
    from .agent.permissions import MODES

    def absolute(p: str) -> Path:
        q = Path(p).expanduser()
        return (q if q.is_absolute() else base / q).resolve()

    mode = raw.get("default_mode", "default")
    if mode not in MODES:
        raise ConfigError(f"agent.default_mode must be one of {MODES}")
    bad_events = set(raw.get("hooks", {})) - set(EVENTS)
    if bad_events:
        raise ConfigError(f"agent.hooks: unknown events {sorted(bad_events)}; valid: {EVENTS}")
    for name, spec in raw.get("mcp_servers", {}).items():
        if not isinstance(spec, dict) or not spec.get("command"):
            raise ConfigError(f"agent.mcp_servers.{name}: needs a 'command'")
    sandbox = raw.get("sandbox", "local")
    if sandbox not in ("local", "k8s"):
        raise ConfigError("agent.sandbox must be \"local\" or \"k8s\"")
    k8s = dict(raw.get("k8s", {}))
    unknown = set(k8s) - set(K8sConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"agent.k8s: unknown keys {sorted(unknown)}")
    data_dir = absolute(raw.get("data_dir", "data"))
    return AgentConfig(
        workspaces=tuple(absolute(p) for p in raw.get("workspaces", [])),
        data_dir=data_dir,
        default_workspace=absolute(raw["default_workspace"]) if raw.get("default_workspace") else data_dir / "workspace",
        sandbox=sandbox,
        k8s=K8sConfig(**{**k8s, "agents_dir": absolute(k8s.get("agents_dir", "data/agents"))}),
        default_mode=mode,
        max_steps=int(raw.get("max_steps", 40)),
        max_running=int(raw.get("max_running", 1)),
        shell=tuple(raw.get("shell", [])),
        hooks=raw.get("hooks", {}),
        mcp_servers=raw.get("mcp_servers", {}),
    )
