"""Load and validate config.json."""

from __future__ import annotations

import json
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
            "args": list(self.args),
            "notes": self.notes,
        }


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
        notes=raw.get("notes", ""),
    )


def load_config(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"{path} not found — copy config.example.json to config.json and edit it")
    raw = json.loads(path.read_text(encoding="utf-8"))
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
    )
