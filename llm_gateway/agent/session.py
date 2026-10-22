"""Session state and its checkpoint on disk.

The whole state of an agent conversation is plain data: the message list (OpenAI format), the
task list, the permission mode and what the user always allowed. The loop saves it after every
step, so a crash, a closed browser tab or Ctrl+C loses at most the step in progress, and any
session can be resumed later (`llm-agent --resume <id>`, or picked in the web UI).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Session:
    workspace: str
    model: str
    mode: str = "default"
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    title: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)      # without the system prompt (rebuilt each turn)
    todos: list[dict] = field(default_factory=list)
    always_allowed: list[str] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    summary: str = ""                                       # summary of compacted older messages

    def meta(self) -> dict:
        return {"id": self.id, "title": self.title, "workspace": self.workspace, "model": self.model,
                "mode": self.mode, "created": self.created, "updated": self.updated,
                "messages": len(self.messages)}


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if not session_id or any(c in session_id for c in "/\\.:"):
            raise KeyError(session_id)
        return self.root / f"{session_id}.json"

    def save(self, session: Session) -> None:
        session.updated = time.time()
        path = self._path(session.id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(session), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)   # atomic: a crash mid-write never leaves a half-written checkpoint

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        return Session(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[dict]:
        out = []
        for p in self.root.glob("*.json"):
            try:
                out.append(Session(**json.loads(p.read_text(encoding="utf-8"))).meta())
            except (ValueError, TypeError):
                continue
        return sorted(out, key=lambda m: m["updated"], reverse=True)

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
