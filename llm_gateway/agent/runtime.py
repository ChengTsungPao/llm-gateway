"""Assembles agents (Factory). The CLI and the web service both build agents here, so they get
exactly the same tools, skills, memory and MCP servers.

Process-wide pieces live here (shell, session store, MCP connections); per-session pieces
(skills and memory of the workspace, the model client, the context budget) are created in `agent()`.
"""

from __future__ import annotations

from pathlib import Path

from ..config import AgentConfig
from .context import ContextManager, read_memory
from .hooks import Hooks
from .llm import LLMClient
from .loop import Agent, AgentSettings
from .mcp import MCPClient, connect_servers
from .permissions import Approver
from .session import Session, SessionStore
from .skills import discover_skills
from .tools import ToolRegistry, builtin_tools
from .tools.base import Tool
from .tools.shell import detect_shell


class AgentRuntime:
    def __init__(self, cfg: AgentConfig, gateway_url: str, api_key: str = ""):
        self.cfg = cfg
        self.gateway_url = gateway_url
        self.api_key = api_key
        self.shell = list(cfg.shell) or detect_shell()
        self.store = SessionStore(cfg.data_dir / "sessions")
        cfg.default_workspace.mkdir(parents=True, exist_ok=True)   # new tasks start in this empty folder
        self.mcp_clients: list[MCPClient] = []
        self.mcp_tools: list[Tool] = []
        self.startup_errors: list[str] = []

    async def start(self) -> None:
        if self.cfg.mcp_servers:
            self.mcp_clients, self.mcp_tools, self.startup_errors = await connect_servers(self.cfg.mcp_servers)

    async def close(self) -> None:
        for client in self.mcp_clients:
            await client.close()

    def workspace_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys((self.cfg.default_workspace, *self.cfg.workspaces)))

    def allowed_workspace(self, path: str) -> Path:
        """Web sessions may only use folders inside the allowed roots (the CLI skips this check:
        whoever runs it in a terminal can already do anything in that terminal)."""
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise ValueError(f"{p} is not a folder")
        roots = self.workspace_roots()
        if not any(p == root or root in p.parents for root in roots):
            raise ValueError(f"{p} is not inside an allowed workspace folder: {', '.join(str(r) for r in roots)}")
        return p

    def new_session(self, workspace: Path, model: str, mode: str | None = None) -> Session:
        session = Session(workspace=str(workspace), model=model, mode=mode or self.cfg.default_mode)
        self.store.save(session)
        return session

    def tools(self) -> ToolRegistry:
        return ToolRegistry([*builtin_tools(self.shell), *self.mcp_tools])

    def agent(self, session: Session, approver: Approver, context_tokens: int) -> Agent:
        workspace = Path(session.workspace)
        user_dir = self.cfg.data_dir
        return Agent(
            session=session,
            llm=LLMClient(self.gateway_url, session.model, self.api_key),
            tools=self.tools(),
            context=ContextManager(context_tokens),
            approver=approver,
            store=self.store,
            hooks=Hooks(self.cfg.hooks, self.shell, workspace) if self.cfg.hooks else None,
            skills=discover_skills(workspace, [user_dir / "skills"]),
            memory=read_memory(workspace, user_dir),
            shell=self.shell,
            settings=AgentSettings(max_steps=self.cfg.max_steps,
                                   max_output_chars=max(4000, min(30_000, int(context_tokens * 0.25 * 3.5)))),
        )
