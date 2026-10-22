"""A minimal MCP (Model Context Protocol) client over stdio, and the adapter that turns each
remote MCP tool into a local Tool (Adapter pattern), so the loop cannot tell them apart.

MCP over stdio is JSON-RPC 2.0, one JSON object per line, on the server process's stdin/stdout:

    -> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
    <- {"jsonrpc":"2.0","id":1,"result":{"capabilities":{...},"serverInfo":{...}}}
    -> {"jsonrpc":"2.0","method":"notifications/initialized"}            (a notification: no id)
    -> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
    -> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"...","arguments":{...}}}

That is all an agent needs to use MCP tools. Configured in config.json, like .mcp.json:

    "mcp_servers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {}}}

Tools appear as `mcp__<server>__<tool>`. They count as `exec` risk (they can do anything)
unless the server marks them read-only (annotations.readOnlyHint).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
import subprocess
import sys

from .errors import ToolError
from .tools.base import Risk, Tool, ToolContext, truncate

PROTOCOL_VERSION = "2025-06-18"


class MCPClient:
    def __init__(self, name: str, command: str, args: list[str] | None = None, env: dict | None = None):
        self.name, self.command, self.args, self.env = name, command, args or [], env or {}
        self.proc: asyncio.subprocess.Process | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self, timeout_s: float = 30) -> None:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        # shutil.which finds `npx.cmd` / `uvx.exe` on Windows, which a bare "npx" would not
        self.proc = await asyncio.create_subprocess_exec(
            shutil.which(self.command) or self.command, *self.args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env={**os.environ, **self.env}, limit=16 * 1024 * 1024, **kwargs)
        self._reader = asyncio.create_task(self._read_loop())
        await asyncio.wait_for(self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "llm-gateway-agent", "version": "0.1"}}), timeout_s)
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def _send(self, message: dict) -> None:
        if not self.proc or self.proc.returncode is not None:
            raise ToolError("mcp_down", f"MCP server '{self.name}' is not running", retryable=False)
        async with self._lock:   # one writer at a time, or two messages could interleave
            self.proc.stdin.write((json.dumps(message) + "\n").encode())
            await self.proc.stdin.drain()

    async def request(self, method: str, params: dict | None = None) -> dict:
        rid = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[rid] = future
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        return await future

    async def _read_loop(self) -> None:
        """Match responses to waiting requests by id; anything else (notifications, logs) is ignored."""
        assert self.proc
        try:
            while line := await self.proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                future = self._pending.pop(msg.get("id"), None) if "id" in msg else None
                if future is None or future.done():
                    continue
                if "error" in msg:
                    err = msg["error"]
                    future.set_exception(ToolError("mcp_error", f"{err.get('message')} (code {err.get('code')})"))
                else:
                    future.set_result(msg.get("result", {}))
        finally:   # server exited: fail everyone still waiting instead of hanging forever
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ToolError("mcp_down", f"MCP server '{self.name}' exited"))
            self._pending.clear()

    async def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += result.get("tools", [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict, timeout_s: float = 120) -> dict:
        try:
            return await asyncio.wait_for(self.request("tools/call", {"name": name, "arguments": arguments}), timeout_s)
        except asyncio.TimeoutError:
            raise ToolError("timeout", f"MCP tool {name} did not answer within {timeout_s:.0f}s",
                            side_effect=True) from None

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()


class MCPTool(Tool):
    """Adapter: presents one remote MCP tool through the local Tool interface."""

    def __init__(self, client: MCPClient, spec: dict):
        self.client = client
        self.remote_name = spec["name"]
        self.name = f"mcp__{client.name}__{spec['name']}"
        self.description = spec.get("description", "")[:1024]
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}
        read_only = (spec.get("annotations") or {}).get("readOnlyHint")
        self.risk = Risk.READ if read_only else Risk.EXEC

    def summary(self, args: dict) -> str:
        return f"{self.client.name}.{self.remote_name}({json.dumps(args, ensure_ascii=False)[:100]})"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        result = await self.client.call_tool(self.remote_name, args)
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item["text"])
            else:
                parts.append(f"[{item.get('type')} content not shown]")
        if result.get("structuredContent") and not parts:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        text = truncate("\n".join(parts), ctx.max_output_chars)
        if result.get("isError"):
            raise ToolError("mcp_tool_error", text or "the tool reported an error")
        return text or "(no output)"


async def connect_servers(config: dict[str, dict]) -> tuple[list[MCPClient], list[Tool], list[str]]:
    """Start every configured server; a server that fails is reported and skipped, not fatal."""
    clients, tools, errors = [], [], []
    for name, spec in config.items():
        client = MCPClient(name, spec["command"], spec.get("args"), spec.get("env"))
        try:
            await client.start()
            tools += [MCPTool(client, t) for t in await client.list_tools()]
            clients.append(client)
        except (OSError, ToolError, asyncio.TimeoutError) as e:
            errors.append(f"MCP server '{name}' failed to start: {getattr(e, 'message', None) or e!r}")
            await client.close()
    return clients, tools, errors
