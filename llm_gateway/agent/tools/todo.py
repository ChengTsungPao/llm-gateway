"""todo_write: the model's own task list ("Planner" in the agent diagram).

There is no separate planning module. Planning is just the model writing a list with a tool, and
the list being shown back to it on every later call through the tool results. It costs nothing,
keeps long tasks on track, and the UI shows the user what the agent intends to do.
"""

from __future__ import annotations

from ..errors import ToolError
from .base import Tool, ToolContext

STATUSES = ("pending", "in_progress", "completed")


class TodoWrite(Tool):
    name = "todo_write"
    description = ("Create or update your task list for multi-step work. Send the whole list each time. "
                   "Mark exactly one item in_progress while you work on it, and mark items completed as soon "
                   "as they are done. Skip this for simple one-step requests.")
    parameters = {"type": "object", "properties": {
        "todos": {"type": "array", "description": "The full list: [{\"content\": str, \"status\": "
                                                  "\"pending\" | \"in_progress\" | \"completed\"}]",
                  "items": {"type": "object"}},
    }, "required": ["todos"]}

    def summary(self, args: dict) -> str:
        return f"todo_write ({len(args.get('todos', []))} items)"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        todos = []
        for i, item in enumerate(args["todos"], 1):
            if not isinstance(item, dict) or not str(item.get("content", "")).strip():
                raise ToolError("invalid_args", f"item {i} needs a non-empty 'content'")
            status = item.get("status", "pending")
            if status not in STATUSES:
                raise ToolError("invalid_args", f"item {i}: status must be one of {STATUSES}")
            todos.append({"content": str(item["content"]).strip(), "status": status})
        ctx.todos[:] = todos
        done = sum(t["status"] == "completed" for t in todos)
        lines = "\n".join(f"[{'x' if t['status'] == 'completed' else '>' if t['status'] == 'in_progress' else ' '}] "
                          f"{t['content']}" for t in todos)
        return f"Task list updated ({done}/{len(todos)} done):\n{lines}"
