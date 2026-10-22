"""skill: load a skill's full instructions on demand (see skills.py for discovery)."""

from __future__ import annotations

from ..errors import ToolError
from ..skills import parse_frontmatter
from .base import Tool, ToolContext

class SkillTool(Tool):
    name = "skill"
    description = ("Load the full instructions of a skill listed under 'Skills' in the system prompt. "
                   "Call it before starting a task that matches a skill's description, then follow it.")
    parameters = {"type": "object", "properties": {
        "name": {"type": "string", "description": "Skill name, exactly as listed."},
    }, "required": ["name"]}

    def summary(self, args: dict) -> str:
        return f"skill {args.get('name', '')}"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        skill = ctx.skills.get(args["name"])
        if skill is None:
            raise ToolError("not_found", f"no skill named '{args['name']}'. Available: {', '.join(ctx.skills) or 'none'}")
        _, body = parse_frontmatter(skill.path.read_text(encoding="utf-8", errors="replace"))
        return (f"Skill '{skill.name}' (files referenced below are relative to {skill.path.parent}):\n\n{body}")
