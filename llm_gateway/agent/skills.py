"""Skills: know-how loaded on demand (progressive disclosure).

A skill is a folder with a SKILL.md file, the same layout Claude Code uses:

    .agents/skills/release-notes/
        SKILL.md          frontmatter (name, description) + instructions
        template.md       anything else the instructions refer to

Only each skill's name and description go into the system prompt. When a request matches a
description, the model calls the `skill` tool, and only then is the full SKILL.md loaded into the
context. Twenty skills cost twenty lines of prompt, not twenty documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILL_DIRS = (".agents/skills", ".claude/skills")   # looked up inside the workspace


@dataclass
class Skill:
    name: str
    description: str
    path: Path            # the SKILL.md file


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal `---` frontmatter parser: `key: value` lines only (enough for name/description)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = {}
    for line in text[3:end].splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[end + 4:].lstrip("\n")


def discover_skills(workspace: Path, extra_dirs: list[Path] = ()) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for base in [workspace / d for d in SKILL_DIRS] + list(extra_dirs):
        if not base.is_dir():
            continue
        for skill_md in sorted(base.glob("*/SKILL.md")):
            meta, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
            name = meta.get("name") or skill_md.parent.name
            if name in skills:   # workspace skills win over user-wide ones with the same name
                continue
            skills[name] = Skill(name, meta.get("description", ""), skill_md)
    return skills
