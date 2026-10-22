"""File tools: read_file, write_file, edit_file, glob, grep.

Same rules as Claude Code's file tools, because they prevent the classic agent mistakes:
  - read_file numbers its lines, so the model can point at code precisely.
  - edit_file replaces an exact string that must be unique; the model cannot "edit line 42"
    of a file that changed under it, and an ambiguous edit fails instead of hitting the wrong spot.
  - write_file and edit_file refuse to touch an existing file the model has not read in this
    session, so it never overwrites content it has not seen.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path

from ..errors import ToolError
from .base import Risk, Tool, ToolContext, truncate

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
MAX_READ_BYTES = 2_000_000


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ToolError("not_found", f"{path.name} does not exist.")
    if path.is_dir():
        raise ToolError("is_directory", f"{path.name} is a directory; use glob to list it.")
    if path.stat().st_size > MAX_READ_BYTES:
        raise ToolError("too_large", f"{path.name} is larger than {MAX_READ_BYTES // 1_000_000} MB; use grep instead.")
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise ToolError("binary", f"{path.name} looks like a binary file.")
    return data.decode("utf-8", errors="replace")


class ReadFile(Tool):
    name = "read_file"
    description = ("Read a text file. Returns numbered lines (`  12\\tcode`). For large files pass offset/limit "
                   "to read a part. Read a file before editing it.")
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path, relative to the workspace."},
        "offset": {"type": "integer", "description": "First line to return (1-based). Default 1."},
        "limit": {"type": "integer", "description": "Number of lines to return. Default 400."},
    }, "required": ["path"]}

    def summary(self, args: dict) -> str:
        return f"read_file {args.get('path', '')}" + (f" (from line {args['offset']})" if args.get("offset") else "")

    async def run(self, args: dict, ctx: ToolContext) -> str:
        path = ctx.resolve(args["path"])
        lines = _read_text(path).splitlines()
        start = max(1, args.get("offset") or 1)
        limit = max(1, args.get("limit") or 400)
        ctx.read_files.add(path)
        if not lines:
            return f"{ctx.rel(path)} is empty."
        chunk = lines[start - 1:start - 1 + limit]
        body = "\n".join(f"{i:>5}\t{line}" for i, line in enumerate(chunk, start))
        end = start + len(chunk) - 1
        more = f"\n[lines {start}-{end} of {len(lines)}; pass offset={end + 1} to continue]" if end < len(lines) else ""
        return body + more


class WriteFile(Tool):
    name = "write_file"
    description = ("Create a file or replace its whole content. Prefer edit_file for changes to an existing file. "
                   "An existing file must be read first.")
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path, relative to the workspace."},
        "content": {"type": "string", "description": "The complete new content."},
    }, "required": ["path", "content"]}
    risk = Risk.WRITE

    def summary(self, args: dict) -> str:
        return f"write_file {args.get('path', '')} ({len(args.get('content', ''))} chars)"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        path = ctx.resolve(args["path"])
        existed = path.exists()
        if existed and path not in ctx.read_files:
            raise ToolError("not_read", f"{ctx.rel(path)} exists. Read it with read_file before overwriting it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8", newline="")
        ctx.read_files.add(path)
        return f"{'Updated' if existed else 'Created'} {ctx.rel(path)} ({len(args['content'].splitlines())} lines)."


class EditFile(Tool):
    name = "edit_file"
    description = ("Replace an exact string in a file. old_string must match the file exactly (including "
                   "indentation) and be unique, unless replace_all is true. Include enough surrounding lines "
                   "to make it unique. Do not include the line-number prefix from read_file.")
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path, relative to the workspace."},
        "old_string": {"type": "string", "description": "Exact text to replace."},
        "new_string": {"type": "string", "description": "Replacement text."},
        "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
    }, "required": ["path", "old_string", "new_string"]}
    risk = Risk.WRITE

    def summary(self, args: dict) -> str:
        return f"edit_file {args.get('path', '')}"

    async def run(self, args: dict, ctx: ToolContext) -> str:
        path = ctx.resolve(args["path"])
        if path not in ctx.read_files:
            raise ToolError("not_read", f"Read {ctx.rel(path)} with read_file before editing it.")
        text = _read_text(path)
        old, new = args["old_string"], args["new_string"]
        if old == new:
            raise ToolError("invalid_args", "old_string and new_string are the same.")
        count = text.count(old)
        if count == 0:
            hint = " (it may still contain the line-number prefix from read_file)" if re.match(r"^\s*\d+\t", old) else ""
            raise ToolError("no_match", f"old_string was not found in {ctx.rel(path)}{hint}. Read the file again "
                                        "and copy the text exactly.")
        if count > 1 and not args.get("replace_all"):
            raise ToolError("not_unique", f"old_string occurs {count} times in {ctx.rel(path)}. Add surrounding "
                                          "lines to make it unique, or set replace_all.")
        path.write_text(text.replace(old, new), encoding="utf-8", newline="")
        line = text[:text.index(old)].count("\n") + 1
        return f"Edited {ctx.rel(path)}: replaced {count if args.get('replace_all') else 1} occurrence(s) starting at line {line}."


def _empty_dir_note(root: Path, ctx: ToolContext) -> str | None:
    """A missing folder is an error, and an empty one is stated plainly (not as a failure).
    Otherwise "no matches" in an empty workspace makes small models search on and on for files
    that do not exist."""
    if not root.exists():
        raise ToolError("not_found", f"{ctx.rel(root)} does not exist.")
    if not root.is_dir():
        raise ToolError("not_a_directory", f"{ctx.rel(root)} is a file; use read_file.")
    if not any(root.iterdir()):
        return f"{ctx.rel(root)} is empty: there are no files yet. Create files with write_file if the task needs them."
    return None


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            yield Path(dirpath) / f


class Glob(Tool):
    name = "glob"
    description = ("Find files by name pattern, e.g. '**/*.py' or 'src/*.ts'. Returns paths relative to the "
                   "workspace, most recently modified first. Skips .git, node_modules and virtualenvs.")
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Glob pattern relative to `path`."},
        "path": {"type": "string", "description": "Directory to search. Default: the workspace."},
    }, "required": ["pattern"]}

    def summary(self, args: dict) -> str:
        return f"glob {args.get('pattern', '')}" + (f" in {args['path']}" if args.get("path") else "")

    async def run(self, args: dict, ctx: ToolContext) -> str:
        root = ctx.resolve(args.get("path") or ".")
        pattern = args["pattern"]
        if note := _empty_dir_note(root, ctx):
            return note

        def find() -> list[Path]:
            hits = []
            for p in _walk(root):
                rel = p.relative_to(root).as_posix()
                # "**/x" should also match "x" at the top level, as in most shells
                if fnmatch.fnmatch(rel, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])):
                    hits.append(p)
            return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)

        hits = await asyncio.to_thread(find)
        if not hits:
            return f"No files match {pattern} in {ctx.rel(root)}."
        shown = hits[:200]
        more = f"\n[{len(hits) - 200} more not shown; narrow the pattern]" if len(hits) > 200 else ""
        return "\n".join(ctx.rel(p) for p in shown) + more


class Grep(Tool):
    name = "grep"
    description = ("Search file contents with a regular expression (Python syntax). Returns `path:line: text`. "
                   "Use `glob` to limit the files, e.g. '*.py'.")
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Regular expression."},
        "path": {"type": "string", "description": "File or directory to search. Default: the workspace."},
        "glob": {"type": "string", "description": "Only search files whose name matches, e.g. '*.py'."},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive. Default false."},
    }, "required": ["pattern"]}

    def summary(self, args: dict) -> str:
        return f"grep /{args.get('pattern', '')}/" + (f" in {args['path']}" if args.get("path") else "") + \
            (f" ({args['glob']})" if args.get("glob") else "")

    async def run(self, args: dict, ctx: ToolContext) -> str:
        try:
            rx = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as e:
            raise ToolError("invalid_args", f"invalid regular expression: {e}") from e
        root = ctx.resolve(args.get("path") or ".")
        name_glob = args.get("glob")
        if not root.is_file() and (note := _empty_dir_note(root, ctx)):
            return note

        def search() -> list[str]:
            out = []
            files = [root] if root.is_file() else _walk(root)
            for p in files:
                if name_glob and not fnmatch.fnmatch(p.name, name_glob):
                    continue
                try:
                    if p.stat().st_size > MAX_READ_BYTES:
                        continue
                    data = p.read_bytes()
                except OSError:
                    continue
                if b"\0" in data[:8192]:
                    continue
                for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{ctx.rel(p)}:{i}: {line.strip()[:300]}")
                        if len(out) >= 300:
                            return out
            return out

        hits = await asyncio.to_thread(search)
        if not hits:
            return f"No matches for /{args['pattern']}/."
        text = "\n".join(hits) + ("\n[stopped at 300 matches; narrow the search]" if len(hits) >= 300 else "")
        return truncate(text, ctx.max_output_chars)
