"""
Splits source files into semantic chunks suitable for embedding.

Python files: AST-based — one chunk per top-level function/class plus a
              preamble chunk for imports and module-level constants.
Other files:  Line-based with overlap so context isn't lost at boundaries.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

CHUNK_LINES = 80
OVERLAP_LINES = 15
MIN_CHUNK_CHARS = 40  # skip trivially short chunks (e.g. empty __init__.py)

INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json",
    ".sh", ".html", ".css",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".chroma", ".mypy_cache", ".pytest_cache",
}


@dataclass
class Chunk:
    id: str          # "{rel_path}:{start_line}" — stable across re-indexes
    content: str
    metadata: dict   # file, start_line, end_line, symbol (optional)


def _lines_to_chunk(rel_path: str, lines: list[str], start: int, symbol: str = "") -> Chunk:
    content = "\n".join(lines).strip()
    return Chunk(
        id=f"{rel_path}:{start}",
        content=content,
        metadata={
            "file": rel_path,
            "start_line": start,
            "end_line": start + len(lines) - 1,
            "symbol": symbol,
        },
    )


def _chunk_by_lines(rel_path: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        window = lines[i : i + CHUNK_LINES]
        chunk = _lines_to_chunk(rel_path, window, i + 1)
        if len(chunk.content) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        i += CHUNK_LINES - OVERLAP_LINES
    return chunks


def _chunk_python(rel_path: str, source: str) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _chunk_by_lines(rel_path, source)

    all_lines = source.splitlines()
    chunks: list[Chunk] = []

    # Track which lines are covered by top-level defs so we can build a preamble.
    top_level_ranges: list[tuple[int, int]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno - 1      # 0-indexed
        end = node.end_lineno        # exclusive upper bound (0-indexed end+1)
        top_level_ranges.append((start, end))

        symbol_lines = all_lines[start:end]
        # For large classes/functions, further split by inner defs if needed.
        if len(symbol_lines) > CHUNK_LINES * 2:
            chunks.extend(_chunk_by_lines(rel_path, "\n".join(symbol_lines)))
        else:
            chunk = _lines_to_chunk(rel_path, symbol_lines, start + 1, symbol=node.name)
            if len(chunk.content) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)

    # Preamble: everything before the first top-level def.
    preamble_end = top_level_ranges[0][0] if top_level_ranges else len(all_lines)
    preamble_lines = all_lines[:preamble_end]
    if preamble_lines:
        chunk = _lines_to_chunk(rel_path, preamble_lines, 1, symbol="<module>")
        if len(chunk.content) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

    # Fallback: if AST found nothing useful, chunk by lines.
    return chunks if chunks else _chunk_by_lines(rel_path, source)


def chunk_file(repo_root: Path, abs_path: Path) -> list[Chunk]:
    rel_path = str(abs_path.relative_to(repo_root))
    try:
        source = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if abs_path.suffix == ".py":
        return _chunk_python(rel_path, source)
    return _chunk_by_lines(rel_path, source)


def iter_repo_files(repo_root: Path):
    """Yield all indexable files, skipping ignored dirs and binary extensions."""
    for path in repo_root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix not in INDEXABLE_EXTENSIONS:
            continue
        yield path
