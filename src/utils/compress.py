"""
Context compression — strips noise before content reaches the LLM.

Each function targets a different content type:
  strip_python     — source code     (~20-35% token reduction)
  strip_ansi       — terminal output (~5-10%)
  compress_diff    — unified diffs   (~15-25%)
  strip_pytest_noise — test output   (~50-70%)
  markitdown_file  — binary/HTML docs → clean Markdown
  token_budget     — hard token cap (char-estimate, model-agnostic)
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

# One token ≈ 3.5 chars for code (denser than prose)
_CHARS_PER_TOKEN = 3.5


# ---------------------------------------------------------------------------
# Python source stripping
# ---------------------------------------------------------------------------

def strip_python(source: str) -> str:
    """
    Remove # comments; truncate multi-line docstrings to their first line;
    collapse 3+ blank lines to 1.  Falls back to regex if source has syntax errors.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return _regex_strip(source)

    out: list[tuple] = []
    prev_semantic = None  # last token type that carries semantic meaning

    for tok in tokens:
        tt, ts = tok.type, tok.string

        if tt == tokenize.COMMENT:
            continue  # drop # comments entirely

        if tt == tokenize.STRING and prev_semantic in (
            tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.OP, None
        ):
            # Likely a docstring — keep only the first line
            if ts.startswith(('"""', "'''", 'r"""', "r'''")):
                first_line = ts.split('\n')[0]
                if first_line != ts:
                    # Multi-line: truncate and close
                    quote = ts[:3]
                    ts = first_line.rstrip() + ' ' + quote
                    tok = tok._replace(string=ts)

        if tt not in (
            tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
        ):
            prev_semantic = tt

        out.append(tok)

    try:
        result = tokenize.untokenize(out)
    except Exception:
        return _regex_strip(source)

    return re.sub(r'\n{3,}', '\n\n', result).strip()


def _regex_strip(source: str) -> str:
    lines = [l for l in source.splitlines() if not l.strip().startswith('#')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


# ---------------------------------------------------------------------------
# ANSI / terminal noise
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHFABCDJMsu]')


def strip_ansi(text: str) -> str:
    """Remove VT100/ANSI escape sequences from terminal/docker output."""
    return _ANSI_RE.sub('', text)


# ---------------------------------------------------------------------------
# Unified diff compression
# ---------------------------------------------------------------------------

def compress_diff(diff: str, context_lines: int = 1) -> str:
    """
    Reduce unchanged-context lines in a unified diff from the standard 3 to
    `context_lines`.  Changed lines (+/-) and hunk headers (@@) are untouched.
    """
    out: list[str] = []
    streak = 0
    for line in diff.split('\n'):
        if line.startswith(('+', '-', '@', '---', '+++')):
            streak = 0
            out.append(line)
        elif line.startswith(' '):
            streak += 1
            if streak <= context_lines:
                out.append(line)
        else:
            streak = 0
            out.append(line)
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# pytest output filtering
# ---------------------------------------------------------------------------

def strip_pytest_noise(output: str) -> str:
    """
    Extract only the FAILURES block and short-summary from pytest output.
    Discards: platform header, collection dots, passed-test lines.
    """
    clean = strip_ansi(output)
    lines = clean.split('\n')

    in_failures = False
    in_summary = False
    kept: list[str] = []

    for line in lines:
        bare = line.strip()

        # Enter FAILURES section
        if re.search(r'={3,}\s*(FAILURES|ERRORS)\s*={3,}', line, re.I):
            in_failures = True
            kept.append(line)
            continue

        # Enter short-summary section (also exits FAILURES)
        if 'short test summary' in line.lower():
            in_failures = False
            in_summary = True
            kept.append(line)
            continue

        # Closing === line after summary
        if bare.startswith('===') and in_summary and len(kept) > 1:
            kept.append(line)
            in_summary = False
            continue

        if in_failures or in_summary:
            kept.append(line)
            continue

        # Always keep assertion / error lines even outside failure blocks
        if bare.startswith('E ') or bare.startswith('E\t') or 'AssertionError' in line:
            kept.append(line)

    return '\n'.join(kept) if kept else clean[:800]


# ---------------------------------------------------------------------------
# markitdown — binary / HTML → clean Markdown
# ---------------------------------------------------------------------------

_MARKITDOWN_EXTS = {'.html', '.htm', '.pdf', '.docx', '.pptx', '.xlsx', '.rst'}


def markitdown_file(path: str | Path) -> str | None:
    """
    Convert a non-Python file to Markdown using markitdown.
    Returns None if the file type isn't supported or conversion fails.
    """
    p = Path(path)
    if p.suffix.lower() not in _MARKITDOWN_EXTS:
        return None
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(p))
        text = result.text_content or ""
        # Collapse excessive blank lines that markitdown sometimes leaves
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Token-budget truncation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: ~3.5 chars/token for code."""
    return int(len(text) / _CHARS_PER_TOKEN)


def token_budget(text: str, max_tokens: int) -> str:
    """
    Hard-truncate `text` to fit within `max_tokens` (character-estimate).
    Appends a compact notice so the model knows content was cut.
    """
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    used = estimate_tokens(text)
    return text[:max_chars] + f"\n…[{used}→{max_tokens}tok]"
