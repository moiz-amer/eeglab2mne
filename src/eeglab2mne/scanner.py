"""Character-level scanning of MATLAB source.

Splitting statements, stripping comments and splitting argument lists all need
the same answer to one question: is this character inside a string literal?
MATLAB makes that awkward, because ``'`` starts a string *or* means transpose
(``x'``), and quotes inside strings are escaped by doubling (``'it''s'``, and
likewise for double quotes). Everything that needs to know goes through
:func:`scan`.
"""

from __future__ import annotations

from typing import Iterator

OPEN = "([{"
CLOSE = ")]}"


def is_transpose_quote(text: str, i: int) -> bool:
    """True if the ``'`` at ``text[i]`` is a transpose rather than a string start.

    A transpose directly follows an identifier, a number, a closing bracket,
    ``.`` (for ``.'``) or another transpose, with no whitespace in between.
    """
    if i == 0:
        return False
    prev = text[i - 1]
    return prev.isalnum() or prev in "_)]}.'"


def scan(text: str) -> Iterator[tuple[int, str, bool]]:
    """Yield ``(index, char, in_string)`` for every character of ``text``.

    ``in_string`` is True for string literals including their quotes, for both
    ``'single'`` and ``"double"`` quoted strings.
    """
    quote = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < n and text[i + 1] == quote:  # doubled quote = escaped
                    yield i, ch, True
                    yield i + 1, ch, True
                    i += 2
                    continue
                quote = None
            yield i, ch, True
        elif ch == '"' or (ch == "'" and not is_transpose_quote(text, i)):
            quote = ch
            yield i, ch, True
        else:
            yield i, ch, False
        i += 1


def code_chars(text: str) -> Iterator[tuple[int, str, int]]:
    """Yield ``(index, char, depth)`` for characters outside string literals.

    ``depth`` is the bracket depth *before* the character, so an opening
    bracket is reported at the depth it opens from.
    """
    depth = 0
    for i, ch, in_str in scan(text):
        if in_str:
            continue
        yield i, ch, depth
        if ch in OPEN:
            depth += 1
        elif ch in CLOSE:
            depth -= 1


def split_top_level(text: str, seps: str) -> list[str]:
    """Split ``text`` on any char in ``seps`` outside brackets and strings."""
    parts, start = [], 0
    for i, ch, depth in code_chars(text):
        if depth == 0 and ch in seps:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def strip_comment(line: str) -> str:
    """Remove a trailing ``%`` comment (a ``%`` inside a string is kept)."""
    for i, ch, _ in code_chars(line):
        if ch == "%":
            return line[:i]
    return line


def split_continuation(line: str) -> tuple[str, bool]:
    """Handle MATLAB's ``...`` line continuation.

    Returns ``(code, continues)``. Anything after ``...`` on the same line is a
    comment, so it is dropped.
    """
    for i, ch, _ in code_chars(line):
        if ch == "." and line.startswith("...", i):
            return line[:i], True
    return line, False
