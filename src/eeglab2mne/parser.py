"""Parse EEGLAB MATLAB scripts (or ``EEG.history``) into structured calls.

This is deliberately *not* a full MATLAB parser. EEGLAB scripts and the
command history EEGLAB writes are overwhelmingly simple, flat statements of
the form::

    EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);

so we parse just enough MATLAB to recover the function name and its
arguments as Python values. Anything we cannot evaluate safely is kept as an
:class:`Expr` holding the original source text, so nothing is silently lost.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Expr:
    """A MATLAB expression we could not (or should not) evaluate.

    Examples: a variable name like ``EEG`` or ``chanlocs``, a field access
    like ``EEG.srate``, or an arithmetic expression like ``EEG.srate/2``.
    """

    text: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Expr({self.text!r})"


@dataclass
class Call:
    """One EEGLAB statement, e.g. ``EEG = pop_resample(EEG, 250);``."""

    func: str
    args: list[Any]
    outputs: list[str]
    source: str
    line: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    positional: list[Any] = field(default_factory=list)
    # Ordered ('key', value) pairs. Unlike ``kwargs`` this keeps repeated keys,
    # which some functions rely on (e.g. several 'changefield' edits).
    pairs: list[tuple[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Case-insensitive lookup of a ``'key', value`` argument."""
        return self.kwargs.get(key.lower(), default)

    def has(self, key: str) -> bool:
        return key.lower() in self.kwargs


# --------------------------------------------------------------------------
# Statement splitting
# --------------------------------------------------------------------------

_OPEN = "([{"
_CLOSE = ")]}"


def _is_transpose_quote(text: str, i: int) -> bool:
    """MATLAB uses ``'`` both for strings and for transpose (``x'``)."""
    j = i - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    if j < 0:
        return False
    prev = text[j]
    # Transpose follows an identifier, a number, a closing bracket or another
    # transpose, *with no whitespace in between*.
    return j == i - 1 and (prev.isalnum() or prev in "_)]}.'")


def strip_comment(line: str) -> str:
    """Remove a trailing ``%`` comment, respecting string literals."""
    in_str = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'":
            if in_str:
                if i + 1 < len(line) and line[i + 1] == "'":
                    i += 2  # escaped quote inside a string
                    continue
                in_str = False
            elif not _is_transpose_quote(line, i):
                in_str = True
        elif ch == "%" and not in_str:
            return line[:i]
        i += 1
    return line


def split_statements(text: str) -> list[tuple[int, str]]:
    """Split MATLAB source into ``(line_number, statement)`` pairs.

    Handles ``%`` comments, ``%{ ... %}`` block comments, ``...`` line
    continuations, and multiple statements per line separated by ``;`` or
    ``,`` at bracket depth zero.
    """
    statements: list[tuple[int, str]] = []
    buf = ""
    buf_line = 0
    in_block_comment = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "%{":
            in_block_comment = True
            continue
        if stripped == "%}":
            in_block_comment = False
            continue
        if in_block_comment:
            continue

        line = strip_comment(raw).rstrip()
        if not buf:
            buf_line = lineno
        if line.endswith("..."):
            buf += line[:-3] + " "
            continue
        buf += line

        # Now split buf at top-level ; or , (outside brackets and strings).
        depth = 0
        in_str = False
        start = 0
        i = 0
        while i < len(buf):
            ch = buf[i]
            if ch == "'":
                if in_str:
                    if i + 1 < len(buf) and buf[i + 1] == "'":
                        i += 2
                        continue
                    in_str = False
                elif not _is_transpose_quote(buf, i):
                    in_str = True
            elif not in_str:
                if ch in _OPEN:
                    depth += 1
                elif ch in _CLOSE:
                    depth -= 1
                elif ch in ";," and depth == 0:
                    piece = buf[start:i].strip()
                    if piece:
                        statements.append((buf_line, piece))
                    start = i + 1
            i += 1
        piece = buf[start:].strip()
        if piece:
            statements.append((buf_line, piece))
        buf = ""

    if buf.strip():
        statements.append((buf_line, buf.strip()))
    return statements


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_RANGE_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*:\s*(?:([+-]?\d+(?:\.\d+)?)\s*:\s*)?([+-]?\d+(?:\.\d+)?)$")
_CALL_RE = re.compile(
    r"^(?:(?P<out>\[[^\]]*\]|[A-Za-z_][\w.]*)\s*=\s*)?"  # optional outputs
    r"(?P<func>[A-Za-z_]\w*)\s*"
    r"(?:\((?P<args>.*)\))?\s*$",
    re.DOTALL,
)


def _split_top_level(s: str, seps: str) -> list[str]:
    """Split ``s`` on any char in ``seps`` that sits at bracket depth zero."""
    parts: list[str] = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_str:
                if i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
            elif not _is_transpose_quote(s, i):
                in_str = True
        elif not in_str:
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth -= 1
            elif ch in seps and depth == 0:
                parts.append(s[start:i])
                start = i + 1
        i += 1
    parts.append(s[start:])
    return parts


def _split_matrix_row(row: str) -> list[str]:
    """Split a matrix row on commas *or* whitespace (MATLAB allows both).

    Whitespace splitting must not break ``1 - 2`` style expressions too
    aggressively, but EEGLAB scripts very rarely contain those inside literal
    matrices, so we accept the simple rule.
    """
    items: list[str] = []
    for chunk in _split_top_level(row, ","):
        # whitespace split, respecting brackets/strings
        depth = 0
        in_str = False
        cur = ""
        i = 0
        while i < len(chunk):
            ch = chunk[i]
            if ch == "'":
                if in_str:
                    if i + 1 < len(chunk) and chunk[i + 1] == "'":
                        cur += "''"
                        i += 2
                        continue
                    in_str = False
                elif not _is_transpose_quote(chunk, i):
                    in_str = True
            if not in_str and ch in _OPEN:
                depth += 1
            if not in_str and ch in _CLOSE:
                depth -= 1
            if ch in " \t" and depth == 0 and not in_str:
                if cur.strip():
                    items.append(cur.strip())
                cur = ""
            else:
                cur += ch
            i += 1
        if cur.strip():
            items.append(cur.strip())
    return items


def parse_value(token: str) -> Any:
    """Convert one MATLAB argument token into a Python value.

    ======================  ===========================
    MATLAB                  Python
    ======================  ===========================
    ``'text'``              ``'text'``
    ``1.5``, ``1e-3``       ``1.5``, ``0.001``
    ``NaN``, ``Inf``        ``math.nan``, ``math.inf``
    ``true`` / ``false``    ``True`` / ``False``
    ``[]``                  ``[]``
    ``[1 2 3]``             ``[1, 2, 3]``
    ``[1 2; 3 4]``          ``[[1, 2], [3, 4]]``
    ``{'a' 'b'}``           ``['a', 'b']``
    ``1:4``                 ``[1, 2, 3, 4]``
    anything else           ``Expr('...')``
    ======================  ===========================
    """
    t = token.strip()
    if not t:
        return Expr("")

    # String literal (whole token is one quoted string)
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        inner = t[1:-1]
        # make sure it's a single string, not 'a' 'b'
        if "'" not in inner.replace("''", ""):
            return inner.replace("''", "'")
    # Double-quoted MATLAB string (R2017a+)
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"' and '"' not in t[1:-1]:
        return t[1:-1]

    if _NUMBER_RE.match(t):
        v = float(t)
        return int(v) if v.is_integer() and "." not in t and "e" not in t.lower() else v

    low = t.lower()
    if low in ("nan", "+nan", "-nan"):
        return math.nan
    if low in ("inf", "+inf"):
        return math.inf
    if low == "-inf":
        return -math.inf
    if low == "true":
        return True
    if low == "false":
        return False

    m = _RANGE_RE.match(t)
    if m:
        a, step, b = m.group(1), m.group(2), m.group(3)
        a_f, b_f = float(a), float(b)
        step_f = float(step) if step else 1.0
        if step_f != 0 and abs((b_f - a_f) / step_f) < 100_000:
            vals = []
            x = a_f
            while (step_f > 0 and x <= b_f + 1e-12) or (step_f < 0 and x >= b_f - 1e-12):
                vals.append(int(x) if x.is_integer() else x)
                x += step_f
            return vals

    if t[0] in "[{" and t[-1] == ("]" if t[0] == "[" else "}"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        rows = [r for r in _split_top_level(inner, ";\n") if r.strip()]
        parsed_rows = [[parse_value(x) for x in _split_matrix_row(r)] for r in rows]
        if len(parsed_rows) == 1:
            row = parsed_rows[0]
            # Flatten [1:4] -> [1, 2, 3, 4] and [a, 1:3] -> [a, 1, 2, 3]
            flat: list[Any] = []
            for item in row:
                if isinstance(item, list) and t[0] == "[":
                    flat.extend(item)
                else:
                    flat.append(item)
            return flat
        return parsed_rows

    return Expr(t)


def _parse_outputs(out: str | None) -> list[str]:
    if not out:
        return []
    out = out.strip()
    if out.startswith("["):
        return [o for o in re.split(r"[\s,]+", out[1:-1].strip()) if o]
    return [out]


def _pair_kwargs(call: Call) -> None:
    """Split args into positional and ``'key', value`` pairs.

    EEGLAB convention: the dataset (``EEG``) comes first, then either
    positional arguments or a run of string keys each followed by a value.
    We find the first string argument after which the remaining arguments
    pair up cleanly as (string, value) and treat that as the kwargs region.
    """
    args = list(call.args)
    if args and isinstance(args[0], Expr) and args[0].text in ("EEG", "ALLEEG", "EEGOUT", "OUTEEG"):
        args = args[1:]

    for start in range(len(args)):
        tail = args[start:]
        if len(tail) % 2 == 0 and all(isinstance(tail[i], str) for i in range(0, len(tail), 2)):
            # Require keys to look like identifiers, so 'Fz' as a *value*
            # is not misread as a key in positional calls.
            if all(re.match(r"^[A-Za-z_]\w*$", tail[i]) for i in range(0, len(tail), 2)) and tail:
                call.positional = args[:start]
                call.pairs = [(tail[i].lower(), tail[i + 1]) for i in range(0, len(tail), 2)]
                call.kwargs = dict(call.pairs)
                return
    call.positional = args
    call.kwargs = {}


def parse_statement(stmt: str, line: int = 0) -> Call | None:
    """Parse one statement. Returns ``None`` for non-call statements."""
    m = _CALL_RE.match(stmt.strip().rstrip(";").strip())
    if not m:
        return None
    func = m.group("func")
    raw_args = m.group("args")
    args: list[Any] = []
    if raw_args is not None and raw_args.strip():
        args = [parse_value(a) for a in _split_top_level(raw_args, ",")]
    call = Call(
        func=func,
        args=args,
        outputs=_parse_outputs(m.group("out")),
        source=stmt.strip(),
        line=line,
    )
    _pair_kwargs(call)
    return call


def parse_script(text: str) -> list[Call | str]:
    """Parse a whole script. Unparseable statements are returned as strings."""
    results: list[Call | str] = []
    for line, stmt in split_statements(text):
        call = parse_statement(stmt, line)
        results.append(call if call is not None else stmt)
    return results
