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

from .scanner import split_continuation, split_top_level, strip_comment


@dataclass(frozen=True)
class Expr:
    """A MATLAB expression we could not (or should not) evaluate.

    Examples: a variable name like ``EEG`` or ``chanlocs``, a field access
    like ``EEG.srate``, or an arithmetic expression like ``EEG.srate/2``.
    """

    text: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Expr({self.text!r})"


class MatList(list):
    """A list parsed from ``[...]`` or ``{...}`` that remembers its MATLAB source.

    Needed because ``[name '.set']`` is a list of parts to the parser but a
    string concatenation to MATLAB; the expression translator decides.
    """

    text: str = ""


def _matlist(items: list, text: str) -> MatList:
    out = MatList(items)
    out.text = text
    return out


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

        line, continues = split_continuation(strip_comment(raw))
        if not buf:
            buf_line = lineno
        if continues:
            buf += line + " "
            continue
        buf += line.rstrip()

        for piece in split_top_level(buf, ";,"):
            if piece.strip():
                statements.append((buf_line, piece.strip()))
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


def _split_matrix_row(row: str) -> list[str]:
    """Split a matrix row on commas *or* whitespace (MATLAB allows both).

    Whitespace splitting must not break ``1 - 2`` style expressions too
    aggressively, but EEGLAB scripts very rarely contain those inside literal
    matrices, so we accept the simple rule.
    """
    items: list[str] = []
    for chunk in split_top_level(row, ","):
        items += [x.strip() for x in split_top_level(chunk, " \t") if x.strip()]
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
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"' and '"' not in t[1:-1].replace('""', ''):
        return t[1:-1].replace('""', '"')

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
        rows = [r for r in split_top_level(inner, ";\n") if r.strip()]
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
            return _matlist(flat, t)
        return _matlist(parsed_rows, t)

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
        args = [parse_value(a) for a in split_top_level(raw_args, ",")]
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
