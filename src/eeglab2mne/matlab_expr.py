"""Translate small MATLAB expressions into Python source.

EEGLAB scripts build file names and loop bounds from variables, e.g.::

    [subjects{s} '.set']          ->  subjects[s - 1] + '.set'
    fullfile(datapath, fname)     ->  os.path.join(datapath, fname)
    sprintf('sub%02d.set', s)     ->  'sub%02d.set' % (s,)
    1:length(subjects)            ->  range(1, len(subjects) + 1)

This is deliberately conservative: anything it does not understand raises
:class:`ExprError` and the caller falls back to a TODO. In particular, a
name is only accepted if the script defined it earlier (``known``), so a
variable that exists only in someone's MATLAB workspace is never silently
assumed to exist in Python.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


class ExprError(Exception):
    """The expression can't be translated safely."""


@dataclass
class ExprEnv:
    known: set[str]  # variables defined earlier in the translated script
    obj: str = "raw"  # current MNE data variable, for EEG.<field>
    imports: set[str] = field(default_factory=set)
    helpers: dict[str, str] = field(default_factory=dict)  # functions emitted once at the top


@dataclass
class _Tok:
    kind: str  # num, str, name, op, end
    text: str
    space_before: bool


_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>==|~=|<=|>=|&&|\|\||\.\*|\./|\.\^|[-+*/^<>&|~!(){}\[\],;:.'"=])
""", re.VERBOSE)


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, space = 0, False
    while i < len(text):
        ch = text[i]
        # Strings: ' after an operand is a transpose; otherwise a string literal.
        if ch in "'\"" and not (ch == "'" and toks and not space and
                                (toks[-1].kind in ("num", "name") or toks[-1].text in ")]}'")):
            j, buf = i + 1, ""
            while True:
                if j >= len(text):
                    raise ExprError("unterminated string")
                if text[j] == ch:
                    if j + 1 < len(text) and text[j + 1] == ch:
                        buf += ch
                        j += 2
                        continue
                    break
                buf += text[j]
                j += 1
            toks.append(_Tok("str", buf, space))
            i, space = j + 1, False
            continue
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise ExprError(f"unexpected character {ch!r}")
        if m.group("ws"):
            space = True
        else:
            kind = m.lastgroup
            toks.append(_Tok(kind, m.group(kind), space))
            space = False
        i = m.end()
    toks.append(_Tok("end", "", space))
    return toks


# MATLAB functions whose behaviour differs from the closest Python builtin.
MATLAB_ROUND = '''def matlab_round(x):
    """MATLAB's round(): halves go away from zero (Python's round() goes to even)."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))'''

MATLAB_NUM2STR = '''def matlab_num2str(x):
    """MATLAB's num2str() for one number: no '.0' on whole numbers, ~5 significant digits."""
    if isinstance(x, str):
        return x
    if float(x).is_integer():
        return str(int(x))
    digits = min(max(math.floor(math.log10(abs(x))) + 5, 5), 16)
    return f"{x:.{digits}g}"'''

MATLAB_COLON = '''def matlab_colon(start, step, stop):
    """MATLAB's start:step:stop: includes stop, allows non-integer and negative steps."""
    if step == 0:
        return []
    n = math.floor((stop - start) / step + 1e-10)
    return [start + i * step for i in range(n + 1)] if n >= 0 else []'''


def _atomic(code: str) -> bool:
    """True if ``code`` can take a method call or operator without parentheses."""
    try:
        node = ast.parse(code, mode="eval").body
    except SyntaxError:
        return False
    # A bare tuple ("a, b") parses fine but is not one operand.
    return isinstance(node, (ast.Name, ast.Constant, ast.Subscript, ast.Call, ast.Attribute,
                             ast.List))


def _wrap(code: str) -> str:
    """Parenthesise ``code`` unless it is already a single operand."""
    return code if _atomic(code) else f"({code})"


_SPRINTF_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f",
                    "v": "\v", "\\": "\\"}


def _sprintf_format(code: str) -> str:
    """sprintf interprets escapes like \\n in its format, even in single-quoted strings."""
    try:
        fmt = ast.literal_eval(code)
    except (ValueError, SyntaxError):
        return _wrap(code)
    if not isinstance(fmt, str):
        return _wrap(code)
    return repr(re.sub(r"\\(.)", lambda m: _SPRINTF_ESCAPES.get(m.group(1), m.group(0)), fmt))


_FUNCS_STR = {"num2str", "int2str", "sprintf", "fullfile", "strcat", "upper", "lower",
              "strrep", "strtrim"}


class _Parser:
    def __init__(self, text: str, env: ExprEnv):
        self.toks = _tokenize(text)
        self.i = 0
        self.env = env
        self.in_brackets = 0

    # -- helpers -----------------------------------------------------------
    def peek(self, k: int = 0) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self, text: str | None = None) -> _Tok:
        t = self.peek()
        if text is not None and t.text != text:
            raise ExprError(f"expected {text!r}, got {t.text!r}")
        self.i += 1
        return t

    def at(self, *texts: str) -> bool:
        t = self.peek()
        return t.kind == "op" and t.text in texts

    def _element_boundary(self) -> bool:
        """Inside [...], whitespace can separate elements: [a b], [a -b]."""
        if not self.in_brackets:
            return False
        t = self.peek()
        if not t.space_before:
            return False
        if t.kind in ("num", "str", "name") or t.text in ("(", "[", "{"):
            return True
        # '[a -b]' is two elements, '[a - b]' is one.
        return t.text in ("+", "-") and not self.peek(1).space_before

    # -- grammar -----------------------------------------------------------
    def parse(self) -> str:
        out = self.expr()
        if self.peek().kind != "end":
            raise ExprError(f"unexpected {self.peek().text!r}")
        return out

    def expr(self) -> str:
        left = self.and_()
        while self.at("||", "|") and not self._element_boundary():
            self.take()
            left = f"({left} or {self.and_()})"
        return left

    def and_(self) -> str:
        left = self.cmp()
        while self.at("&&", "&") and not self._element_boundary():
            self.take()
            left = f"({left} and {self.cmp()})"
        return left

    def cmp(self) -> str:
        left = self.add()
        ops = {"==": "==", "~=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">="}
        if self.at(*ops) and not self._element_boundary():
            op = ops[self.take().text]
            left = f"{left} {op} {self.add()}"
        return left

    def add(self) -> str:
        left = self.mul()
        while self.at("+", "-") and not self._element_boundary():
            op = self.take().text
            left = f"{left} {op} {self.mul()}"
        return left

    def mul(self) -> str:
        left = self.unary()
        while self.at("*", "/", ".*", "./") and not self._element_boundary():
            op = self.take().text.lstrip(".")
            left = f"{left} {op} {self.unary()}"
        return left

    def unary(self) -> str:
        if self.at("-", "+"):
            op = self.take().text
            return f"{op}{self.unary()}"
        if self.at("~", "!"):
            self.take()
            return f"(not {self.unary()})"
        return self.power()

    def power(self) -> str:
        # MATLAB's ^ binds tighter than unary minus (-2^2 == -4) and, unlike
        # Python's **, groups left to right: 2^3^2 == (2^3)^2 == 64.
        left = self.postfix()
        chained = False
        while self.at("^", ".^") and not self._element_boundary():
            self.take()
            if self.at("-", "+"):  # 2^-1
                sign = self.take().text
                right = f"{sign}{self.postfix()}"
            else:
                right = self.postfix()
            left = f"({left}) ** {_wrap(right)}" if chained else f"{_wrap(left)} ** {_wrap(right)}"
            chained = True
        return left

    def args(self, close: str) -> list[str]:
        out: list[str] = []
        saved, self.in_brackets = self.in_brackets, 0
        if not self.at(close):
            while True:
                out.append(self.expr())
                if self.at(","):
                    self.take()
                    continue
                break
        self.take(close)
        self.in_brackets = saved
        return out

    def postfix(self) -> str:
        t = self.peek()
        if t.kind == "name" and self.peek(1).text == "(" and not self.peek(1).space_before:
            name = self.take().text
            self.take("(")
            return self.call(name, self.args(")"))
        base = self.primary()
        while True:
            nxt = self.peek()
            if nxt.space_before and self.in_brackets:
                return base
            if self.at("{"):
                self.take()
                idx = self.args("}")
                base = f"{base}[{self.index(idx)}]"
            elif self.at("("):
                self.take()
                idx = self.args(")")
                base = f"{base}[{self.index(idx)}]"
            elif self.at("'"):
                raise ExprError("transpose")
            else:
                return base

    def index(self, idx: list[str]) -> str:
        if len(idx) != 1:
            raise ExprError("multi-dimensional indexing")
        i = idx[0]
        if re.fullmatch(r"\d+", i):
            return str(int(i) - 1)  # MATLAB is 1-based
        return f"{i} - 1"

    def primary(self) -> str:
        t = self.take()
        if t.kind == "num":
            v = float(t.text)
            return str(int(v)) if v.is_integer() and "." not in t.text and "e" not in t.text.lower() \
                else t.text
        if t.kind == "str":
            return repr(t.text)
        if t.kind == "name":
            if t.text in ("true", "false"):
                return t.text.capitalize()
            if t.text == "pi":
                self.env.imports.add("import numpy as np")
                return "np.pi"
            if t.text == "EEG" and self.at("."):
                self.take(".")
                return self.eeg_field(self.take().text)
            if t.text not in self.env.known:
                raise ExprError(f"`{t.text}` is not defined earlier in the script")
            if self.at("."):
                raise ExprError("struct field access")
            return t.text
        if t.text == "(":
            saved, self.in_brackets = self.in_brackets, 0
            inner = self.expr()
            self.take(")")
            self.in_brackets = saved
            return f"({inner})"
        if t.text in ("[", "{"):
            return self.bracket("]" if t.text == "[" else "}", cell=t.text == "{")
        raise ExprError(f"unexpected {t.text!r}")

    def bracket(self, close: str, cell: bool) -> str:
        elems: list[str] = []
        kinds: list[str] = []
        self.in_brackets += 1
        while not self.at(close):
            if self.at(",", ";"):
                if self.peek().text == ";":
                    raise ExprError("matrix with several rows")
                self.take()
                continue
            start = self.peek().kind
            elems.append(self.expr())
            kinds.append(start)
        self.in_brackets -= 1
        self.take(close)
        if len(elems) == 1 and not cell:
            return _wrap(elems[0])  # [a + b] * 2 must stay (a + b) * 2
        stringy = [k == "str" or _is_stringy(e) for k, e in zip(kinds, elems)]
        if cell or not any(stringy):
            return "[" + ", ".join(elems) + "]"
        if any(k == "num" for k in kinds):
            raise ExprError("mixes text and numbers in [...]; MATLAB would convert to characters")
        return " + ".join(elems)  # MATLAB [a b] on text is concatenation

    def eeg_field(self, name: str) -> str:
        o = self.env.obj
        fields = {"srate": f"{o}.info['sfreq']", "nbchan": f"len({o}.ch_names)",
                  "pnts": f"len({o}.times)"}
        if name == "trials" and o == "epochs":
            return "len(epochs)"
        if name not in fields:
            raise ExprError(f"EEG.{name} has no direct MNE equivalent here")
        return fields[name]

    def helper(self, name: str, source: str) -> str:
        self.env.imports.add("import math")
        self.env.helpers[name] = source
        return name

    def call(self, name: str, a: list[str]) -> str:
        env = self.env
        if name in env.known:  # array indexing with ()
            return f"{name}[{self.index(a)}]"
        if name == "fullfile" and a:
            env.imports.add("import os")
            return f"os.path.join({', '.join(a)})"
        if name == "sprintf" and a:
            fmt, rest = _sprintf_format(a[0]), a[1:]
            return f"({fmt} % ({', '.join(rest)},))" if rest else f"({fmt} % ())"
        if name == "num2str" and len(a) == 1:
            return f"{self.helper('matlab_num2str', MATLAB_NUM2STR)}({a[0]})"
        if name == "int2str" and len(a) == 1:
            return f"str({self.helper('matlab_round', MATLAB_ROUND)}({a[0]}))"
        if name == "strcat" and a:
            return "(" + " + ".join(_wrap(x) for x in a) + ")"
        if name in ("length", "numel") and len(a) == 1:
            return f"len({a[0]})"
        if name in ("upper", "lower") and len(a) == 1:
            return f"{_wrap(a[0])}.{name}()"
        if name == "strtrim" and len(a) == 1:
            return f"{_wrap(a[0])}.strip()"
        if name == "strrep" and len(a) == 3:
            return f"{_wrap(a[0])}.replace({a[1]}, {a[2]})"
        if name == "isempty" and len(a) == 1:
            return f"(len({a[0]}) == 0)"
        if name == "strcmp" and len(a) == 2:
            return f"({a[0]} == {a[1]})"
        if name == "strcmpi" and len(a) == 2:
            return f"({_wrap(a[0])}.lower() == {_wrap(a[1])}.lower())"
        if name == "mod" and len(a) == 2:
            return f"({_wrap(a[0])} % {_wrap(a[1])})"
        if name in ("floor", "ceil") and len(a) == 1:
            env.imports.add("import math")
            return f"math.{name}({a[0]})"
        if name == "round" and len(a) == 1:
            return f"{self.helper('matlab_round', MATLAB_ROUND)}({a[0]})"
        if name == "exist" and len(a) in (1, 2):
            # MATLAB returns a code, not a bool: 2 = file, 7 = folder, 0 = neither.
            kind = a[1] if len(a) == 2 else "'file'"
            env.imports.add("import os")
            if kind == "'file'":
                return f"(2 if os.path.isfile({a[0]}) else 7 if os.path.isdir({a[0]}) else 0)"
            if kind == "'dir'":
                return f"(7 if os.path.isdir({a[0]}) else 0)"
            raise ExprError(f"exist(..., {kind}) checks MATLAB's workspace, not the file system")
        raise ExprError(f"function `{name}` is not supported in expressions")


_STRINGY = re.compile(r"""^(\(?['"]|str\(|matlab_num2str\(|os\.path\.join\(|\(\(?['"])|\.(upper|lower|strip|replace)\(""")


def _is_stringy(code: str) -> bool:
    return bool(_STRINGY.search(code)) or code.endswith((".upper()", ".lower()", ".strip()"))


def to_python(text: str, env: ExprEnv) -> str:
    """Translate one MATLAB expression, or raise :class:`ExprError`.

    Imports and helpers the expression needs are added to ``env`` only when the
    whole expression translates, so a failed attempt leaves nothing behind.
    """
    scratch = ExprEnv(known=env.known, obj=env.obj)
    code = _Parser(text, scratch).parse()
    env.imports |= scratch.imports
    env.helpers.update(scratch.helpers)
    return code


_INT_LIT = re.compile(r"-?\d+")
_FLOAT_LIT = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+(?=[eE]))(?:[eE][+-]?\d+)?")


def range_to_python(text: str, env: ExprEnv) -> str:
    """Translate a ``for`` range such as ``1:n`` or ``1:2:n`` into Python.

    Whole-number literal steps become ``range()``; anything else (a step held
    in a variable, or non-integer bounds) uses a helper that follows MATLAB's
    colon rules, because ``range()`` can't take floats and its stop depends on
    the step's sign.
    """
    parts = _split_colon(text)
    if len(parts) == 1:
        return to_python(text, env)  # iterate over a list
    if len(parts) not in (2, 3):
        raise ExprError("unsupported range")
    p = [to_python(x, env) for x in parts]
    start, step, stop = (p[0], "1", p[1]) if len(p) == 2 else p
    if _INT_LIT.fullmatch(step) and int(step) != 0 and not (
            _FLOAT_LIT.fullmatch(start) or _FLOAT_LIT.fullmatch(stop)):
        n = int(step)
        end = f"{_wrap(stop)} + 1" if n > 0 else f"{_wrap(stop)} - 1"
        return f"range({start}, {end})" if n == 1 else f"range({start}, {end}, {n})"
    env.imports.add("import math")
    env.helpers["matlab_colon"] = MATLAB_COLON
    return f"matlab_colon({start}, {step}, {stop})"


def _split_colon(text: str) -> list[str]:
    parts, depth, buf, quote = [], 0, "", False
    for i, ch in enumerate(text):
        if ch == "'" and (quote or not buf.rstrip() or not (buf.rstrip()[-1].isalnum() or
                                                            buf.rstrip()[-1] in ")]}")):
            quote = not quote
        if not quote:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ":" and depth == 0:
                parts.append(buf)
                buf = ""
                continue
        buf += ch
    parts.append(buf)
    return [p.strip() for p in parts]
