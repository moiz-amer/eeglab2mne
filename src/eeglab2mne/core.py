"""High-level API: EEGLAB script / .set history -> MNE script + audit report."""

from __future__ import annotations

import keyword
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .matlab_expr import ExprError, range_to_python, to_python
from .parser import Call, parse_script, split_statements
from .translators import (CONTROL_FLOW, CONTROL_NOTE, HISTORY_PATH_NOTE, ICA_REUSE_NOTE, REGISTRY,
                          Context, Status, Step, ica_reuse_code, translate_call)

_BADGE = {
    Status.EXACT: "✅ exact",
    Status.APPROX: "⚠️ approximate",
    Status.MANUAL: "❌ manual",
    Status.SKIPPED: "➖ skipped",
}


@dataclass
class Translation:
    steps: list[Step]
    ctx: Context
    source_name: str

    @property
    def counts(self) -> Counter:
        return Counter(s.status for s in self.steps if not s.structural)

    def to_python(self) -> str:
        return render_python(self)

    def to_report(self) -> str:
        return render_report(self)


def translate(text: str, *, ica_mode: str = "refit", source_name: str = "<script>",
              ica_source: str | None = None, set_info: dict | None = None) -> Translation:
    """Translate EEGLAB MATLAB code into a :class:`Translation`.

    Parameters
    ----------
    text
        EEGLAB script, or the contents of ``EEG.history``.
    ica_mode
        ``"refit"`` fits a fresh ICA in MNE; ``"reuse"`` loads EEGLAB's own
        decomposition (and its rejected components) from the .set file.
    ica_source
        In ``"reuse"`` mode, the .set file that holds the ICA. Defaults to the
        file loaded by ``pop_loadset`` in the script.
    set_info
        Facts about the .set file being translated (see :func:`inspect_set`).
        Lets the translation point at the real file, read epoched data as
        Epochs, and load an ICA that was computed before the recorded history.
    """
    if ica_mode not in ("refit", "reuse"):
        raise ValueError("ica_mode must be 'refit' or 'reuse'")
    ctx = Context(ica_mode=ica_mode, ica_source=ica_source, set_info=set_info)
    builder = _Builder(ctx)
    for line, stmt in split_statements(text):
        builder.add(line, stmt)
    steps = builder.finish()

    if set_info and not any(st.indent for st in steps):
        _mark_superseded(steps, set_info["path"])

    # Paths in a .set history were chosen by whoever made the file: surface the risky ones.
    for st in steps:
        if st.status is not Status.SKIPPED:
            ctx.warnings += [f"Line {st.line}: {n[len(HISTORY_PATH_NOTE):]}" for n in st.notes
                             if n.startswith(HISTORY_PATH_NOTE)]

    # The dataset may carry an ICA computed before its recorded history began.
    if set_info and set_info.get("has_ica") and not ctx.has_ica:
        if ica_mode == "reuse":
            code = ica_reuse_code(ctx) + [
                f"# ica.apply({ctx.obj})  # removes the components in ica.exclude"]
            steps.append(Step(None, "(ICA decomposition stored in the dataset)", 0, Status.EXACT,
                              code, [ICA_REUSE_NOTE, "This ICA is not in EEG.history (it was "
                                     "computed before the recorded steps), so it is loaded here."]))
        else:
            ctx.warnings.append("This dataset contains an ICA decomposition that is not in its "
                                "history. Run with --ica reuse to load it.")
    return Translation(steps, ctx, source_name)


TODO_HELPER = [
    "def TODO(what):",
    "    \"\"\"Stops the script where eeglab2mne could not translate MATLAB code.\"\"\"",
    "    raise NotImplementedError(f'eeglab2mne: translate this MATLAB code by hand: {what}')",
]

_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _is_ident(name: str) -> bool:
    """True for a plain MATLAB/Python identifier (and not a Python keyword)."""
    return bool(_IDENT_RE.fullmatch(name)) and not keyword.iskeyword(name)


_FUNC_RE = re.compile(r"^function\s+(?:(\[[^\]]*\]|[A-Za-z_]\w*)\s*=\s*)?([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?$")


class _Builder:
    """Turns MATLAB statements into Steps, tracking Python block structure."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.steps: list[Step] = []
        # Open blocks: dicts with kind, has_body, and (switch) subject/seen_case.
        self.blocks: list[dict] = []

    # -- structure -----------------------------------------------------------
    @property
    def depth(self) -> int:
        return sum(1 for b in self.blocks if b["kind"] != "switch" or b.get("in_case"))

    def emit(self, step: Step) -> None:
        step.indent = self.depth
        if step.code and self.blocks:
            self.blocks[-1]["has_body"] = True
        self.steps.append(step)

    def ensure_body(self) -> None:
        if self.blocks and not self.blocks[-1]["has_body"]:
            self.steps.append(Step(None, "pass", 0, Status.SKIPPED, ["pass"],
                                   indent=self.depth, structural=True))
            self.blocks[-1]["has_body"] = True

    def todo(self, what: str) -> str:
        self.ctx.uses_todo = True
        return f"TODO({what!r})"

    def expr(self, text: str) -> tuple[str, bool]:
        try:
            return to_python(text.strip(), self.ctx.env()), True
        except ExprError:
            return self.todo(text.strip()), False

    def opener(self, line: int, src: str, code: str, ok: bool, kind: str, note: str) -> None:
        self.emit(Step(None, src, line, Status.EXACT if ok else Status.MANUAL, [code],
                       [note] if ok else [note, "Could not translate this part automatically; "
                                                "the script stops here with a TODO error until you do."]))
        self.blocks.append({"kind": kind, "has_body": False})

    # -- statements ----------------------------------------------------------
    def add(self, line: int, stmt: str) -> None:
        stmt = stmt.strip().rstrip(";").strip()
        if not stmt:
            return
        word = stmt.split(None, 1)[0].split("(")[0]
        rest = stmt[len(word):].strip()
        if word in CONTROL_FLOW:
            self.control(line, stmt, word, rest)
            return
        m = _ASSIGN_RE.match(stmt)
        if m and not stmt.startswith("[") and m.group(1) not in ("EEG", "ALLEEG"):
            rhs_func = re.match(r"^([A-Za-z_]\w*)\s*\(", m.group(2).strip())
            if not (rhs_func and rhs_func.group(1).lower() in REGISTRY):
                self.assignment(line, stmt, m.group(1), m.group(2))
                return
        items = parse_script(stmt)
        for item in items:
            if isinstance(item, Call):
                item.line = line
                self.emit(translate_call(item, self.ctx))
            else:
                self.emit(Step(None, item, line, Status.MANUAL, [f"# TODO: {item}"],
                               ["Not a function call (assignment or expression); translate by hand."]))

    def assignment(self, line: int, stmt: str, name: str, rhs: str) -> None:
        if rhs.strip() in ("EEG", "ALLEEG"):
            self.emit(Step(None, stmt, line, Status.SKIPPED, [],
                           ["Dataset copy. MNE methods modify in place; use .copy() if you need "
                            "the original later."]))
            return
        try:
            code = to_python(rhs.strip(), self.ctx.env())
        except ExprError as e:
            self.ctx.known_vars.add(name)  # later uses fail loudly via TODO, not NameError
            self.emit(Step(None, stmt, line, Status.MANUAL, [f"{name} = {self.todo(rhs.strip())}"],
                           [f"Could not translate the right-hand side ({e})."]))
            return
        self.ctx.known_vars.add(name)
        self.emit(Step(None, stmt, line, Status.EXACT, [f"{name} = {code}"],
                       ["Variable assignment."]))

    def control(self, line: int, stmt: str, word: str, rest: str) -> None:
        ctx = self.ctx
        if word in ("for", "parfor"):
            inner = rest[1:-1] if rest.startswith("(") and rest.endswith(")") else rest
            m = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$", inner)
            if m:
                var, rng = m.group(1), m.group(2)
                try:
                    it, ok = range_to_python(rng, ctx.env()), True
                except ExprError:
                    it, ok = self.todo(rng), False
                ctx.known_vars.add(var)
                self.opener(line, stmt, f"for {var} in {it}:", ok, "for",
                            "Loop. MATLAB ranges include the end value, so range() goes one past it.")
            else:
                self.opener(line, stmt, f"for _ in {self.todo(rest)}:", False, "for", "Loop.")
        elif word == "while":
            cond, ok = self.expr(rest)
            self.opener(line, stmt, f"while {cond}:", ok, "while", "Loop.")
        elif word == "if":
            cond, ok = self.expr(rest)
            self.opener(line, stmt, f"if {cond}:", ok, "if", "Condition.")
        elif word in ("elseif", "else"):
            if not self.blocks or self.blocks[-1]["kind"] != "if":
                self.emit(Step(None, stmt, line, Status.MANUAL, [f"# TODO: {stmt}"], [CONTROL_NOTE]))
                return
            self.ensure_body()
            self.blocks.pop()
            if word == "else":
                self.opener(line, stmt, "else:", True, "if", "Condition.")
            else:
                cond, ok = self.expr(rest)
                self.opener(line, stmt, f"elif {cond}:", ok, "if", "Condition.")
        elif word == "switch":
            subject, ok = self.expr(rest)
            self.emit(Step(None, stmt, line, Status.EXACT if ok else Status.MANUAL, [],
                           ["switch/case becomes an if/elif chain."], structural=True))
            self.blocks.append({"kind": "switch", "has_body": True, "subject": subject,
                                "seen_case": False, "in_case": False})
        elif word in ("case", "otherwise"):
            sw = self.blocks[-1] if self.blocks else None
            if not sw or sw["kind"] != "switch":
                self.emit(Step(None, stmt, line, Status.MANUAL, [f"# TODO: {stmt}"], [CONTROL_NOTE]))
                return
            if sw["in_case"]:
                self.ensure_body()
            sw["in_case"] = False
            if word == "otherwise":
                code, ok = ("else:" if sw["seen_case"] else "if True:"), True
            else:
                value, ok = self.expr(rest)
                test = f"{sw['subject']} in {value}" if value.startswith("[") else \
                    f"{sw['subject']} == {value}"
                code = f"{'elif' if sw['seen_case'] else 'if'} {test}:"
            self.emit(Step(None, stmt, line, Status.EXACT if ok else Status.MANUAL, [code],
                           ["switch/case becomes an if/elif chain."]))
            sw.update(seen_case=True, in_case=True, has_body=False)
        elif word == "try":
            self.opener(line, stmt, "try:", True, "try", "Error handling.")
        elif word == "catch":
            if not self.blocks or self.blocks[-1]["kind"] != "try":
                self.emit(Step(None, stmt, line, Status.MANUAL, [f"# TODO: {stmt}"], [CONTROL_NOTE]))
                return
            self.ensure_body()
            self.blocks.pop()
            var = rest.split()[0] if rest else ""
            if var and not _is_ident(var):
                # Copied into Python source, so anything but a plain name is refused.
                self.opener(line, stmt, "except Exception:", False, "try",
                            f"`{var}` is not a plain variable name, so it was left out.")
                return
            if var:
                ctx.known_vars.add(var)
            self.opener(line, stmt, f"except Exception{' as ' + var if var else ''}:", True, "try",
                        "Error handling.")
        elif word == "function":
            m = _FUNC_RE.match(stmt)
            if not m:
                self.emit(Step(None, stmt, line, Status.MANUAL, [f"# TODO: {stmt}"], [CONTROL_NOTE]))
                return
            outs = [o for o in re.split(r"[\s,]+", (m.group(1) or "").strip("[]")) if o]
            params = [a.strip() for a in (m.group(3) or "").split(",") if a.strip()]
            params = ["_" if a == "~" else a for a in params]  # ~ marks an ignored input
            bad = [n for n in outs + params + [m.group(2)] if n != "_" and not _is_ident(n)]
            if bad or len(set(params) - {"_"}) != len([a for a in params if a != "_"]):
                # Names are copied into Python source: refuse anything that isn't a plain name.
                self.opener(line, stmt, f"def _untranslated_function_L{line}(*args, **kwargs):",
                            False, "function", f"Unsupported function signature "
                            f"({', '.join(bad) or 'repeated input name'}).")
                return
            if params.count("_") > 1:
                params = [f"_{i}" if a == "_" else a for i, a in enumerate(params)]
            ctx.known_vars.update(params)
            self.opener(line, stmt, f"def {m.group(2)}({', '.join(params)}):", True, "function",
                        "Function definition.")
            self.blocks[-1]["outs"] = outs
        elif word == "end":
            if not self.blocks:
                self.emit(Step(None, stmt, line, Status.MANUAL, [f"# TODO: {stmt}"], [CONTROL_NOTE]))
                return
            block = self.blocks[-1]
            if block["kind"] == "switch":
                if block["in_case"]:
                    self.ensure_body()
                self.blocks.pop()
                return
            if block["kind"] == "function" and block.get("outs"):
                self.emit(Step(None, "(function outputs)", line, Status.EXACT,
                               [f"return {', '.join(block['outs'])}"], [], structural=True))
            self.ensure_body()
            self.blocks.pop()
        elif word in ("return", "break", "continue"):
            self.emit(Step(None, stmt, line, Status.EXACT, [word], [], structural=True))
        else:
            self.emit(Step(None, stmt, line, Status.MANUAL, [f"# TODO: {stmt}"], [CONTROL_NOTE]))

    def finish(self) -> list[Step]:
        while self.blocks:  # unterminated blocks (e.g. a function file without 'end')
            block = self.blocks[-1]
            if block["kind"] == "switch":
                if block["in_case"]:
                    self.ensure_body()
            else:
                self.ensure_body()
            self.blocks.pop()
        return self.steps


def _mark_superseded(steps: list[Step], set_path: str) -> None:
    """Skip steps that a later reload of the translated .set file makes moot.

    EEG.history often ends by reloading the dataset from disk. Everything
    before that reload is either already baked into the saved file or thrown
    away by the reload, so re-running it is redundant (and can fail when it
    points at files on the machine that created the dataset).
    """
    reloads = [i for i, s in enumerate(steps)
               if s.call is not None and s.call.func.lower() == "pop_loadset"
               and s.code and repr(set_path) in s.code[0]]
    if not reloads:
        return
    last = reloads[-1]
    line = steps[last].line
    for s in steps[:last]:
        if s.status is not Status.SKIPPED:
            s.status = Status.SKIPPED
            s.code = []
            s.notes = [f"Superseded: the dataset is reloaded from disk at line {line}, so this "
                       "step's effect is already in the saved file (or discarded by the reload)."]
    steps[last].notes.append("This is the dataset you translated. Steps above it are already "
                             "reflected in the saved file.")


def inspect_set(path: str | Path) -> dict:
    """Read what we need from an EEGLAB ``.set`` file.

    Returns ``{"path", "history", "epoched", "has_ica"}``.
    """
    import numpy as np
    from pymatreader import read_mat

    mat = read_mat(str(path))
    eeg = mat.get("EEG", mat)
    hist = eeg.get("history", "")
    if isinstance(hist, (list, tuple)):
        hist = "\n".join(str(h) for h in hist)
    trials = eeg.get("trials", 1)
    weights = eeg.get("icaweights")
    return {
        "path": str(path),
        "history": str(hist),
        "epoched": int(np.atleast_1d(trials)[0]) > 1 if trials is not None else False,
        "has_ica": weights is not None and np.size(weights) > 0,
    }


def history_from_set(path: str | Path) -> str:
    """Read the command history EEGLAB stores inside a ``.set`` file."""
    hist = inspect_set(path)["history"]
    if not hist.strip():
        raise ValueError(f"{path} has an empty EEG.history (was it created by a script that "
                         "did not record history?)")
    return hist


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_python(t: Translation) -> str:
    c = t.counts
    lines = [
        '"""MNE-Python pipeline generated by eeglab2mne ' + __version__ + '.',
        "",
        f"Source: {t.source_name}",
        f"Steps: {c[Status.EXACT]} exact, {c[Status.APPROX]} approximate, "
        f"{c[Status.MANUAL]} need manual work, {c[Status.SKIPPED]} skipped.",
        "",
        "Search for 'TODO' before running. See the migration report for why each",
        "step is marked the way it is.",
        '"""',
        "",
    ]
    lines += sorted(t.ctx.imports, key=lambda s: (not s.startswith("import"), s))
    lines += ["", ""]
    if t.ctx.uses_todo:
        lines += TODO_HELPER + ["", ""]
    for helper in t.ctx.helpers.values():
        lines += helper.split("\n") + ["", ""]
    for w in t.ctx.warnings:
        lines.append(f"# WARNING: {w}")
    if t.ctx.warnings:
        lines.append("")
    for s in t.steps:
        if not s.code:
            continue
        pad = "    " * s.indent
        if not s.structural:
            loc = f"L{s.line}: " if s.line else ""
            lines.append(f"{pad}# --- {loc}{_one_line(s.source)}  [{s.status.value}]")
        for chunk in s.code:
            lines += [pad + ln if ln else ln for ln in chunk.split("\n")]
        if not s.structural and not s.code[-1].rstrip().endswith(":"):
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report(t: Translation) -> str:
    c = t.counts
    total = sum(c.values()) - c[Status.SKIPPED]
    out = [
        "# EEGLAB → MNE migration report",
        "",
        f"**Source:** {_md_code(t.source_name)}  ",
        f"**ICA mode:** `{t.ctx.ica_mode}`  ",
        f"**Generated by:** eeglab2mne {__version__}",
        "",
        "## Summary",
        "",
        "| Status | Steps | Meaning |",
        "|---|---|---|",
        f"| ✅ exact | {c[Status.EXACT]} | Same operation; results should match up to floating-point error. |",
        f"| ⚠️ approximate | {c[Status.APPROX]} | Same intent, different implementation. Verify numerically. |",
        f"| ❌ manual | {c[Status.MANUAL]} | No safe automatic translation; a TODO is left in the code. |",
        f"| ➖ skipped | {c[Status.SKIPPED]} | EEGLAB bookkeeping, GUI or plotting. |",
        "",
    ]
    if total:
        pct = 100 * c[Status.EXACT] / total
        out += [f"{pct:.0f}% of processing steps translate exactly.", ""]
    if t.ctx.warnings:
        out += ["## ⚠️ Pipeline-level warnings", ""] + [f"- {_md(w)}" for w in t.ctx.warnings] + [""]

    out += ["## Step by step", "", "| # | Line | EEGLAB | Status | Notes |", "|---|---|---|---|---|"]
    for i, s in enumerate([s for s in t.steps if not s.structural], 1):
        notes = "<br>".join(_md(n) for n in s.notes)
        out.append(f"| {i} | {s.line or ''} | {_md_code(_one_line(s.source, 70))} | {_BADGE[s.status]} | {notes} |")
    out.append("")
    return "\n".join(out)


def _one_line(s: str, limit: int = 110) -> str:
    # Collapse whitespace outside string literals only ('S  1' must survive).
    out, in_str, prev_space = [], False, False
    for ch in s:
        if ch == "'":
            in_str = not in_str
        if not in_str and ch.isspace():
            if not prev_space:
                out.append(" ")
            prev_space = True
            continue
        out.append(ch)
        prev_space = False
    s = "".join(out).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _md(s: str) -> str:
    """Escape text for a Markdown table cell: no table breaks and no raw HTML.

    Text inside `code spans` is shown literally by Markdown, so only ``|`` is
    escaped there (entities like ``&lt;`` would show up verbatim).
    """
    parts = re.split(r"(`[^`]*`)", s)
    out = []
    for part in parts:
        if not (part.startswith("`") and part.endswith("`") and len(part) > 1):
            part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(part.replace("|", "\\|"))
    return "".join(out)


def _md_code(s: str) -> str:
    """Inline code span for ``s`` that survives backticks and pipes inside it."""
    s = s.replace("|", "\\|")
    fence = "`" * (max((len(m) for m in re.findall(r"`+", s)), default=0) + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"
