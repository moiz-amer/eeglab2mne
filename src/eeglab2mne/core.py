"""High-level API: EEGLAB script / .set history -> MNE script + audit report."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .parser import Call, parse_script
from .translators import ICA_REUSE_NOTE, Context, Status, Step, ica_reuse_code, translate_call

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
        return Counter(s.status for s in self.steps)

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
    steps: list[Step] = []
    for item in parse_script(text):
        if isinstance(item, Call):
            steps.append(translate_call(item, ctx))
        else:
            steps.append(Step(None, item, 0, Status.MANUAL, [f"# TODO: {item}"],
                              ["Not a function call (assignment, loop or expression); translate by hand."]))

    if set_info:
        _mark_superseded(steps, set_info["path"])

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
    for w in t.ctx.warnings:
        lines.append(f"# WARNING: {w}")
    if t.ctx.warnings:
        lines.append("")
    for s in t.steps:
        if s.status is Status.SKIPPED and not s.code:
            continue
        loc = f"L{s.line}: " if s.line else ""
        lines.append(f"# --- {loc}{_one_line(s.source)}  [{s.status.value}]")
        lines += s.code
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report(t: Translation) -> str:
    c = t.counts
    total = sum(c.values()) - c[Status.SKIPPED]
    out = [
        "# EEGLAB → MNE migration report",
        "",
        f"**Source:** `{t.source_name}`  ",
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
        out += ["## ⚠️ Pipeline-level warnings", ""] + [f"- {w}" for w in t.ctx.warnings] + [""]

    out += ["## Step by step", "", "| # | Line | EEGLAB | Status | Notes |", "|---|---|---|---|---|"]
    for i, s in enumerate(t.steps, 1):
        notes = "<br>".join(_md(n) for n in s.notes)
        out.append(f"| {i} | {s.line or ''} | `{_md(_one_line(s.source, 70))}` | {_BADGE[s.status]} | {notes} |")
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
    return s.replace("|", "\\|")
