"""Command-line interface.

Examples
--------
Translate a script::

    eeglab2mne preprocess.m -o preprocess_mne.py --report report.md

Translate the history stored in a processed dataset, reusing its ICA::

    eeglab2mne sub01_clean.set --ica reuse -o sub01_mne.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import history_from_set, translate
from .translators import Status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eeglab2mne",
        description="Translate an EEGLAB script or .set history into MNE-Python, "
                    "with a report of which steps are exact, approximate or manual.")
    ap.add_argument("input", help="EEGLAB .m script, a text file of EEG.history, or a .set dataset")
    ap.add_argument("-o", "--output", help="write the generated Python here (default: stdout)")
    ap.add_argument("-r", "--report", help="write the Markdown migration report here")
    ap.add_argument("--ica", choices=["refit", "reuse"], default="refit",
                    help="refit ICA in MNE (default), or reuse EEGLAB's decomposition from the .set")
    ap.add_argument("--strict", action="store_true",
                    help="exit with code 2 if any step needs manual translation (useful in CI)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    path = Path(args.input)
    if path.suffix.lower() == ".set":
        text = history_from_set(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    # A processed .set holds its own ICA, so that is where reuse mode reads it.
    ica_source = str(path) if path.suffix.lower() == ".set" else None
    result = translate(text, ica_mode=args.ica, source_name=path.name, ica_source=ica_source)

    code = result.to_python()
    if args.output:
        Path(args.output).write_text(code, encoding="utf-8")
    else:
        sys.stdout.write(code)
    if args.report:
        Path(args.report).write_text(result.to_report(), encoding="utf-8")

    c = result.counts
    print(f"eeglab2mne: {c[Status.EXACT]} exact, {c[Status.APPROX]} approximate, "
          f"{c[Status.MANUAL]} manual, {c[Status.SKIPPED]} skipped", file=sys.stderr)
    return 2 if (args.strict and c[Status.MANUAL]) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
