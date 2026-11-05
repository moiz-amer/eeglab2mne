"""Bridge between the browser demo and eeglab2mne.

The web page runs this module in Pyodide (Python compiled to WebAssembly), so
the translation happens entirely in the visitor's browser. It returns plain
JSON that the page renders; keeping the shaping here means it is covered by
the normal test suite instead of living in JavaScript.
"""

from __future__ import annotations

import json

from eeglab2mne import __version__, translate
from eeglab2mne.translators import Status

MAX_INPUT_CHARS = 200_000  # a pasted script, not a dataset


def run(source: str, ica_mode: str = "refit") -> str:
    """Translate MATLAB source and return the result as a JSON string."""
    if len(source) > MAX_INPUT_CHARS:
        return json.dumps({"error": f"The script is longer than {MAX_INPUT_CHARS:,} "
                                    "characters; use the command-line tool for files this big."})
    if not source.strip():
        return json.dumps({"error": "Paste an EEGLAB script to translate."})
    if ica_mode not in ("refit", "reuse"):
        ica_mode = "refit"

    t = translate(source, ica_mode=ica_mode, source_name="pasted script")
    counts = t.counts
    steps = [
        {"line": s.line, "source": s.source, "status": s.status.value, "notes": s.notes}
        for s in t.steps if not s.structural
    ]
    return json.dumps({
        "version": __version__,
        "code": t.to_python(),
        "report": t.to_report(),
        "counts": {st.value: counts[st] for st in Status},
        "warnings": list(t.ctx.warnings),
        "steps": steps,
    })
