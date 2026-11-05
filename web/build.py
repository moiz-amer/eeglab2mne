"""Assemble the browser demo into a static folder (default: ``_site``).

The site gets the current ``src/eeglab2mne`` package, so the demo always runs
the same code as the released tool. Usage::

    python web/build.py [output_dir]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PACKAGE = ROOT / "src" / "eeglab2mne"
EXAMPLES = ("erp_preprocessing.m", "subject_loop.m")


def build(out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    (out / "py" / "eeglab2mne").mkdir(parents=True)
    (out / "examples").mkdir()

    shutil.copy(WEB / "index.html", out / "index.html")
    files = ["demo_api.py"]
    shutil.copy(WEB / "demo_api.py", out / "py" / "demo_api.py")
    for src in sorted(PACKAGE.glob("*.py")):
        shutil.copy(src, out / "py" / "eeglab2mne" / src.name)
        files.append(f"eeglab2mne/{src.name}")
    for name in EXAMPLES:
        shutil.copy(ROOT / "examples" / name, out / "examples" / name)

    (out / "py" / "manifest.json").write_text(json.dumps({"files": files}, indent=1))
    (out / ".nojekyll").write_text("")  # serve files as-is on GitHub Pages
    return files


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
    print(f"Built {target} with {len(build(target))} Python files")
