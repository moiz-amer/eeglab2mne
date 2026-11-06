"""Assemble the browser demo into a static folder (default: ``_site``).

The site gets the current ``src/eeglab2mne`` package, so the demo always runs
the same code as the released tool. Usage::

    python web/build.py [output_dir]
"""

from __future__ import annotations

import hashlib
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

    # A hash of the shipped code: the page appends it to every file URL, so a new
    # deploy is never served from a browser's cache of the previous one.
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.encode() + b"\0" + (out / "py" / f).read_bytes())
    manifest = {"version": digest.hexdigest()[:12], "files": files}
    (out / "py" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / ".nojekyll").write_text("")  # serve files as-is on GitHub Pages
    return files


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
    print(f"Built {target} with {len(build(target))} Python files")
