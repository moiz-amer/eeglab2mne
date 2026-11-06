"""The browser demo: its Python bridge and the static-site build."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

import build  # noqa: E402
import demo_api  # noqa: E402


def run(src, mode="refit"):
    return json.loads(demo_api.run(src, mode))


def test_translates_example_like_the_cli():
    from eeglab2mne import translate

    src = (ROOT / "examples" / "erp_preprocessing.m").read_text()
    r = run(src)
    t = translate(src, source_name="pasted script")
    assert r["code"] == t.to_python()
    assert r["counts"] == {"exact": 11, "approximate": 4, "manual": 2, "skipped": 4}
    assert r["steps"][0] == {"line": 2, "source": "[ALLEEG EEG CURRENTSET ALLCOM] = eeglab",
                             "status": "skipped",
                             "notes": ["EEGLAB session/GUI bookkeeping; MNE needs no equivalent."]}


def test_reuse_mode_and_bad_mode():
    src = (ROOT / "examples" / "erp_preprocessing.m").read_text()
    assert "read_ica_eeglab" in run(src, "reuse")["code"]
    assert run(src, "nonsense")["code"] == run(src, "refit")["code"]


@pytest.mark.parametrize("src, msg", [("", "Paste"), ("  \n", "Paste"),
                                      ("x" * (demo_api.MAX_INPUT_CHARS + 1), "longer")])
def test_rejects_empty_and_huge_input(src, msg):
    assert msg in run(src)["error"]


def test_build_ships_every_package_module(tmp_path):
    files = build.build(tmp_path / "site")
    package = sorted(p.name for p in (ROOT / "src" / "eeglab2mne").glob("*.py"))
    assert sorted(f.split("/")[-1] for f in files if f.startswith("eeglab2mne/")) == package
    manifest = json.loads((tmp_path / "site" / "py" / "manifest.json").read_text())
    assert manifest["files"] == files
    for f in files:
        assert (tmp_path / "site" / "py" / f).is_file()
    html = (tmp_path / "site" / "index.html").read_text()
    for example in build.EXAMPLES:
        assert f'value="examples/{example}"' in html
        assert (tmp_path / "site" / "examples" / example).is_file()


def test_warnings_point_to_the_ica_menu_not_the_cli_flag():
    src = (ROOT / "examples" / "erp_preprocessing.m").read_text()
    warnings = run(src)["warnings"]
    assert any("set ICA to “reuse EEGLAB's”" in w for w in warnings)
    assert not any("--ica" in w for w in warnings)


def test_manifest_version_changes_with_the_code(tmp_path, monkeypatch):
    build.build(tmp_path / "a")
    first = json.loads((tmp_path / "a" / "py" / "manifest.json").read_text())["version"]
    build.build(tmp_path / "b")
    assert json.loads((tmp_path / "b" / "py" / "manifest.json").read_text())["version"] == first

    changed = tmp_path / "web"
    changed.mkdir()
    for name in ("index.html", "demo_api.py"):
        (changed / name).write_text((build.WEB / name).read_text())
    (changed / "demo_api.py").write_text((changed / "demo_api.py").read_text() + "\n# edit\n")
    monkeypatch.setattr(build, "WEB", changed)
    build.build(tmp_path / "c")
    assert json.loads((tmp_path / "c" / "py" / "manifest.json").read_text())["version"] != first
    assert "?v=${manifest.version}" in (tmp_path / "c" / "index.html").read_text()
