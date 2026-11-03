"""Untrusted input (a .set history, someone else's script) must not inject code
into the generated Python, write outside the working folder unnoticed, or
inject markup into the report."""

import ast

import pytest

from eeglab2mne import translate
from eeglab2mne.cli import main
from eeglab2mne.translators import Status

SET_INFO = {"path": "data/sub01.set", "history": "", "epoched": False, "has_ica": False}


def _code(src, **kw):
    t = translate(src, **kw)
    code = t.to_python()
    ast.parse(code)  # always valid Python
    return t, code


def test_function_outputs_cannot_inject_code():
    t, code = _code("function [a, os.system('id')] = f(x)\n a = x;\nend")
    assert all("os.system" not in ln for ln in code.splitlines() if not ln.lstrip().startswith("#"))
    assert any(s.status is Status.MANUAL for s in t.steps)


@pytest.mark.parametrize("sig", [
    "function f(a=print)",
    "function [x] = f(import)",
    "function [y] = f(a, a)",
    "function [o.p] = f(a)",
])
def test_bad_function_signatures_are_manual(sig):
    t, code = _code(sig + "\n y = 1;\nend")
    assert t.steps[0].status is Status.MANUAL
    assert "def _untranslated_function_L1(*args, **kwargs):" in code


def test_ignored_inputs_become_placeholders():
    _, code = _code("function [y] = f(~, b, ~)\n y = b;\nend")
    assert "def f(_0, b, _2):" in code


def test_catch_variable_must_be_a_name():
    t, code = _code("try\n x = 1;\ncatch err.message\n x = 2;\nend")
    assert "except Exception:" in code
    assert "err.message" not in "\n".join(ln for ln in code.splitlines()
                                          if not ln.lstrip().startswith("#"))
    assert any(s.status is Status.MANUAL for s in t.steps)


def test_history_save_outside_working_folder_is_commented_out():
    src = ("EEG = pop_loadset('filename', 'sub01.set');\n"
           "EEG = pop_saveset(EEG, 'filename', '../../home/user/important.set');")
    t, code = _code(src, set_info=SET_INFO)
    assert "\nraw.save(" not in code
    assert "# raw.save('../../home/user/important-raw.fif'" in code
    assert any("important-raw.fif" in w for w in t.ctx.warnings)


def test_history_absolute_read_is_warned():
    src = "EEG = pop_loadset('filename', 'other.set', 'filepath', '/etc/');"
    t, _ = _code(src, set_info=SET_INFO)
    assert any("/etc/other.set" in w for w in t.ctx.warnings)


def test_superseded_history_paths_do_not_warn():
    src = ("EEG = pop_loadset('filename', 'raw.set', 'filepath', '/data/common/');\n"
           "EEG = pop_loadset('filename', 'sub01.set', 'filepath', '/data/common/');")
    t, _ = _code(src, set_info=SET_INFO)
    assert not t.ctx.warnings


def test_script_paths_are_trusted():
    # A script the user wrote may save wherever it likes.
    _, code = _code("EEG = pop_loadset('filename', 'a.set');\n"
                    "EEG = pop_saveset(EEG, 'filename', 'b.set', 'filepath', '/out');")
    assert "raw.save('/out/b-raw.fif', overwrite=True)" in code


def test_report_escapes_markup():
    rep = translate("EEG = pop_foo(EEG, '<img src=x onerror=alert(1)>`|');").to_report()
    row = next(ln for ln in rep.splitlines() if "pop_foo" in ln)
    # Source is shown in a code span that a backtick in it cannot close early,
    # so the HTML is displayed as text, and the pipe doesn't add a column.
    assert "| ``EEG = pop_foo(EEG, '<img src=x onerror=alert(1)>`\\|')`` |" in row
    assert row.count(" | ") == 4
    t = translate("EEG = pop_reref(EEG, []);")
    t.ctx.warnings.append("a <script> warning | with a pipe")
    assert "- a &lt;script&gt; warning \\| with a pipe" in t.to_report()


def test_cli_refuses_to_overwrite_input(tmp_path, capsys):
    src = tmp_path / "a.m"
    src.write_text("EEG = pop_reref(EEG, []);")
    assert main([str(src), "-o", str(src)]) == 1
    assert src.read_text() == "EEG = pop_reref(EEG, []);"
    assert "refusing" in capsys.readouterr().err


def test_cli_missing_file_is_a_clean_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope.m")]) == 1
    assert "no such file" in capsys.readouterr().err


def test_cli_empty_script_is_an_error(tmp_path):
    src = tmp_path / "empty.m"
    src.write_text("  \n")
    assert main([str(src)]) == 1
