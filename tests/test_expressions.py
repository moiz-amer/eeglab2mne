"""MATLAB expressions must translate to Python that computes the same value."""

import math

import pytest

from eeglab2mne import translate
from eeglab2mne.matlab_expr import (MATLAB_COLON, MATLAB_NUM2STR, MATLAB_ROUND, ExprEnv,
                                    ExprError, range_to_python, to_python)


def run(expr, **vars):
    env = ExprEnv(known=set(vars))
    code = to_python(expr, env)
    ns = dict(vars)
    exec("import math, os\nimport numpy as np\n" + "\n".join(env.helpers.values()), ns)
    return eval(code, ns)


@pytest.mark.parametrize("expr, expected", [
    ("2^3^2", 64),            # MATLAB's ^ is left-associative
    ("-2^2", -4),             # ...and binds tighter than unary minus
    ("2^-1", 0.5),
    ("[1 + 2] * 3", 9),       # brackets around one element still group
    ("round(2.5)", 3),        # half away from zero, not to even
    ("round(-2.5)", -3),
    ("int2str(0.5)", "1"),
    ("floor(2.7) + ceil(2.1)", 5),
    ("num2str(2.0)", "2"),
    ("num2str(pi)", "3.1416"),
    ("num2str(123.456)", "123.456"),
    ("num2str(0.001234567)", "0.0012346"),
    ("mod(-7, 3)", 2),
    ("mod(5 + 2, 3)", 1),
])
def test_values_match_matlab(expr, expected):
    assert run(expr) == expected


def test_string_functions_keep_grouping():
    assert run("upper([a 'b'])", a="x") == "XB"
    assert run("lower([a 'B'])", a="X") == "xb"
    assert run("strtrim([a ' '])", a=" x") == "x"
    assert run("strrep([a 'b'], 'b', 'c')", a="a") == "ac"
    assert run("strcmpi([a 'B'], 'xb')", a="X") is True
    assert run("sprintf([a '%d'], 3)", a="s") == "s3"
    assert run("[num2str(s) num2str(t)]", s=1, t=2) == "12"


def test_sprintf_interprets_escapes():
    assert run(r"sprintf('a\tb\n')") == "a\tb\n"
    assert run(r"sprintf('100%% %s\\x', 'done')") == "100% done\\x"


def test_exist_returns_matlabs_codes(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("")
    assert run("exist(p, 'file') == 2", p=str(f)) is True
    assert run("exist(p, 'file')", p=str(tmp_path)) == 7
    assert run("exist(p, 'dir')", p=str(tmp_path)) == 7
    assert run("exist(p)", p=str(tmp_path / "nope")) == 0
    with pytest.raises(ExprError):
        to_python("exist('x', 'var')", ExprEnv(known=set()))


@pytest.mark.parametrize("rng, expected", [
    ("1:3", [1, 2, 3]),
    ("10:-2:1", [10, 8, 6, 4, 2]),
    ("10:-n:1", [10, 7, 4, 1]),
    ("1:n:7", [1, 4, 7]),
    ("0:0.5:2", [0, 0.5, 1.0, 1.5, 2.0]),
    ("0.5:2", [0.5, 1.5]),
    ("3:1", []),
    ("1:0:3", []),
])
def test_ranges_match_matlab(rng, expected):
    env = ExprEnv(known={"n"})
    code = range_to_python(rng, env)
    ns = {"n": 3}
    exec("import math\n" + "\n".join(env.helpers.values()), ns)
    assert list(eval(code, ns)) == expected


def test_failed_expression_leaves_no_imports_or_helpers():
    env = ExprEnv(known=set())
    with pytest.raises(ExprError):
        to_python("[num2str(1) undefined_var]", env)
    assert not env.imports and not env.helpers


def test_generated_loops_run():
    src = "n = 2;\ntotal = 0;\nfor k = 10:-n:1\n total = total + round(k / 4);\nend"
    ns = {}
    code = translate(src).to_python().replace("import mne\n", "")
    exec(code, ns)
    assert ns["total"] == 3 + 2 + 2 + 1 + 1  # round(2.5)=3, 2, 1.5->2, 1, 0.5->1


def test_lit_inf_gets_numpy_import():
    code = translate("EEG = pop_loadset('filename', 'a.set');\n"
                     "EEG = pop_select(EEG, 'time', [0 Inf]);").to_python()
    assert "np.inf" in code and "import numpy as np" in code


def test_helpers_are_valid_python():
    for src in (MATLAB_ROUND, MATLAB_NUM2STR, MATLAB_COLON):
        compile(src, "<helper>", "exec")
    assert math  # imported for the helpers' namespace in run()
