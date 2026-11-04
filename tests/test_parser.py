import math

from eeglab2mne.parser import Expr, parse_script, parse_statement, parse_value, split_statements


def test_values():
    assert parse_value("'abc'") == "abc"
    assert parse_value("'it''s'") == "it's"
    assert parse_value("40") == 40 and isinstance(parse_value("40"), int)
    assert parse_value("0.5") == 0.5
    assert parse_value("1e-3") == 0.001
    assert math.isnan(parse_value("NaN"))
    assert parse_value("[]") == []
    assert parse_value("[1 2 3]") == [1, 2, 3]
    assert parse_value("[1, 2,3]") == [1, 2, 3]
    assert parse_value("[1 2; 3 4]") == [[1, 2], [3, 4]]
    assert parse_value("{'Fz' 'Cz'}") == ["Fz", "Cz"]
    assert parse_value("1:4") == [1, 2, 3, 4]
    assert parse_value("[1 3:5]") == [1, 3, 4, 5]
    assert parse_value("EEG.srate") == Expr("EEG.srate")


def test_strings_with_spaces_survive():
    # BrainVision markers are often 'S  1' with two spaces
    assert parse_value("{'S  1' 'S  2'}") == ["S  1", "S  2"]


def test_comments_continuations_and_multiple_statements():
    src = (
        "%{\nblock comment\n%}\n"
        "EEG = pop_eegfiltnew(EEG, 'locutoff', 1, ... % keep going\n"
        "    'hicutoff', 40); % 1-40 Hz\n"
        "EEG = eeg_checkset(EEG); EEG = pop_reref(EEG, []);\n"
        "disp('100% done');\n"
    )
    stmts = [s for _, s in split_statements(src)]
    assert len(stmts) == 4
    assert stmts[0].startswith("EEG = pop_eegfiltnew")
    assert "'hicutoff', 40" in stmts[0]
    assert stmts[3] == "disp('100% done')"  # % inside a string is not a comment


def test_kwarg_pairing():
    c = parse_statement("EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',40);")
    assert c.func == "pop_eegfiltnew"
    assert c.outputs == ["EEG"]
    assert c.positional == []
    assert c.get("LOCUTOFF") == 1 and c.get("hicutoff") == 40


def test_positional_then_kwargs():
    c = parse_statement("EEG = pop_epoch(EEG, {'A' 'B'}, [-0.2 0.8], 'newname', 'x', 'epochinfo', 'yes')")
    assert c.positional == [["A", "B"], [-0.2, 0.8]]
    assert c.get("newname") == "x"


def test_string_positional_not_mistaken_for_key():
    c = parse_statement("EEG = pop_reref(EEG, 'TP9', 'keepref', 'on')")
    assert c.positional == ["TP9"]
    assert c.get("keepref") == "on"


def test_multi_output_and_transpose():
    items = parse_script("[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'gui', 'off');\nx = EEG.data';")
    assert items[0].outputs == ["ALLEEG", "EEG", "CURRENTSET"]
    assert isinstance(items[1], str)  # not a call: left for a human


# --- shared scanner: double quotes, escapes, continuations ---------------------

from eeglab2mne.scanner import scan, split_top_level, strip_comment  # noqa: E402


def test_percent_inside_double_quotes_is_not_a_comment():
    assert strip_comment('f = sprintf("sub%02d.set", s); % note') == 'f = sprintf("sub%02d.set", s); '
    assert split_statements('a = "x;y"; b = 2') == [(1, 'a = "x;y"'), (1, "b = 2")]


def test_doubled_quotes_are_escapes():
    assert strip_comment("x = 'it''s 100%'; % c") == "x = 'it''s 100%'; "
    assert strip_comment('x = "say ""hi"" 5%"; % c') == 'x = "say ""hi"" 5%"; '
    assert parse_statement('f("a""b")').args == ['a"b']


def test_transpose_is_not_a_string():
    assert split_top_level("x', 'a,b'", ",") == ["x'", " 'a,b'"]
    assert [c for _, c, s in scan("a'+'b'") if s] == ["'", "b", "'"]


def test_text_after_continuation_is_a_comment():
    src = "x = 1 + ... add one\n    2;"
    assert split_statements(src) == [(1, "x = 1 +      2")]


def test_double_quoted_sprintf_translates():
    from eeglab2mne import translate
    from eeglab2mne.translators import Status

    t = translate('s = 3;\nfname = sprintf("sub%02d.set", s);')
    assert t.steps[-1].status is Status.EXACT
    ns = {}
    exec(t.to_python().replace("import mne\n", ""), ns)
    assert ns["fname"] == "sub03.set"
