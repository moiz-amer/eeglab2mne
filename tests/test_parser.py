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
