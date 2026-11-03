import ast

import pytest

from eeglab2mne import translate
from eeglab2mne.translators import Status


def one(src, **kw):
    t = translate(src, **kw)
    steps = [s for s in t.steps if s.status is not Status.SKIPPED]
    assert len(steps) == 1, steps
    return steps[0], t


def test_filter_bandpass_is_exact_eeglab_design():
    s, _ = one("EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);")
    assert s.status is Status.EXACT
    call = s.code[-1]
    assert "l_freq=1, h_freq=40" in call
    assert "l_trans_bandwidth=_df, h_trans_bandwidth=_df" in call  # same band at both edges
    assert "pad='edge'" in call


def test_filter_positional_with_order():
    s, _ = one("EEG = pop_eegfiltnew(EEG, 0.1, [], 3300);")
    assert "_order = 3300" in "\n".join(s.code)
    assert "l_freq=0.1, h_freq=None" in s.code[-1]
    assert "filter_length=_order + 1" in s.code[-1]  # EEGLAB order = taps - 1


def test_filter_revfilt_is_bandstop():
    s, _ = one("EEG = pop_eegfiltnew(EEG, 'locutoff', 48, 'hicutoff', 52, 'revfilt', 1);")
    assert "l_freq=52, h_freq=48" in s.code[-1]


def test_legacy_eegfilt_stays_approximate():
    s, _ = one("EEG = pop_eegfilt(EEG, 1, 40);")
    assert s.status is Status.APPROX


def test_average_reference():
    s, _ = one("EEG = pop_reref(EEG, []);")
    assert s.status is Status.EXACT
    assert "set_eeg_reference('average'" in s.code[0]


def test_reference_drops_ref_unless_keepref():
    s, _ = one("EEG = pop_reref(EEG, {'TP9' 'TP10'});")
    assert any("drop_channels(ref_chs)" in c for c in s.code)
    s, _ = one("EEG = pop_reref(EEG, {'TP9' 'TP10'}, 'keepref', 'on');")
    assert not any("drop_channels" in c for c in s.code)


def test_channel_indices_are_zero_based():
    s, _ = one("EEG = pop_select(EEG, 'channel', [1:4]);")
    assert "raw.ch_names[0:4]" in s.code[0]


def test_baseline_ms_to_seconds():
    t = translate("EEG = pop_epoch(EEG, {'A'}, [-0.2 0.8]);\nEEG = pop_rmbase(EEG, [-200 0]);")
    assert "apply_baseline((-0.2, 0.0))" in t.steps[1].code[0]


def test_epoching_switches_object_and_skips_baseline():
    s, t = one("EEG = pop_epoch(EEG, {'S  1'}, [-0.2 0.8]);")
    assert t.ctx.obj == "epochs"
    assert "baseline=None" in "\n".join(s.code)
    assert s.status is Status.EXACT


def test_threshold_uses_microvolts_not_peak_to_peak():
    t = translate("EEG = pop_epoch(EEG, {}, [-1 2]);\nEEG = pop_eegthresh(EEG, 1, [1:3], -75, 75, -1, 1.996, 0, 1);")
    code = "\n".join(t.steps[1].code)
    assert "* 1e6" in code and "reject=" not in code


def test_subcomp_after_refit_is_manual():
    t = translate("EEG = pop_runica(EEG, 'extended', 1);\nEEG = pop_subcomp(EEG, [1 2], 0);")
    assert t.steps[0].status is Status.APPROX
    assert t.steps[1].status is Status.MANUAL
    assert t.ctx.warnings


def test_subcomp_after_reuse_is_exact_and_zero_based():
    t = translate("EEG = pop_runica(EEG, 'extended', 1);\nEEG = pop_subcomp(EEG, [1 3], 0);",
                  ica_mode="reuse", ica_source="x.set")
    assert t.steps[0].status is Status.EXACT
    assert t.steps[1].status is Status.EXACT
    assert "ica.exclude = [0, 2]" in t.steps[1].code[0]


def test_unknown_function_is_manual_not_dropped():
    s, _ = one("EEG = pop_my_custom_thing(EEG, 3);")
    assert s.status is Status.MANUAL
    assert "TODO" in s.code[0]


def test_clean_rawdata_is_flagged_with_mapping():
    s, _ = one("EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'BurstCriterion', 20);")
    assert s.status is Status.MANUAL
    joined = "\n".join(s.code)
    assert "asrpy" in joined and "flatlinecriterion=5" in joined


@pytest.mark.parametrize("mode", ["refit", "reuse"])
def test_example_script_generates_valid_python(mode):
    from pathlib import Path

    src = (Path(__file__).parent.parent / "examples" / "erp_preprocessing.m").read_text()
    t = translate(src, ica_mode=mode)
    ast.parse(t.to_python())
    report = t.to_report()
    assert "## Summary" in report and "pop_clean_rawdata" in report


def test_plotting_is_skipped_not_manual():
    t = translate("figure; pop_topoplot(EEG, 0, 1, 'x', [1 1], 0, 'electrodes', 'on');")
    assert all(s.status is Status.SKIPPED for s in t.steps)


def test_chanedit_types_and_fields_in_order():
    src = ("EEG = pop_chanedit(EEG, 'settype', {'1:32', 'EEG'}, "
           "'changefield', {2, 'type', 'EOG'}, 'changefield', {6, 'type', 'EOG'}, "
           "'changefield', {1, 'labels', 'Fp1'});")
    s, _ = one(src)
    assert s.status is Status.EXACT
    assert "raw.ch_names[0:32]" in s.code[0]
    assert "raw.ch_names[1]: 'eog'" in s.code[1]   # both changefields kept, in order
    assert "raw.ch_names[5]: 'eog'" in s.code[2]
    assert "rename_channels({raw.ch_names[0]: 'Fp1'})" in s.code[3]


def test_chanedit_display_only_is_skipped():
    t = translate("EEG.chanlocs = pop_chanedit(EEG.chanlocs, 'plotrad', 0.5);")
    assert t.steps[0].status is Status.SKIPPED


def test_set_info_reads_epochs_and_loads_prior_ica():
    info = {"path": "/my/data/sub01.set", "epoched": True, "has_ica": True}
    hist = ("EEG = pop_chanedit(EEG, 'load', {'/other/machine/locs.ced'});\n"
            "EEG = pop_loadset('filename', 'sub01.set', 'filepath', '/other/machine/');")
    t = translate(hist, ica_mode="reuse", ica_source=info["path"], set_info=info)
    assert t.steps[0].status is Status.SKIPPED            # superseded by the reload
    assert "mne.read_epochs_eeglab('/my/data/sub01.set')" in t.steps[1].code[0]
    assert t.steps[-1].source.startswith("(ICA")           # ICA appended
    assert t.steps[-1].status is Status.EXACT


def test_prior_ica_warns_in_refit_mode():
    info = {"path": "x.set", "epoched": False, "has_ica": True}
    t = translate("EEG = eeg_checkset(EEG);", set_info=info)
    assert any("--ica reuse" in w for w in t.ctx.warnings)


def test_runica_maxsteps_becomes_max_iter():
    s, _ = one("EEG = pop_runica(EEG, 'extended', 1, 'maxsteps', 300);")
    assert "max_iter=300" in s.code[0]


LOOP_SCRIPT = r"""
subjects = {'sub01', 'sub02'};
for s = 1:length(subjects)
    EEG = pop_loadset('filename', [subjects{s} '.set'], 'filepath', datapath);
    EEG = pop_biosig(fullfile(rawpath, [subjects{s} '.bdf']));
    EEG = pop_runica(EEG, 'extended', 1);
    EEG = pop_saveset(EEG, 'filename', [subjects{s} '_clean.set'], 'filepath', outpath);
end
"""


@pytest.mark.parametrize("mode", ["refit", "reuse"])
def test_variable_paths_still_generate_valid_python(mode):
    # Regression: a '# TODO' inside an expression used to comment out the rest of the line.
    ast.parse(translate(LOOP_SCRIPT, ica_mode=mode).to_python())


def test_variable_paths_are_manual_not_exact():
    t = translate(LOOP_SCRIPT)
    by_func = {s.func: s for s in t.steps if s.call}
    for func in ("pop_loadset", "pop_biosig", "pop_saveset"):
        assert by_func[func].status is Status.MANUAL, func
    assert "INPUT_SET = None  # TODO" in by_func["pop_loadset"].code[0]


def test_saveset_with_variable_name_does_not_use_a_fixed_file():
    # Regression: every subject in a loop used to be saved to the same 'output-raw.fif'.
    t = translate(LOOP_SCRIPT)
    save = next(s for s in t.steps if s.call and s.func == "pop_saveset")
    assert "'output-" not in "\n".join(save.code)
    assert "save(OUTPUT_FILE" in "\n".join(save.code)


def test_saveset_keeps_folder_and_resave_uses_loaded_name():
    t = translate("EEG = pop_loadset('filename', 'a.set', 'filepath', 'data/');\n"
                  "EEG = pop_saveset(EEG, 'filename', 'b.set', 'filepath', 'out/');\n"
                  "EEG = pop_saveset(EEG, 'savemode', 'resave');")
    assert "'out/b-raw.fif'" in t.steps[1].code[0]
    assert "'data/a-raw.fif'" in t.steps[2].code[0]


def test_loop_and_conditions_become_python_blocks():
    src = ("subjects = {'a', 'b'};\n"
           "for s = 1:length(subjects)\n"
           "  if strcmp(subjects{s}, 'a')\n"
           "    EEG = pop_reref(EEG, []);\n"
           "  elseif s > 1\n"
           "    EEG = pop_resample(EEG, 250);\n"
           "  else\n"
           "    disp('x');\n"
           "  end\n"
           "end\n")
    code = translate(src).to_python()
    ast.parse(code)
    assert "for s in range(1, len(subjects) + 1):" in code
    assert "    if (subjects[s - 1] == 'a'):" in code
    assert "    elif s > 1:" in code
    assert "        pass" in code  # empty else-branch stays valid Python


def test_switch_becomes_if_chain():
    src = ("cond = 'rest';\nswitch cond\n  case 'rest'\n    EEG = pop_reref(EEG, []);\n"
           "  case {'task', 'oddball'}\n    EEG = pop_resample(EEG, 250);\n"
           "  otherwise\n    disp('?');\nend\n")
    code = translate(src).to_python()
    ast.parse(code)
    assert "if cond == 'rest':" in code
    assert "elif cond in ['task', 'oddball']:" in code
    assert "else:" in code


def test_untranslatable_condition_fails_loudly_not_silently():
    # An unknown variable must not be guessed; the script stops with a clear error instead.
    t = translate("if someWorkspaceVar > 3\n  EEG = pop_reref(EEG, []);\nend\n")
    code = t.to_python()
    ast.parse(code)
    assert "if TODO('someWorkspaceVar > 3'):" in code
    assert "def TODO(what):" in code
    assert t.steps[0].status is Status.MANUAL


def test_unknown_variable_in_path_is_not_assumed():
    t = translate("EEG = pop_loadset('filename', 'a.set', 'filepath', datapath);")
    assert t.steps[0].status is Status.MANUAL  # datapath only exists in someone's workspace


def test_expressions_in_paths():
    t = translate("subjects = {'s1'};\ns = 1;\n"
                  "EEG = pop_loadset('filename', sprintf('%s_task.set', subjects{s}), "
                  "'filepath', fullfile('data', 'raw'));")
    step = t.steps[-1]
    assert step.status is Status.EXACT
    assert ("os.path.join(os.path.join('data', 'raw'), ('%s_task.set' % (subjects[s - 1],)))"
            in step.code[0])


def test_unexpected_literal_arguments_are_reported_not_hidden():
    t = translate("EEG = pop_subcomp(EEG, {'a' 'b'}, 0);")
    assert t.steps[0].status is Status.MANUAL
    assert any("unexpected arguments to `pop_subcomp`" in w for w in t.ctx.warnings)


@pytest.mark.parametrize("name", ["erp_preprocessing.m", "subject_loop.m"])
def test_examples_hit_no_unexpected_arguments(name):
    from pathlib import Path
    src = (Path(__file__).parent.parent / "examples" / name).read_text()
    for mode in ("refit", "reuse"):
        assert not [w for w in translate(src, ica_mode=mode).ctx.warnings if "unexpected" in w]
