import ast

import pytest

from eeglab2mne import translate
from eeglab2mne.translators import Status


def one(src, **kw):
    t = translate(src, **kw)
    steps = [s for s in t.steps if s.status is not Status.SKIPPED]
    assert len(steps) == 1, steps
    return steps[0], t


def test_filter_bandpass():
    s, _ = one("EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);")
    assert s.status is Status.APPROX
    assert "l_freq=1, h_freq=40" in s.code[0]


def test_filter_positional_with_order():
    s, _ = one("EEG = pop_eegfiltnew(EEG, 0.1, [], 3300);")
    assert "l_freq=0.1, h_freq=None" in s.code[0]
    assert "filter_length=3301" in s.code[0]  # EEGLAB order = taps - 1


def test_filter_revfilt_is_bandstop():
    s, _ = one("EEG = pop_eegfiltnew(EEG, 'locutoff', 48, 'hicutoff', 52, 'revfilt', 1);")
    assert "l_freq=52, h_freq=48" in s.code[0]


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
    assert "baseline=None" in s.code[-1]


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
