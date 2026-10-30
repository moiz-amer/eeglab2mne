"""Run generated pipelines on real (synthetic) EEGLAB files.

These are the tests that matter most: a translation that parses but fails at
runtime, or silently does the wrong thing, is worse than no translation.
"""

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from eeglab2mne import translate
from eeglab2mne.cli import main

pytest.importorskip("mne")


def _run(code: str, workdir: Path) -> dict:
    script = workdir / "generated_pipeline.py"
    script.write_text(code)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        return runpy.run_path(str(script))
    finally:
        os.chdir(cwd)


def test_refit_pipeline_runs(eeglab_set, tmp_path):
    pytest.importorskip("mne_icalabel")
    src = f"""
EEG = pop_loadset('filename', '{eeglab_set.name}', 'filepath', '{eeglab_set.parent}');
EEG = pop_chanedit(EEG, 'lookup', 'standard_1005.elc');
EEG = pop_select(EEG, 'nochannel', {{'HEOG'}});
EEG = pop_resample(EEG, 250);
EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 100);
EEG = pop_cleanline(EEG, 'linefreqs', 50);
EEG = pop_reref(EEG, []);
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'pca', 14, 'maxsteps', 50);
EEG = pop_iclabel(EEG, 'default');
EEG = pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);
EEG = pop_subcomp(EEG, [], 0);
EEG = pop_epoch(EEG, {{'S  1' 'S  2'}}, [-0.2 0.8]);
EEG = pop_rmbase(EEG, [-200 0]);
EEG = pop_eegthresh(EEG, 1, 1:15, -500, 500, -0.2, 0.796, 0, 1);
EEG = pop_saveset(EEG, 'filename', 'out.set', 'filepath', './');
"""
    t = translate(src)
    assert not [s for s in t.steps if s.status.value == "manual"]
    ns = _run(t.to_python(), tmp_path)

    epochs = ns["epochs"]
    assert set(epochs.event_id) == {"S  1", "S  2"}
    assert len(epochs) > 10
    assert epochs.info["sfreq"] == 250
    assert "HEOG" not in epochs.ch_names
    assert ns["ic_proba"].shape == (14, 7)
    assert ns["ica"].max_iter == 50  # 'maxsteps' carried over
    np.testing.assert_allclose(ns["ic_proba"].sum(axis=1), 1, atol=1e-4)
    # baseline really is zero-mean over -200..0 ms
    base = epochs.copy().crop(-0.2, 0).get_data().mean(axis=-1)
    assert np.abs(base).max() < 1e-9
    assert (tmp_path / "out-epo.fif").exists()


def test_reuse_from_set_history(eeglab_set_with_ica, tmp_path):
    out = tmp_path / "from_history.py"
    report = tmp_path / "report.md"
    rc = main([str(eeglab_set_with_ica), "--ica", "reuse", "-o", str(out), "-r", str(report)])
    assert rc == 0
    code = out.read_text()
    assert "read_ica_eeglab" in code
    assert "ica.exclude = [0, 2]" in code  # EEGLAB ICs 1 and 3

    # The history loads './sub01.set'; run where that file lives.
    ns = _run(code, eeglab_set_with_ica.parent)
    ica = ns["ica"]
    assert ica.n_components_ == 15
    assert ica.exclude == [0, 2]
    # gcompreject was read from the file before pop_subcomp overrode it
    assert list(np.flatnonzero(ns["_flags"])) == [0, 2]
    assert "✅ exact" in report.read_text()


def test_strict_mode_fails_on_manual_steps(tmp_path):
    m = tmp_path / "s.m"
    m.write_text("EEG = pop_clean_rawdata(EEG, 'BurstCriterion', 20);")
    assert main([str(m), "-o", str(tmp_path / "o.py"), "--strict"]) == 2
    assert main([str(m), "-o", str(tmp_path / "o.py")]) == 0
