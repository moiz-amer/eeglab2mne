"""Validation against EEGLAB's own tutorial dataset.

Downloads ``eeglab_data_epochs_ica.set`` from the EEGLAB GitHub repository
(skipped when offline). This file was produced by EEGLAB itself, with an ICA
computed in EEGLAB, so it checks the translation against the real thing rather
than against data we made up.
"""

import runpy
import urllib.request

import numpy as np
import pytest

from eeglab2mne.cli import main

BASE = "https://raw.githubusercontent.com/sccn/eeglab/develop/sample_data/"
FILES = ("eeglab_data_epochs_ica.set", "eeglab_data_epochs_ica.fdt")

pytest.importorskip("mne")


@pytest.fixture(scope="session")
def tutorial_set(tmp_path_factory):
    folder = tmp_path_factory.mktemp("eeglab_tutorial")
    try:
        for name in FILES:
            urllib.request.urlretrieve(BASE + name, folder / name)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not download EEGLAB sample data: {exc}")
    return folder / FILES[0]


def test_history_translation_runs_on_eeglab_tutorial(tutorial_set, tmp_path):
    out, report = tmp_path / "tutorial.py", tmp_path / "tutorial.md"
    assert main([str(tutorial_set), "--ica", "reuse", "-o", str(out), "-r", str(report)]) == 0
    code = out.read_text()
    assert "read_epochs_eeglab" in code          # epoched file read as Epochs
    assert "/data/common" not in code            # original machine's paths were not kept
    assert "read_ica_eeglab" in code             # ICA from before the history was loaded

    ns = runpy.run_path(str(out))
    epochs, ica = ns["epochs"], ns["ica"]
    assert len(epochs) == 80 and len(epochs.ch_names) == 32
    assert epochs.info["sfreq"] == 128
    assert ica.n_components_ == 32
    assert "❌ manual | 0" in report.read_text()


def test_reused_ica_matches_eeglab_numerically(tutorial_set):
    """MNE with EEGLAB's ICA must reproduce EEGLAB's own computations."""
    import mne
    from pymatreader import read_mat

    mne.set_log_level("ERROR")
    mat = read_mat(str(tutorial_set))
    eeg = mat.get("EEG", mat)
    epochs = mne.read_epochs_eeglab(tutorial_set)
    ica = mne.preprocessing.read_ica_eeglab(tutorial_set)

    data = epochs.get_data() * 1e6  # EEGLAB works in µV
    ch = np.atleast_1d(eeg["icachansind"]).astype(int) - 1
    unmix = eeg["icaweights"] @ eeg["icasphere"]
    eeglab_act = np.einsum("ij,ejt->eit", unmix, data[:, ch, :])
    mne_act = ica.get_sources(epochs).get_data()
    for k in range(unmix.shape[0]):
        r = np.corrcoef(mne_act[:, k].ravel(), eeglab_act[:, k].ravel())[0, 1]
        assert r > 1 - 1e-9, f"component {k} differs (r={r})"

    # Removing components (pop_subcomp) gives the same cleaned data
    excl = [0, 3]
    ica.exclude = excl
    mne_clean = ica.apply(epochs.copy()).get_data() * 1e6
    keep = [k for k in range(unmix.shape[0]) if k not in excl]
    eeglab_clean = data.copy()
    eeglab_clean[:, ch, :] = np.einsum("ij,ejt->eit", eeg["icawinv"][:, keep], eeglab_act[:, keep, :])
    rel = np.abs(mne_clean - eeglab_clean).max() / np.abs(eeglab_clean).max()
    assert rel < 1e-10
