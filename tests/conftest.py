"""Shared fixtures: a small synthetic EEGLAB dataset written to disk."""

from __future__ import annotations

import numpy as np
import pytest


def _make_raw(sfreq=500.0, seconds=60, seed=0):
    import mne

    rng = np.random.default_rng(seed)
    name = next(m for m in ("colin27_1020", "standard_1020") if m in mne.channels.get_builtin_montages())
    montage = mne.channels.make_standard_montage(name)
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "C3", "C4", "Cz",
                "P3", "P4", "Pz", "O1", "O2", "T7", "T8", "HEOG"]
    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    data = rng.normal(0, 10e-6, (len(ch_names), n))
    data += 5e-6 * np.sin(2 * np.pi * 10 * t)  # alpha
    data += 3e-6 * np.sin(2 * np.pi * 50 * t)  # line noise
    blink = np.zeros(n)
    for onset in rng.uniform(1, seconds - 1, 20):
        i = int(onset * sfreq)
        blink[i:i + 100] += 150e-6 * np.hanning(100)
    data[:2] += blink  # frontal blinks
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")
    onsets = np.arange(2, seconds - 2, 1.5)
    desc = ["S  1" if i % 2 else "S  2" for i in range(len(onsets))]
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), desc))
    return raw


@pytest.fixture(scope="session")
def eeglab_set(tmp_path_factory):
    """Path to a synthetic continuous EEGLAB .set file."""
    pytest.importorskip("mne")
    pytest.importorskip("eeglabio")
    import mne

    path = tmp_path_factory.mktemp("data") / "sub01.set"
    mne.export.export_raw(path, _make_raw(), fmt="eeglab", overwrite=True)
    return path


@pytest.fixture(scope="session")
def eeglab_set_with_ica(tmp_path_factory, eeglab_set):
    """Same data with an ICA decomposition, two ICs flagged, and a history.

    Mimics a dataset processed in EEGLAB by writing the fields EEGLAB uses
    (icaweights, icasphere, icawinv, icachansind, reject.gcompreject, history).
    The unmixing matrix is arbitrary; we only test that it is carried over.
    """
    import scipy.io

    mat = scipy.io.loadmat(eeglab_set, squeeze_me=False, struct_as_record=False)
    rng = np.random.default_rng(1)
    n_ch = 15  # every channel except HEOG
    weights = np.linalg.qr(rng.normal(size=(n_ch, n_ch)))[0]  # any full-rank unmixing
    sphere = np.eye(n_ch)
    winv = np.linalg.pinv(weights @ sphere)
    chansind = np.arange(1, n_ch + 1)  # 1-based, HEOG (16th) excluded

    out = {k: v for k, v in mat.items() if not k.startswith("__")}
    out["icaweights"] = weights
    out["icasphere"] = sphere
    out["icawinv"] = winv
    out["icachansind"] = chansind.astype(float)[None, :]
    gcomp = np.zeros((1, weights.shape[0]))
    gcomp[0, [0, 2]] = 1  # flag ICs 1 and 3 (EEGLAB numbering)
    out["reject"] = {"gcompreject": gcomp}
    out["history"] = np.array(
        "EEG = pop_loadset('filename','sub01.set','filepath','./');\n"
        "EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',40);\n"
        "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1);\n"
        "EEG = pop_subcomp( EEG, [1  3], 0);\n"
    )
    path = eeglab_set.parent / "sub01_ica.set"
    scipy.io.savemat(path, out, appendmat=False)
    return path
