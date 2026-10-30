"""pop_eegfiltnew translations must reproduce EEGLAB's filter exactly.

``eeglab_eegfiltnew`` below is a line-by-line NumPy port of EEGLAB's
firfilt plugin (pop_eegfiltnew.m, firws.m, firfilt.m), kept independent of
the translator. Each test runs the *generated* MNE code and compares.
The README also reports a comparison against real EEGLAB output.
"""

import math

import numpy as np
import pytest

from eeglab2mne import translate

mne = pytest.importorskip("mne")
from scipy.signal import firwin  # noqa: E402


def _fkernel(m, f, w):
    """firws.m fkernel: windowed sinc normalised to unity DC gain."""
    x = np.arange(-m / 2, m / 2 + 1)
    b = np.where(x == 0, 2 * np.pi * f, np.sin(2 * np.pi * f * x) / np.where(x == 0, 1, x))
    b = b * w
    return b / b.sum()


def _fspecinv(b):
    b = -b.copy()
    b[(len(b) - 1) // 2] += 1
    return b


def eeglab_eegfiltnew(data, fs, locutoff=None, hicutoff=None, filtorder=None, revfilt=False,
                      boundaries=()):
    """NumPy port of EEGLAB pop_eegfiltnew + firws + firfilt (data in rows)."""
    if locutoff == 0:
        locutoff = None
    if hicutoff == 0:
        hicutoff = None
    if hicutoff is None:  # highpass -> inverted lowpass
        hicutoff, locutoff, revfilt = locutoff, None, not revfilt
    edges = np.sort([e for e in (locutoff, hicutoff) if e is not None]).astype(float)
    nyq = fs / 2
    max_tbw = edges.copy()
    if not revfilt:
        max_tbw[-1] = nyq - edges[-1]
    elif len(edges) == 2:
        max_tbw = np.array([np.diff(edges)[0] / 2])
    max_df = max_tbw.min()
    if filtorder is None:
        df = min(max(max_df * 0.25, 2), max_df) if revfilt else min(max(edges[0] * 0.25, 2), max_df)
        m = math.ceil(3.3 / (df / fs) / 2) * 2
    else:
        m = filtorder
    df = 3.3 / m * fs
    shift = {(False, 1): [df], (False, 2): [-df, df], (True, 1): [-df], (True, 2): [df, -df]}
    cutoff = edges + np.array(shift[(bool(revfilt), len(edges))]) / 2
    n = m + 1
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    f = cutoff / nyq / 2
    b = _fkernel(m, f[0], w)
    if len(f) == 1 and revfilt:
        b = _fspecinv(b)
    if len(f) == 2:
        b = b + _fspecinv(_fkernel(m, f[1], w))
        if not revfilt:
            b = _fspecinv(b)
    # firfilt.m: filter each boundary-delimited segment, padding with edge samples
    g = (len(b) - 1) // 2
    out = np.empty_like(data)
    cuts = [0, *boundaries, data.shape[1]]
    for a, z in zip(cuts[:-1], cuts[1:]):
        seg = data[:, a:z]
        pad = np.concatenate([np.repeat(seg[:, :1], g, 1), seg, np.repeat(seg[:, -1:], g, 1)], 1)
        out[:, a:z] = np.array([np.convolve(x, b, mode="valid") for x in pad])
    return out


def _raw(fs=256.0, seconds=40, boundaries=()):
    rng = np.random.default_rng(3)
    data = rng.normal(0, 20e-6, (4, int(fs * seconds)))
    data += np.linspace(0, 50e-6, data.shape[1])  # drift, so edges matter
    raw = mne.io.RawArray(data, mne.create_info(4, fs, "eeg"), verbose=False)
    if boundaries:
        raw.set_annotations(mne.Annotations([b / fs for b in boundaries], [0] * len(boundaries),
                                            ["boundary"] * len(boundaries)))
    return raw


def _run(src, raw):
    step = translate(src).steps[0]
    ns = {"raw": raw, "np": np, "mne": mne}
    exec("\n".join(step.code), ns)
    return step, ns["raw"].get_data()


@pytest.mark.parametrize("src, kw", [
    ("pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 30)", dict(locutoff=1, hicutoff=30)),
    ("pop_eegfiltnew(EEG, 0.5, [])", dict(locutoff=0.5)),
    ("pop_eegfiltnew(EEG, [], 40)", dict(hicutoff=40)),
    ("pop_eegfiltnew(EEG, 48, 52, [], 1)", dict(locutoff=48, hicutoff=52, revfilt=True)),
    ("pop_eegfiltnew(EEG, 'hicutoff', 40, 'filtorder', 166)", dict(hicutoff=40, filtorder=166)),
    ("pop_eegfiltnew(EEG, 20, [], [], 1)", dict(locutoff=20, revfilt=True)),
])
def test_matches_eeglab_filter(src, kw):
    raw = _raw()
    expected = eeglab_eegfiltnew(raw.get_data(), raw.info["sfreq"], **kw)
    step, got = _run("EEG = " + src + ";", raw)
    assert step.status.value == "exact"
    assert np.abs(got - expected).max() < 1e-12 * np.abs(expected).max() + 1e-18


def test_restarts_at_boundary_events():
    cut = 256 * 15
    raw = _raw(boundaries=[cut])
    expected = eeglab_eegfiltnew(raw.get_data(), 256.0, locutoff=1, boundaries=[cut])
    _, got = _run("EEG = pop_eegfiltnew(EEG, 1, []);", raw)
    assert np.abs(got - expected).max() < 1e-12 * np.abs(expected).max()


def test_port_matches_scipy_firwin():
    """Sanity check of the port itself against an independent implementation."""
    fs, m, c = 128.0, 424, 30.5
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(m + 1) / m)
    assert np.allclose(_fkernel(m, c / (fs / 2) / 2, w), firwin(m + 1, c, window="hamming", fs=fs))
