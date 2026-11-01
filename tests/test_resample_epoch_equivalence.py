"""pop_resample and pop_epoch translations must reproduce EEGLAB.

The reference functions below are NumPy ports of EEGLAB's own code, kept
independent of the translator:

* ``eeglab_resample``: pop_resample.m ``myresample`` (Signal Processing Toolbox
  branch): firfilt Kaiser anti-aliasing kernel, edge padding, MATLAB
  ``resample(x, p, q, b)`` (ported from resample.m: upfirdn + delay removal).
* ``eeglab_epoch``: epoch.m sample selection (MATLAB rounding, last sample
  excluded, events at floor(latency)).
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from eeglab2mne import translate

mne = pytest.importorskip("mne")
from scipy.signal import upfirdn  # noqa: E402


def _firws_kaiser(m, fc_nyq, beta=5.0):
    """firws(m, fc, windows('kaiser', m + 1, beta)) from the firfilt plugin."""
    n = m + 1
    x = np.arange(n) / (n - 1)
    w = np.i0(beta * np.sqrt(1 - (2 * x - 1) ** 2)) / np.i0(beta)
    f = fc_nyq / 2
    k = np.arange(-m / 2, m / 2 + 1)
    b = np.where(k == 0, 2 * np.pi * f, np.sin(2 * np.pi * f * k) / np.where(k == 0, 1, k)) * w
    return b / b.sum()


def _matlab_resample(x, p, q, h):
    """MATLAB resample(x, p, q, h) for a column vector (resample.m)."""
    lx = len(x)
    lhalf = (len(h) - 1) / 2
    nz = int(math.floor(q - (lhalf % q)))
    h = np.concatenate([np.zeros(nz), h])
    lhalf += nz
    delay = int(math.floor(math.ceil(lhalf) / q))
    ly = int(math.ceil(lx * p / q))
    nz1 = 0
    while math.ceil(((lx - 1) * p + len(h) + nz1) / q) - delay < ly:
        nz1 += 1
    h = np.concatenate([h, np.zeros(nz1)])
    y = upfirdn(h, x, p, q)
    return y[delay:delay + ly]


def eeglab_resample(data, fs, new_fs, fc=0.9, df=0.2):
    r = Fraction(new_fs / fs).limit_denominator(10**6)
    p, q = r.numerator, r.denominator
    nyq = 1 / max(p, q)
    fc, df = fc * nyq, df * nyq
    devdb = -20 * np.log10(0.002)
    m = math.ceil((1 + (devdb - 8) / (2.285 * 2 * np.pi * (df / 2))) / 2) * 2  # pop_firwsord
    b = p * _firws_kaiser(m, fc)
    npad = math.ceil((m / 2) / q) * q
    out = []
    for x in data:
        xp = np.concatenate([np.full(npad, x[0]), x, np.full(npad, x[-1])])
        y = _matlab_resample(xp, p, q, b)
        unpad = int(npad * p / q)
        out.append(y[unpad:len(y) - unpad])
    return np.array(out)


def _mround(x):
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def eeglab_epoch(data, fs, latencies, lims):
    """epoch.m with timeunit 'points': latencies are 1-based EEGLAB latencies."""
    r0, r1 = _mround(lims[0] * fs), _mround(lims[1] * fs - 1)
    out = []
    for lat in latencies:
        pos0 = math.floor(lat)
        a, z = pos0 + r0, pos0 + r1
        if a >= 1 and z <= data.shape[1]:
            out.append(data[:, a - 1:z])
    return np.array(out)


def _raw(fs=500.0, seconds=30, onsets=None):
    rng = np.random.default_rng(7)
    data = rng.normal(0, 10e-6, (3, int(fs * seconds)))
    data += 20e-6 * np.sin(2 * np.pi * 7 * np.arange(data.shape[1]) / fs)
    raw = mne.io.RawArray(data, mne.create_info(3, fs, "eeg"), verbose=False)
    if onsets is not None:
        raw.set_annotations(mne.Annotations(onsets, 0, ["stim"] * len(onsets)))
    return raw


def _exec(src, raw):
    t = translate(src)
    ns = {"raw": raw}
    for step in t.steps:
        exec("\n".join(sorted(t.ctx.imports)), ns)
        exec("\n".join(step.code), ns)
    return t, ns


@pytest.mark.parametrize("new_fs", [250, 128, 1000])
def test_resample_matches_eeglab(new_fs):
    raw = _raw()
    expected = eeglab_resample(raw.get_data() * 1e6, 500.0, new_fs)
    t, ns = _exec(f"EEG = pop_resample(EEG, {new_fs});", raw)
    got = ns["raw"].get_data() * 1e6
    assert t.steps[0].status.value == "exact"
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-9 * np.abs(expected).max()


def test_epoch_matches_eeglab_including_fractional_latencies():
    # Odd sample onsets at 500 Hz land between samples after resampling to 250 Hz.
    onsets_samples = np.arange(1001, 13000, 997)
    raw = _raw(onsets=onsets_samples / 500.0)
    src = ("EEG = pop_resample(EEG, 250);\n"
           "EEG = pop_epoch(EEG, {'stim'}, [-0.2 0.8]);")
    t, ns = _exec(src, raw)
    epochs = ns["epochs"]
    assert t.steps[1].status.value == "exact"

    resampled = eeglab_resample(_raw().get_data() * 1e6, 500.0, 250)
    latencies = onsets_samples * 250 / 500 + 1  # pop_resample: (lat - 1) * p/q + 1
    assert np.any(latencies % 1)  # really fractional
    expected = eeglab_epoch(resampled, 250.0, latencies, (-0.2, 0.8))
    got = epochs.get_data() * 1e6
    assert got.shape == expected.shape  # same count and length (250 samples, not 251)
    assert np.abs(got - expected).max() < 1e-9 * np.abs(expected).max()


def test_epoch_length_matches_eeglab_tutorial_dataset():
    # EEGLAB's own eeglab_data_epochs_ica.set: [-1 2] s at 128 Hz -> 384 samples.
    raw = _raw(fs=128.0, seconds=20, onsets=[5.0, 10.0])
    _, ns = _exec("EEG = pop_epoch(EEG, {'stim'}, [-1 2]);", raw)
    assert ns["epochs"].get_data().shape[-1] == 384
    assert ns["epochs"].times[0] == -1.0


def test_epochs_spanning_a_boundary_are_dropped():
    raw = _raw(fs=100.0, seconds=20, onsets=[3.0, 8.0])
    raw.annotations.append(8.3, 0, "boundary")
    _, ns = _exec("EEG = pop_epoch(EEG, {'stim'}, [-0.5 1]);", raw)
    assert len(ns["epochs"]) == 1
