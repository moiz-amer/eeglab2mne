# eeglab2mne
[![tests](https://github.com/moiz-amer/eeglab2mne/actions/workflows/tests.yml/badge.svg)](https://github.com/moiz-amer/eeglab2mne/actions/workflows/tests.yml)

**Translate EEGLAB pipelines into MNE-Python, and know exactly which steps you can trust.**

Many EEG labs are moving from MATLAB/EEGLAB to Python/MNE, but years of
preprocessing live in EEGLAB scripts and in the processing history stored
inside `.set` files. Rewriting them by hand is slow, and the dangerous part is
not the syntax — it's the silent differences: a baseline given in
milliseconds instead of seconds, 1-based component numbers, an amplitude
threshold that turns into a peak-to-peak one, or ICA component numbers that
stop meaning anything once ICA is refit.

`eeglab2mne` reads an EEGLAB script (or the `EEG.history` inside a `.set`
file), writes an equivalent MNE-Python script, and produces a **migration
report** grading every step:

| | Status | Meaning |
|---|---|---|
| ✅ | **exact** | Same operation; results match up to floating-point error |
| ⚠️ | **approximate** | Same intent, different implementation — verify numerically |
| ❌ | **manual** | No safe automatic translation; a `TODO` is left in the code |
| ➖ | **skipped** | EEGLAB bookkeeping, GUI or plotting |

Nothing is silently dropped: anything the tool can't translate becomes a
`TODO` with the original MATLAB line and a note on what to use instead.

## Quick start

```bash
pip install "eeglab2mne[run] @ git+https://github.com/moiz-amer/eeglab2mne"
# [run] adds MNE + mne-icalabel to execute the output (needs Python 3.10–3.14)

# translate a script
eeglab2mne preprocess.m -o preprocess_mne.py --report migration_report.md

# or recover the pipeline from a processed dataset, reusing its ICA
eeglab2mne sub01_clean.set --ica reuse -o sub01_mne.py
```

Use `--strict` in CI to fail when any step still needs manual translation.

From Python:

```python
from eeglab2mne import translate

t = translate(open("preprocess.m").read())
print(t.counts)          # Counter({exact: 11, approximate: 4, manual: 2, skipped: 4})
open("pipeline.py", "w").write(t.to_python())
```

## Example

Input (`examples/erp_preprocessing.m`, excerpt):

```matlab
EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);
EEG = pop_reref(EEG, []);
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);
EEG = pop_iclabel(EEG, 'default');
EEG = pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);
EEG = pop_subcomp(EEG, [], 0);
EEG = pop_epoch(EEG, {'S  1' 'S  2'}, [-0.2 0.8]);
EEG = pop_rmbase(EEG, [-200 0]);
EEG = pop_eegthresh(EEG, 1, 1:30, -100, 100, -0.2, 0.796, 0, 1);
```

Output (excerpt):

```python
_df = min(max(1 * 0.25, 2), min(1, _fs / 2 - 40))   # EEGLAB's transition-band rule
_order = int(np.ceil(3.3 / (_df / _fs) / 2) * 2)
raw.filter(l_freq=1, h_freq=40, l_trans_bandwidth=_df, h_trans_bandwidth=_df,
           filter_length=_order + 1, pad='edge', ...)   # identical to EEGLAB's filter
raw.set_eeg_reference('average', projection=False)
ica = mne.preprocessing.ICA(n_components=None, method='infomax', fit_params=dict(extended=True), ...)
ica.fit(raw)
ic_proba = iclabel_label_components(raw, ica, inplace=True)  # (n_ics, 7)
...
epochs.apply_baseline((-0.2, 0.0))          # EEGLAB ms -> MNE s
data = epochs.get_data(picks=epochs.ch_names[0:30], tmin=-0.2, tmax=0.796) * 1e6  # V -> µV
bad = np.any((data < -100) | (data > 100), axis=(1, 2))   # absolute, not peak-to-peak
```

See the full [generated script](examples/erp_preprocessing_mne.py) and
[migration report](examples/erp_preprocessing_report.md).

## Validated on EEGLAB's own data

The test suite downloads EEGLAB's tutorial dataset (`eeglab_data_epochs_ica.set`,
processed and ICA-decomposed in EEGLAB itself), translates its stored history,
runs the generated MNE code, and compares the result with EEGLAB's math:

| Check | Result |
|---|---|
| ICA activations, MNE vs EEGLAB (all 32 components) | correlation = 1.0000000000 |
| Data after removing components (`pop_subcomp`) | max relative difference 3 × 10⁻¹⁵ (floating-point rounding) |

So when you migrate already-processed data with `--ica reuse`, MNE reproduces
EEGLAB's ICA cleaning exactly.

### Filtering (`pop_eegfiltnew`)

A real-EEGLAB comparison caught a silent mismatch in v0.2: EEGLAB designs a
1–30 Hz bandpass with a ~1 Hz transition band at *both* edges, while MNE's
defaults use 7.5 Hz at the 30 Hz edge (correlation 0.998, up to 12.8 µV off).
The translation now reproduces EEGLAB's own design rules — transition band,
filter order, Hamming window, cutoffs and edge padding — and is checked against
a line-by-line port of EEGLAB's firfilt plugin:

| Check | Result |
|---|---|
| Filter coefficients, MNE vs EEGLAB `firws` (1–30 Hz, 128 Hz) | identical |
| Filtered data: bandpass, highpass, lowpass, band-stop, fixed order, `boundary` events | max relative difference < 10⁻¹² |
| **Same script in real EEGLAB (MATLAB) vs translated MNE code**: load → 1–30 Hz filter → average reference → epoch → baseline, 80 epochs | correlation 1.000000, max difference < 0.001 µV (before the fix: 0.998 and 12.8 µV) |

### Resampling and epoching (`pop_resample`, `pop_epoch`)

`pop_resample` uses its own anti-aliasing filter (Kaiser window, designed by
firfilt) with edge padding; MNE's default resampler is FFT-based, and MNE's
polyphase option derives its ratio from sample counts (which breaks EEGLAB's
filter whenever the length doesn't divide evenly). The generated code therefore
includes a small `eeglab_resample()` helper that resamples exactly like EEGLAB. `pop_epoch`
rounds its limits to whole samples, drops the final sample and places events at
`floor(latency)`; MNE's defaults keep one extra sample and round latencies.
Both translations now reproduce EEGLAB, checked against NumPy ports of
`pop_resample.m` (with MATLAB's `resample`) and `epoch.m`:

| Check | Result |
|---|---|
| Resampling 500 Hz → 250, 128 and 1000 Hz, and recording lengths that don't divide evenly (e.g. 30504 samples, 128 → 100 Hz) | max relative difference < 10⁻⁹ |
| Epochs after resampling, with event latencies between samples | identical samples (relative difference < 10⁻⁹) |
| Epoch length, `[-1 2]` s at 128 Hz | 384 samples, as in EEGLAB's tutorial dataset (MNE's default: 385) |

## Loops, conditions and variables

Real preprocessing scripts loop over subjects and build file names from
variables. These translate into ordinary Python:

```matlab
datapath = '/data/study';
subjects = {'sub01', 'sub02'};
for s = 1:length(subjects)
    EEG = pop_loadset('filename', [subjects{s} '.set'], 'filepath', datapath);
    if strcmp(subjects{s}, 'sub02')
        EEG = pop_resample(EEG, 250);
    end
    EEG = pop_saveset(EEG, 'filename', [subjects{s} '_clean.set'], 'filepath', datapath);
end
```

```python
datapath = '/data/study'
subjects = ['sub01', 'sub02']
for s in range(1, len(subjects) + 1):
    raw = mne.io.read_raw_eeglab(os.path.join(datapath, subjects[s - 1] + '.set'), preload=True)
    if (subjects[s - 1] == 'sub02'):
        ...  # EEGLAB's resampling
    raw.save(os.path.join(datapath, os.path.splitext(subjects[s - 1] + '_clean.set')[0] + '-raw.fif'), overwrite=True)
```

Supported: `for`, `while`, `if`/`elseif`/`else`, `switch`/`case`, `try`/`catch`,
simple assignments, cell arrays, `{}`/`()` indexing (converted to 0-based),
string concatenation, `fullfile`, `sprintf`, `num2str`, `strcmp`, `length` and
`EEG.srate`/`EEG.nbchan`/`EEG.pnts`. A variable is only used if the script
defines it; anything else becomes `TODO(...)`, which stops the generated script
with a clear message instead of guessing. See the full
[loop example](examples/subject_loop.m) → [generated Python](examples/subject_loop_mne.py).

## Pitfalls it handles

- **Units.** EEGLAB baselines are in ms and amplitudes in µV; MNE uses s and V.
- **Indexing.** Channel, trial and component numbers are converted from 1-based to 0-based.
- **Amplitude rejection.** `pop_eegthresh` is an *absolute* threshold. MNE's `reject=dict(eeg=...)` is *peak-to-peak* and rejects different epochs, so the tool writes an explicit absolute check instead.
- **ICA refits.** A fresh ICA in Python gives different components in a different order. The tool warns, and turns any hard-coded `pop_subcomp` numbers after a refit into a `TODO` rather than removing the wrong components.
- **Rejected components.** `mne.preprocessing.read_ica_eeglab` loads EEGLAB's decomposition but not which components were flagged. In `--ica reuse` mode the generated code also recovers `EEG.reject.gcompreject`.
- **Reference channels.** `pop_reref` removes the new reference channels unless `'keepref','on'`; MNE keeps them. The translation drops them to match.
- **Filters.** `pop_eegfiltnew` and MNE's `filter()` pick different transition bands by default, which silently changes your data. The translation computes EEGLAB's transition band and filter order and pads data edges the way EEGLAB does, so the filter is identical. `revfilt` becomes a band-stop, `filtorder` is honoured, and filtering restarts at `boundary` events.
- **Epoching.** `pop_epoch` doesn't baseline-correct, so `baseline=None` (MNE's default would). Epochs have EEGLAB's exact samples and length, and epochs spanning a `boundary` are dropped as EEGLAB does.
- **Resampling.** EEGLAB's anti-aliasing filter is reproduced instead of MNE's FFT default.
- **Translating a `.set` file.** History paths point at the original author's machine, so loads of that dataset are redirected to your file; epoched datasets are read as `mne.Epochs`; steps made moot by a later reload of the same file are marked superseded; and an ICA computed *before* the recorded history is still loaded.

## Supported EEGLAB functions

| Area | Functions |
|---|---|
| I/O | `pop_loadset`, `pop_saveset`, `pop_biosig`, `pop_fileio`, `pop_loadbv`, and other importers |
| Channels | `pop_select`, `pop_chanedit` (`'lookup'`, `'load'`, `'settype'`, `'changefield'` for type/labels), `pop_interp`, `pop_reref` |
| Filtering | `pop_eegfiltnew`, `pop_eegfilt` (legacy), `pop_resample`, `pop_cleanline` |
| Epochs | `pop_epoch`, `pop_rmbase`, `pop_eegthresh`, `pop_rejepoch` |
| ICA | `pop_runica` (runica/binica/picard/fastica), `pop_iclabel`, `pop_icflag`, `pop_subcomp`, `pop_selectcomps` |
| Plotting (skipped) | `pop_topoplot`, `pop_eegplot`, `pop_spectopo`, `pop_erpimage`, `pop_timtopo`, `pop_prop`, and similar |
| Flagged for manual work | `pop_clean_rawdata` / `clean_artifacts` (with a parameter-by-parameter mapping to pyprep, ASR and EEGPrep) |

Unknown functions are never dropped; they become `TODO`s.

## How this relates to other tools

- **[MNE-Python](https://mne.tools)** reads `.set` files and has a [migration table](https://mne.tools/stable/help/migrating.html). `eeglab2mne` automates that table for your actual scripts and records what changed.
- **[eeglabio](https://github.com/jackz314/eeglabio)** handles file I/O between the two formats. It doesn't touch processing steps.
- **[mne-icalabel](https://github.com/mne-tools/mne-icalabel)** is a Python port of ICLabel, used by the generated code.
- **EEGPrep** ([arXiv, 2026](https://arxiv.org/abs/2607.16647)) is a numerically validated Python re-implementation of EEGLAB's default pipeline. It's the best target when you need EEGLAB-identical numbers for `clean_rawdata`. `eeglab2mne` is complementary: it converts *your* script, step by step, and audits the result.

## Development

```bash
git clone https://github.com/YOUR-USERNAME/eeglab2mne && cd eeglab2mne
pip install -e ".[test]"
pytest
```

The end-to-end tests build synthetic EEGLAB `.set` files, translate pipelines,
**execute the generated Python**, and check the results (event names, sampling
rate, baseline, ICLabel output shape, carried-over rejected components).

## Roadmap

- [x] Numerical validation of ICA reuse against EEGLAB's tutorial dataset
- [x] Exact `pop_eegfiltnew` filtering, validated against EEGLAB
- [x] Exact `pop_resample` and `pop_epoch`
- [x] Loops, conditions and variables
- [ ] Exact CleanLine and spherical-spline interpolation (`pop_interp`)
- [ ] Translate `clean_rawdata` directly to pyprep + ASR with matched parameters
- [ ] ERPLAB functions (`pop_binlister`, `pop_averager`)
- [ ] `STUDY` scripts → MNE-BIDS / group loops
- [ ] Web demo: paste a script, get code + report

## License

MIT
