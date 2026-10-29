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
pip install "eeglab2mne[run]"     # [run] adds MNE + mne-icalabel to execute the output

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
print(t.counts)          # Counter({exact: 8, approximate: 7, manual: 2, skipped: 4})
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
raw.filter(l_freq=1, h_freq=40, method='fir', fir_design='firwin', fir_window='hamming', phase='zero')
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

## Pitfalls it handles

- **Units.** EEGLAB baselines are in ms and amplitudes in µV; MNE uses s and V.
- **Indexing.** Channel, trial and component numbers are converted from 1-based to 0-based.
- **Amplitude rejection.** `pop_eegthresh` is an *absolute* threshold. MNE's `reject=dict(eeg=...)` is *peak-to-peak* and rejects different epochs, so the tool writes an explicit absolute check instead.
- **ICA refits.** A fresh ICA in Python gives different components in a different order. The tool warns, and turns any hard-coded `pop_subcomp` numbers after a refit into a `TODO` rather than removing the wrong components.
- **Rejected components.** `mne.preprocessing.read_ica_eeglab` loads EEGLAB's decomposition but not which components were flagged. In `--ica reuse` mode the generated code also recovers `EEG.reject.gcompreject`.
- **Reference channels.** `pop_reref` removes the new reference channels unless `'keepref','on'`; MNE keeps them. The translation drops them to match.
- **Filters.** `revfilt` becomes a band-stop; an explicit `filtorder` becomes `filter_length = order + 1`.
- **Epoching.** `pop_epoch` doesn't baseline-correct, so `baseline=None` (MNE's default would).

## Supported EEGLAB functions

| Area | Functions |
|---|---|
| I/O | `pop_loadset`, `pop_saveset`, `pop_biosig`, `pop_fileio`, `pop_loadbv`, and other importers |
| Channels | `pop_select`, `pop_chanedit` (`'lookup'`), `pop_interp`, `pop_reref` |
| Filtering | `pop_eegfiltnew`, `pop_eegfilt` (legacy), `pop_resample`, `pop_cleanline` |
| Epochs | `pop_epoch`, `pop_rmbase`, `pop_eegthresh`, `pop_rejepoch` |
| ICA | `pop_runica` (runica/binica/picard/fastica), `pop_iclabel`, `pop_icflag`, `pop_subcomp`, `pop_selectcomps` |
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

- [ ] Numerical validation: run each translated step against EEGLAB output on public datasets and publish per-step error (the "approximate" grades should become measured numbers)
- [ ] Translate `clean_rawdata` directly to pyprep + ASR with matched parameters
- [ ] ERPLAB functions (`pop_binlister`, `pop_averager`)
- [ ] `STUDY` scripts → MNE-BIDS / group loops
- [ ] Web demo: paste a script, get code + report

## License

MIT
