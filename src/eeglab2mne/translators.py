"""Translate parsed EEGLAB calls into MNE-Python code.

Every translator returns a :class:`Step` that records not just the code but
*how faithful* the translation is. That audit trail is the point of this
tool: a migrated pipeline is only trustworthy if you know which steps are
numerically identical, which are close, and which need a human.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .parser import Call, Expr


class Status(str, Enum):
    EXACT = "exact"  # same operation; numerically equivalent up to float error
    APPROX = "approximate"  # same intent, implementation differs; verify
    MANUAL = "manual"  # no safe automatic translation; TODO left in code
    SKIPPED = "skipped"  # EEGLAB bookkeeping/GUI/plotting; nothing to do


@dataclass
class Step:
    call: Call | None
    source: str
    line: int
    status: Status
    code: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def func(self) -> str:
        return self.call.func if self.call else "(unparsed)"


@dataclass
class Context:
    """Mutable state carried across steps while translating a script."""

    obj: str = "raw"  # current data variable: "raw" or "epochs"
    ica_mode: str = "refit"  # "refit" (fit new ICA) or "reuse" (load EEGLAB's)
    has_ica: bool = False
    ica_refit: bool = False  # True once an ICA was *refit* in Python
    imports: set[str] = field(default_factory=lambda: {"import mne"})
    warnings: list[str] = field(default_factory=list)
    input_path: str | None = None
    ica_source: str | None = None  # .set file holding EEGLAB's ICA (reuse mode)
    # Facts read from the .set file being translated (when translating a .set):
    # {"path": str, "epoched": bool, "has_ica": bool}
    set_info: dict | None = None


class NeedsManual(Exception):
    """Raised inside a translator when an argument can't be converted."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def lit(v: Any) -> str:
    """Render a Python literal for generated code."""
    if isinstance(v, Expr):
        raise NeedsManual(f"MATLAB expression `{v.text}` must be resolved by hand")
    if isinstance(v, float):
        if math.isnan(v):
            return "np.nan"
        if math.isinf(v):
            return "np.inf" if v > 0 else "-np.inf"
    if isinstance(v, list):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    return repr(v)


def is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, list) and len(v) == 0)


def as_list(v: Any) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def ch_names_expr(ctx: Context, chans: Any) -> str:
    """Channels as names (strings) or 1-based indices -> Python expression."""
    chans = as_list(chans)
    if not chans:
        raise NeedsManual("empty channel list")
    if all(isinstance(c, str) for c in chans):
        return lit(chans)
    if all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in chans):
        idx = [int(c) - 1 for c in chans]  # EEGLAB is 1-based
        if idx == list(range(idx[0], idx[0] + len(idx))):
            return f"{ctx.obj}.ch_names[{idx[0]}:{idx[-1] + 1}]"
        return f"[{ctx.obj}.ch_names[i] for i in {idx}]"
    raise NeedsManual(f"cannot interpret channel list {chans!r}")


def pos(call: Call, i: int, key: str | None = None, default: Any = None) -> Any:
    """Get an argument by keyword, falling back to positional index."""
    if key and call.has(key):
        return call.get(key)
    return call.positional[i] if i < len(call.positional) else default


def truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.lower() in ("on", "yes", "true", "1")
    return bool(v) and not is_empty(v)


# --------------------------------------------------------------------------
# translators
# --------------------------------------------------------------------------

Translator = Callable[[Call, Context], Step]
REGISTRY: dict[str, Translator] = {}


def translates(*names: str):
    def deco(fn: Translator) -> Translator:
        for n in names:
            REGISTRY[n.lower()] = fn
        return fn

    return deco


def _step(call: Call, status: Status, code=None, notes=None) -> Step:
    return Step(call, call.source, call.line, status, list(code or []), list(notes or []))


# ---- bookkeeping / GUI / plotting ----------------------------------------

_BOOKKEEPING = (
    "eeglab", "eeg_checkset", "eeg_store", "pop_newset", "eeg_retrieve",
    "pop_delset", "eegh", "pop_saveh", "clear", "close", "clc", "addpath",
    "rmpath", "redraw", "disp", "fprintf", "pause", "drawnow", "figure",
)


@translates(*_BOOKKEEPING)
def _bookkeeping(call: Call, ctx: Context) -> Step:
    return _step(call, Status.SKIPPED,
                 notes=["EEGLAB session/GUI bookkeeping; MNE needs no equivalent."])


@translates("pop_eegplot")
def _eegplot(call: Call, ctx: Context) -> Step:
    return _step(call, Status.SKIPPED, [f"# {ctx.obj}.plot()  # interactive browser (optional)"],
                 ["Plotting only. Marking bad segments in the MNE browser creates BAD_ annotations."])


_PLOTTING = (
    "pop_topoplot", "topoplot", "pop_plottopo", "plottopo", "pop_erpimage", "erpimage",
    "pop_timtopo", "timtopo", "pop_envtopo", "envtopo", "pop_prop", "pop_prop_extended",
    "pop_headplot", "headplot", "pop_comperp", "pop_plotdata", "pop_eegplot_w",
)


@translates(*_PLOTTING)
def _plotting(call: Call, ctx: Context) -> Step:
    return _step(call, Status.SKIPPED,
                 notes=["Plotting only; does not change the data. MNE equivalents: "
                        "evoked.plot_topomap(), epochs.plot_image(), ica.plot_properties()."])


@translates("pop_spectopo", "spectopo")
def _spectopo(call: Call, ctx: Context) -> Step:
    return _step(call, Status.SKIPPED, [f"# {ctx.obj}.compute_psd().plot()  # optional"],
                 ["Plotting only. EEGLAB spectopo uses Welch; MNE compute_psd defaults to Welch for Raw, multitaper for Epochs."])


# ---- I/O ------------------------------------------------------------------


@translates("pop_loadset")
def _loadset(call: Call, ctx: Context) -> Step:
    fname = call.get("filename", call.positional[0] if call.positional else None)
    fpath = call.get("filepath", call.positional[1] if len(call.positional) > 1 else "")
    if fname is None:
        raise NeedsManual("no filename given")
    notes: list[str] = []
    info = ctx.set_info
    epoched = None
    if isinstance(fname, Expr) or isinstance(fpath, Expr):
        fname_code = "INPUT_SET  # TODO: set path (was a MATLAB variable)"
    elif info and str(fname).replace("\\", "/").split("/")[-1] == Path(info["path"]).name:
        # The history records paths on the machine that made it; point at the real file.
        fname_code = lit(info["path"])
        ctx.input_path = info["path"]
        epoched = info["epoched"]
        if fpath:
            notes.append(f"Original path `{fpath}` is from the machine that created the file; "
                         "replaced with the file you translated.")
    else:
        full = (fpath.rstrip("/\\") + "/" + fname) if fpath else fname
        fname_code = lit(full)
        ctx.input_path = full

    if epoched:
        ctx.obj = "epochs"
        code = [f"epochs = mne.read_epochs_eeglab({fname_code})"]
        notes.append("Dataset is epoched (EEG.trials > 1), so it is read as mne.Epochs.")
    else:
        ctx.obj = "raw"
        code = [f"raw = mne.io.read_raw_eeglab({fname_code}, preload=True)"]
        if epoched is None:
            notes.append("If this .set is already epoched, use mne.read_epochs_eeglab() instead "
                         "and rename `raw` to `epochs`.")
    notes.append("Channel types: EEGLAB marks everything as EEG; set EOG/ECG with set_channel_types().")
    return _step(call, Status.EXACT, code, notes)


@translates("pop_biosig", "pop_fileio", "pop_loadbv", "pop_readegi", "pop_loadcnt", "pop_loadeep")
def _import(call: Call, ctx: Context) -> Step:
    fname = call.positional[0] if call.positional else call.get("filename")
    if fname is None or isinstance(fname, Expr):
        code = ["raw = mne.io.read_raw(INPUT_FILE, preload=True)  # TODO: set path"]
    else:
        code = [f"raw = mne.io.read_raw({lit(fname)}, preload=True)"]
    if call.func.lower() == "pop_loadbv" and len(call.positional) >= 2:
        p, f = call.positional[0], call.positional[1]
        if isinstance(p, str) and isinstance(f, str):
            code = [f"raw = mne.io.read_raw_brainvision({lit(p.rstrip('/') + '/' + f)}, preload=True)"]
    ctx.obj = "raw"
    return _step(call, Status.APPROX, code,
                 ["Readers differ in how they map event codes, channel types and units; "
                  "compare raw.annotations with EEG.event before trusting downstream epoching."])


@translates("pop_saveset")
def _saveset(call: Call, ctx: Context) -> Step:
    fname = call.get("filename", "output.set")
    stem = fname.rsplit(".", 1)[0] if isinstance(fname, str) else "output"
    suffix = "epo" if ctx.obj == "epochs" else "raw"
    return _step(call, Status.EXACT,
                 [f"{ctx.obj}.save({stem + '-' + suffix + '.fif'!r}, overwrite=True)",
                  f"# Or keep EEGLAB format: mne.export.export_raw/export_epochs(..., fmt='eeglab')"],
                 ["Saved as .fif (MNE native). Exporting back to .set needs the eeglabio package."])


# ---- channels -------------------------------------------------------------


_MNE_TYPES = {"eeg": "eeg", "eog": "eog", "heog": "eog", "veog": "eog", "ecg": "ecg",
              "ekg": "ecg", "emg": "emg", "misc": "misc", "ref": "misc", "stim": "stim"}
_RANK = {Status.SKIPPED: 0, Status.EXACT: 1, Status.APPROX: 2, Status.MANUAL: 3}


def _ch_index_expr(ctx: Context, spec: Any) -> str:
    """EEGLAB channel spec (1-based number, list, or '1:32' string) -> names expr."""
    from .parser import parse_value

    if isinstance(spec, str):
        spec = parse_value(spec)
    return ch_names_expr(ctx, spec)


@translates("pop_chanedit")
def _chanedit(call: Call, ctx: Context) -> Step:
    o = ctx.obj
    code: list[str] = []
    notes: list[str] = []
    status = Status.SKIPPED
    for key, val in call.pairs:
        st = Status.EXACT
        if key == "lookup":
            st = Status.APPROX
            code += ["# MNE 1.13 renamed 'standard_1005' to 'colin27_1005'; pick whichever exists",
                     "_template = next(m for m in ('colin27_1005', 'standard_1005')",
                     "                 if m in mne.channels.get_builtin_montages())",
                     f"{o}.set_montage(_template, on_missing='warn')"]
            notes.append("'lookup' matches labels against a template file (here "
                         f"`{str(val).split('/')[-1]}`). MNE's built-in 10-05 template is close but "
                         "not identical, so coordinates and topoplots can differ slightly.")
        elif key == "load":
            path = val[0] if isinstance(val, list) and val else val
            if not isinstance(path, str):
                raise NeedsManual("pop_chanedit 'load' with a non-literal file")
            st = Status.APPROX
            code.append(f"{o}.set_montage(mne.channels.read_custom_montage({path!r}), on_missing='warn')")
            notes.append(f"Loads electrode positions from `{path.split('/')[-1]}`; update the path "
                         "if the file lives elsewhere on your machine.")
        elif key == "settype":
            if not (isinstance(val, list) and len(val) == 2):
                raise NeedsManual("unexpected 'settype' value")
            chans, typ = val
            mne_type = _MNE_TYPES.get(str(typ).lower())
            if mne_type is None:
                raise NeedsManual(f"channel type '{typ}'")
            code.append(f"{o}.set_channel_types({{ch: {mne_type!r} for ch in {_ch_index_expr(ctx, chans)}}})")
        elif key == "changefield":
            if not (isinstance(val, list) and len(val) == 3):
                raise NeedsManual("unexpected 'changefield' value")
            idx, fieldname, new = val
            if not isinstance(idx, int):
                raise NeedsManual("'changefield' channel index is not a number")
            ch = f"{o}.ch_names[{idx - 1}]"
            if str(fieldname).lower() == "type":
                mne_type = _MNE_TYPES.get(str(new).lower())
                if mne_type is None:
                    raise NeedsManual(f"channel type '{new}'")
                code.append(f"{o}.set_channel_types({{{ch}: {mne_type!r}}})  # EEGLAB channel {idx}")
            elif str(fieldname).lower() == "labels":
                code.append(f"{o}.rename_channels({{{ch}: {new!r}}})  # EEGLAB channel {idx}")
            else:
                raise NeedsManual(f"'changefield' of '{fieldname}'")
        elif key in ("plotrad", "shrink", "nosedir"):
            st = Status.SKIPPED
            notes.append(f"'{key}' only changes how scalp maps are drawn.")
        elif key == "eval" and "writelocs" in str(val):
            st = Status.SKIPPED
            notes.append("'eval' here only exports channel locations to a file.")
        else:
            raise NeedsManual(f"pop_chanedit option '{key}'")
        if _RANK[st] > _RANK[status]:
            status = st
    if not call.pairs:
        raise NeedsManual("pop_chanedit with no options (interactive GUI)")
    return _step(call, status, code, notes)


@translates("pop_select")
def _select(call: Call, ctx: Context) -> Step:
    code: list[str] = []
    notes: list[str] = []
    status = Status.EXACT
    o = ctx.obj
    for key, val in call.kwargs.items():
        if key == "channel":
            code.append(f"{o}.pick({ch_names_expr(ctx, val)})")
        elif key in ("nochannel", "rmchannel"):
            code.append(f"{o}.drop_channels({ch_names_expr(ctx, val)})")
        elif key == "time":
            if isinstance(val, list) and len(val) == 2 and not isinstance(val[0], list):
                code.append(f"{o}.crop(tmin={lit(val[0])}, tmax={lit(val[1])})")
            else:
                raise NeedsManual("pop_select 'time' with several ranges")
        elif key == "notime":
            rows = val if (isinstance(val, list) and val and isinstance(val[0], list)) else [val]
            code.append("# Remove time ranges by marking them BAD; most MNE functions skip BAD_ spans.")
            for r in rows:
                if not (isinstance(r, list) and len(r) == 2):
                    raise NeedsManual("unexpected 'notime' shape")
                code.append(f"{o}.annotations.append(onset={lit(r[0])}, duration={lit(r[1])} - {lit(r[0])}, "
                            f"description='BAD_eeglab_notime')")
            status = Status.APPROX
            notes.append("EEGLAB cuts the data and inserts 'boundary' events; MNE keeps the samples "
                         "but annotates them BAD. Filtering/epoching skip them, but array shapes differ.")
        elif key == "trial":
            idx = [int(i) - 1 for i in as_list(val)]
            code.append(f"{o} = {o}[{idx}]  # EEGLAB trials {as_list(val)} (1-based)")
        elif key == "notrial":
            idx = [int(i) - 1 for i in as_list(val)]
            code.append(f"{o}.drop({idx}, reason='eeglab_notrial')  # EEGLAB trials {as_list(val)} (1-based)")
        else:
            raise NeedsManual(f"pop_select option '{key}'")
    if not code:
        raise NeedsManual("pop_select with no recognised options")
    return _step(call, status, code, notes)


@translates("pop_interp")
def _interp(call: Call, ctx: Context) -> Step:
    bad = call.positional[0] if call.positional else None
    method = call.positional[1] if len(call.positional) > 1 else "spherical"
    if isinstance(bad, Expr):
        return _step(call, Status.MANUAL,
                     [f"# TODO: {call.source}",
                      "# EEGLAB is restoring channels that were removed earlier (from a saved chanlocs).",
                      "# In MNE, keep the channels instead of dropping them: mark them in",
                      f"# {ctx.obj}.info['bads'] where you removed them, then call:",
                      f"# {ctx.obj}.interpolate_bads(reset_bads=True)"],
                     ["Common idiom after clean_rawdata. MNE's cleaner workflow is to mark channels bad "
                      "rather than delete them, so interpolation needs no stored chanlocs."])
    if method != "spherical":
        raise NeedsManual(f"interpolation method '{method}'")
    return _step(call, Status.APPROX,
                 [f"{ctx.obj}.info['bads'] = {ch_names_expr(ctx, bad)}",
                  f"{ctx.obj}.interpolate_bads(reset_bads=True)"],
                 ["Both use spherical splines, but default Legendre order and regularisation differ; "
                  "interpolated values will be close, not identical. Requires a montage."])


# ---- filtering / resampling / referencing ---------------------------------


@translates("pop_eegfiltnew", "pop_eegfilt")
def _filter(call: Call, ctx: Context) -> Step:
    legacy = call.func.lower() == "pop_eegfilt"
    lo = pos(call, 0, "locutoff")
    hi = pos(call, 1, "hicutoff")
    order = pos(call, 2, "filtorder")
    revfilt = truthy(pos(call, 3, "revfilt", 0))
    lo = None if is_empty(lo) or lo == 0 else lo
    hi = None if is_empty(hi) or hi == 0 else hi
    if lo is None and hi is None:
        raise NeedsManual("no cutoff frequencies found")
    for v in (lo, hi, order):
        if isinstance(v, Expr):
            raise NeedsManual(f"cutoff/order given as expression `{v.text}`")

    if revfilt:
        if lo is not None and hi is not None:
            l_freq, h_freq = hi, lo  # MNE: l_freq > h_freq means band-stop
        elif lo is not None:
            l_freq, h_freq = None, lo  # inverted high-pass = low-pass
        else:
            l_freq, h_freq = hi, None  # inverted low-pass = high-pass
    else:
        l_freq, h_freq = lo, hi

    extra = ""
    if not is_empty(order) and not legacy:
        extra = f", filter_length={int(order) + 1}"  # EEGLAB order = taps - 1
    code = [f"{ctx.obj}.filter(l_freq={lit(l_freq)}, h_freq={lit(h_freq)}, method='fir', "
            f"fir_design='firwin', fir_window='hamming', phase='zero'{extra})"]
    notes = ["Both are zero-phase windowed-sinc FIR (Hamming) defined by pass-band edges. "
             "MNE's 'auto' transition band is min(max(0.25*f, 2 Hz), f); check it matches the "
             "'transition bandwidth' EEGLAB printed in its log, and pass l_trans_bandwidth/"
             "h_trans_bandwidth explicitly if not."]
    if revfilt:
        notes.append("revfilt=1 (notch/inverted) translated as band-stop: MNE treats l_freq > h_freq as stop-band.")
    if legacy:
        notes.append("pop_eegfilt is EEGLAB's *legacy* least-squares filter. MNE has no equivalent; this "
                     "substitutes the modern windowed-sinc design, so expect visible differences.")
    return _step(call, Status.APPROX, code, notes)


@translates("pop_resample")
def _resample(call: Call, ctx: Context) -> Step:
    fs = pos(call, 0, "freq")
    if is_empty(fs):
        raise NeedsManual("no target rate")
    return _step(call, Status.APPROX, [f"{ctx.obj}.resample({lit(fs)})"],
                 ["Different anti-aliasing filters (EEGLAB: firfilt-based or MATLAB resample; "
                  "MNE: FFT by default). Content well below the new Nyquist matches closely."])


@translates("pop_reref")
def _reref(call: Call, ctx: Context) -> Step:
    ref = call.positional[0] if call.positional else []
    notes = []
    if call.has("exclude"):
        notes.append("'exclude' channels: in MNE, set their type to non-EEG (e.g. 'eog') so the "
                     "average excludes them.")
    if is_empty(ref):
        code = [f"{ctx.obj}.set_eeg_reference('average', projection=False)"]
        notes.append("Average reference reduces data rank by one. If ICA follows, set "
                     "n_components to at most (number of channels - 1).")
        return _step(call, Status.EXACT, code, notes)
    names = ch_names_expr(ctx, ref)
    code = [f"ref_chs = {names}", f"{ctx.obj}.set_eeg_reference(ref_channels=ref_chs)"]
    if not truthy(call.get("keepref", "off")):
        code.append(f"{ctx.obj}.drop_channels(ref_chs)  # EEGLAB removes ref channels unless 'keepref','on'")
    return _step(call, Status.EXACT, code, notes)


@translates("pop_cleanline")
def _cleanline(call: Call, ctx: Context) -> Step:
    freqs = call.get("linefreqs", [50, 100])
    if isinstance(freqs, Expr):
        raise NeedsManual("line frequencies given as expression")
    return _step(call, Status.APPROX,
                 [f"{ctx.obj}.notch_filter(freqs={lit(as_list(freqs))}, method='spectrum_fit')"],
                 ["CleanLine fits and subtracts sinusoids with multitapers; MNE's "
                  "method='spectrum_fit' is the closest analogue (same idea, different windowing). "
                  "Consider Zapline (meegkit) as a stronger alternative."])


# ---- artifact rejection (no direct MNE equivalent) ------------------------


_CLEAN_RAWDATA_MAP = {
    "flatlinecriterion": "flat channels (s)  -> mne.preprocessing.annotate_amplitude(flat=...) / pyprep",
    "channelcriterion": "correlation-based bad channels -> pyprep.NoisyChannels.find_bad_by_correlation",
    "linenoisecriterion": "line-noise bad channels -> pyprep.NoisyChannels.find_bad_by_hfnoise",
    "burstcriterion": "Artifact Subspace Reconstruction -> asrpy.ASR(cutoff=...)",
    "windowcriterion": "bad-window removal -> mne annotations from a power threshold",
}


@translates("pop_clean_rawdata", "clean_rawdata", "clean_artifacts")
def _clean_rawdata(call: Call, ctx: Context) -> Step:
    code = [f"# TODO: {call.source}",
            "# clean_rawdata has no MNE equivalent. Options, from closest to simplest:",
            "#   1. EEGPrep (2026): validated Python port of EEGLAB's default pipeline,",
            "#      including clean_rawdata -- best if you need EEGLAB-equivalent numbers.",
            "#   2. pyprep (bad channels) + asrpy (ASR bursts), mapped below.",
            "#   3. mne.preprocessing.annotate_amplitude for simple flat/extreme segments."]
    for key, val in call.kwargs.items():
        if key in _CLEAN_RAWDATA_MAP:
            shown = "off" if (isinstance(val, str) and val.lower() == "off") or val == -1 else val
            code.append(f"#   {key}={shown!r}: {_CLEAN_RAWDATA_MAP[key]}")
    return _step(call, Status.MANUAL, code,
                 ["The default EEGLAB pipeline's advantage over MNE pipelines in a recent benchmark "
                  "was largely attributed to clean_rawdata's bad-channel step, so do not drop it silently."])


@translates("pop_eegthresh")
def _eegthresh(call: Call, ctx: Context) -> Step:
    # pop_eegthresh(EEG, typerej, elec, lowthresh, upthresh, starttime, endtime, superpose, reject)
    p = call.positional
    if len(p) < 4 or p[0] != 1:
        raise NeedsManual("pop_eegthresh on ICA activations, or too few arguments")
    elec, lo, hi = p[1], p[2], p[3]
    t0 = p[4] if len(p) > 4 else None
    t1 = p[5] if len(p) > 5 else None
    picks = ch_names_expr(ctx, elec) if not isinstance(elec, Expr) else "'eeg'"
    tmin = "" if is_empty(t0) else f", tmin={lit(t0)}"
    tmax = "" if is_empty(t1) else f", tmax={lit(t1)}"
    ctx.imports.add("import numpy as np")
    code = [f"data = epochs.get_data(picks={picks}{tmin}{tmax}) * 1e6  # V -> µV, EEGLAB thresholds are µV",
            f"bad = np.any((data < {lit(lo)}) | (data > {lit(hi)}), axis=(1, 2))",
            "epochs.drop(np.flatnonzero(bad), reason='eeglab_threshold')"]
    return _step(call, Status.EXACT, code,
                 ["Absolute-amplitude rejection. Do NOT translate this to MNE's reject=dict(eeg=...): "
                  "that is a peak-to-peak criterion and rejects different epochs.",
                  "Units: MNE stores Volts, EEGLAB µV."])


@translates("pop_rejepoch")
def _rejepoch(call: Call, ctx: Context) -> Step:
    trials = call.positional[0] if call.positional else None
    if isinstance(trials, Expr) or trials is None:
        raise NeedsManual("trial list given as expression (often EEG.reject.rejthresh etc.)")
    vals = as_list(trials)
    if vals and set(vals) <= {0, 1} and len(vals) > 2:
        idx = [i for i, v in enumerate(vals) if v]
        note = "Interpreted as a logical mask."
    else:
        idx = [int(v) - 1 for v in vals]
        note = "Interpreted as 1-based indices, converted to 0-based."
    return _step(call, Status.EXACT, [f"epochs.drop({idx}, reason='eeglab_rejepoch')"], [note])


# ---- epoching -------------------------------------------------------------


@translates("pop_epoch")
def _epoch(call: Call, ctx: Context) -> Step:
    types = call.positional[0] if call.positional else []
    lims = call.positional[1] if len(call.positional) > 1 else call.get("timelim")
    if not (isinstance(lims, list) and len(lims) == 2):
        raise NeedsManual("epoch limits not a 2-element vector")
    types = as_list(types)
    if any(isinstance(t, Expr) for t in types):
        raise NeedsManual("event types given as expressions")
    types = [str(int(t)) if isinstance(t, (int, float)) else t for t in types]
    code = ["events, event_id = mne.events_from_annotations(raw)"]
    if types:
        code.append(f"event_id = {{k: v for k, v in event_id.items() if k in {lit(types)}}}")
    code.append(f"epochs = mne.Epochs(raw, events, event_id=event_id, tmin={lit(lims[0])}, "
                f"tmax={lit(lims[1])}, baseline=None, preload=True)")
    notes = ["baseline=None because EEGLAB's pop_epoch does not baseline-correct; "
             "pop_rmbase does that separately.",
             "MNE includes the sample at tmax, so epochs may be one sample longer than EEGLAB's.",
             "Event names come from EEG.event.type via read_raw_eeglab annotations; check "
             "`event_id` if your types are numeric or padded (e.g. 'S  1')."]
    if call.has("valuelim"):
        notes.append("'valuelim' (absolute amplitude rejection at epoching) not translated; "
                     "add a pop_eegthresh-style step.")
    ctx.obj = "epochs"
    return _step(call, Status.APPROX, code, notes)


@translates("pop_rmbase")
def _rmbase(call: Call, ctx: Context) -> Step:
    trange = call.positional[0] if call.positional else []
    prange = call.positional[1] if len(call.positional) > 1 else []
    if ctx.obj == "raw":
        ctx.imports.add("import numpy as np")
        return _step(call, Status.EXACT,
                     ["raw.apply_function(lambda x: x - np.mean(x), picks='eeg')"],
                     ["On continuous data pop_rmbase removes each channel's mean."])
    if not is_empty(prange):
        raise NeedsManual("baseline given in samples (pointrange)")
    if is_empty(trange):
        return _step(call, Status.EXACT, ["epochs.apply_baseline((None, None))"],
                     ["Whole-epoch baseline."])
    if isinstance(trange, list) and len(trange) == 2:
        a, b = trange
        conv = lambda v: "None" if (isinstance(v, float) and math.isinf(v)) else lit(v / 1000)
        return _step(call, Status.EXACT, [f"epochs.apply_baseline(({conv(a)}, {conv(b)}))"],
                     [f"EEGLAB gives the baseline in milliseconds ({trange}); MNE uses seconds."])
    raise NeedsManual("unexpected baseline range")


# ---- ICA ------------------------------------------------------------------

_ICA_METHODS = {"runica": "infomax", "binica": "infomax", "picard": "picard", "fastica": "fastica"}


ICA_REUSE_NOTE = ("Reuses EEGLAB's own unmixing matrix, so components (and their indices) "
                  "match exactly. Use this when migrating already-processed data.")


def ica_reuse_code(ctx: Context) -> list[str]:
    """Code that loads EEGLAB's ICA, including the components it flagged."""
    ctx.has_ica = True
    ctx.imports.add("from pymatreader import read_mat")
    ctx.imports.add("import numpy as np")
    src_path = ctx.ica_source or ctx.input_path
    src = lit(src_path) if src_path else "ICA_SET  # TODO: .set file holding the ICA"
    return [f"ica_set = {src}",
            "ica = mne.preprocessing.read_ica_eeglab(ica_set)",
            "# read_ica_eeglab does not carry over components flagged for rejection,",
            "# so recover EEG.reject.gcompreject ourselves:",
            "_mat = read_mat(ica_set)",
            "_mat = _mat.get('EEG', _mat)",
            "_flags = np.atleast_1d(_mat.get('reject', {}).get('gcompreject', []))",
            "ica.exclude = [int(i) for i in np.flatnonzero(_flags)]"]


@translates("pop_runica", "runica")
def _runica(call: Call, ctx: Context) -> Step:
    ctx.has_ica = True
    if ctx.ica_mode == "reuse":
        return _step(call, Status.EXACT, ica_reuse_code(ctx), [ICA_REUSE_NOTE])

    icatype = str(call.get("icatype", "runica")).lower()
    method = _ICA_METHODS.get(icatype)
    if method is None:
        raise NeedsManual(f"icatype '{icatype}'")
    fit_params = ""
    if method == "infomax":
        fit_params = f", fit_params=dict(extended={truthy(call.get('extended', 0))})"
    n = call.get("pca", call.get("pcakeep"))
    n_comp = lit(n) if not is_empty(n) and not isinstance(n, Expr) else "None"
    ctx.ica_refit = True
    ctx.warnings.append(
        "ICA is refit in Python. Component order and signs will NOT match EEGLAB, so any "
        "hard-coded component numbers later in the script are meaningless. Re-select components "
        "(ICLabel or by eye), or run with --ica reuse to load EEGLAB's decomposition.")
    return _step(call, Status.APPROX,
                 [f"ica = mne.preprocessing.ICA(n_components={n_comp}, method={method!r}{fit_params}, "
                  "random_state=97, max_iter='auto')",
                  f"ica.fit({ctx.obj})"],
                 ["ICA is stochastic and implementations differ (EEGLAB runica vs MNE infomax "
                  "start values, stopping rules, PCA). Expect similar but not identical components."])


@translates("iclabel", "pop_iclabel")
def _iclabel(call: Call, ctx: Context) -> Step:
    ctx.imports.add("from mne_icalabel.iclabel import iclabel_label_components")
    return _step(call, Status.APPROX,
                 [f"ic_proba = iclabel_label_components({ctx.obj}, ica, inplace=True)  # (n_ics, 7)",
                  "ICLABEL_CLASSES = ['Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other']"],
                 ["Uses mne-icalabel, a port of the ICLabel network. It expects 1-100 Hz filtered, "
                  "average-referenced data; probabilities are close to EEGLAB's but not bit-identical.",
                  "Columns of ic_proba are in the same class order as EEG.etc.ic_classification."])


@translates("pop_icflag")
def _icflag(call: Call, ctx: Context) -> Step:
    thr = call.positional[0] if call.positional else None
    if not (isinstance(thr, list) and len(thr) == 7 and all(isinstance(r, list) and len(r) == 2 for r in thr)):
        raise NeedsManual("threshold matrix is not 7x2")
    ctx.imports.add("import numpy as np")
    rows = ",\n    ".join(lit(r) for r in thr)
    code = [f"icflag_thresholds = np.array([\n    {rows},\n])  # rows: ICLABEL_CLASSES; [min max] probability",
            "flagged = set()",
            "for cls, (lo, hi) in enumerate(icflag_thresholds):",
            "    if np.isnan(lo):",
            "        continue",
            "    hits = np.flatnonzero((ic_proba[:, cls] >= lo) & (ic_proba[:, cls] <= hi))",
            "    flagged.update(int(i) for i in hits)",
            "ica.exclude = sorted(set(ica.exclude) | flagged)",
            "print(f'Flagged ICs (0-based): {ica.exclude}')"]
    return _step(call, Status.EXACT, code,
                 ["Same rule as pop_icflag: an IC is flagged if any class probability falls inside "
                  "that class's [min max]. Indices are now 0-based."])


@translates("pop_subcomp")
def _subcomp(call: Call, ctx: Context) -> Step:
    comps = call.positional[0] if call.positional else []
    keep = truthy(call.positional[2]) if len(call.positional) > 2 else False
    o = ctx.obj
    if is_empty(comps):
        return _step(call, Status.EXACT, [f"ica.apply({o})"],
                     ["Empty component list = remove the ICs already flagged (EEG.reject.gcompreject), "
                      "which here is ica.exclude."])
    if isinstance(comps, Expr):
        if "gcompreject" in comps.text:
            return _step(call, Status.EXACT, [f"ica.apply({o})"],
                         ["find(EEG.reject.gcompreject) corresponds to ica.exclude."])
        raise NeedsManual("component list given as expression")
    idx = [int(c) - 1 for c in as_list(comps)]
    if keep:
        code = [f"ica.exclude = [i for i in range(ica.n_components_) if i not in {idx}]  # keep EEGLAB ICs {as_list(comps)}"]
    else:
        code = [f"ica.exclude = {idx}  # EEGLAB ICs {as_list(comps)} (1-based)"]
    code.append(f"ica.apply({o})")
    if ctx.ica_refit:
        return _step(call, Status.MANUAL, ["# TODO: component numbers below came from EEGLAB's ICA, "
                                            "not the one fit above. Re-select before running."] +
                     ["# " + c for c in code],
                     ["Hard-coded IC numbers cannot survive an ICA refit."])
    return _step(call, Status.EXACT, code, ["1-based EEGLAB indices converted to 0-based."])


@translates("pop_selectcomps")
def _selectcomps(call: Call, ctx: Context) -> Step:
    return _step(call, Status.MANUAL,
                 ["ica.plot_components(inst=" + ctx.obj + ")  # click topomaps to toggle exclusion",
                  "# ica.exclude now holds your choices"],
                 ["Interactive selection: MNE's equivalent is clicking components in plot_components."])


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def translate_call(call: Call, ctx: Context) -> Step:
    if not call.args and call.func in ("EEG", "ALLEEG", "OUTEEG"):
        return _step(call, Status.SKIPPED,
                     notes=["Dataset copy. MNE methods modify in place; use .copy() if you need the "
                            "original later."])
    fn = REGISTRY.get(call.func.lower())
    if fn is None:
        return _step(call, Status.MANUAL, [f"# TODO: no translation for `{call.func}`: {call.source}"],
                     [f"`{call.func}` is not in the translation table yet."])
    try:
        return fn(call, ctx)
    except NeedsManual as e:
        return _step(call, Status.MANUAL, [f"# TODO: {call.source}"], [str(e)])
