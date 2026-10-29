% Typical EEGLAB ERP preprocessing script (example input for eeglab2mne)
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

EEG = pop_loadset('filename', 'sub01.set', 'filepath', 'data/');
EEG = pop_chanedit(EEG, 'lookup', 'standard_1005.elc');
EEG = pop_select(EEG, 'nochannel', {'HEOG' 'VEOG'});
EEG = pop_resample(EEG, 250);

% Filter 1-40 Hz and remove line noise
EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40, 'plotfreqz', 0);
EEG = pop_cleanline(EEG, 'linefreqs', [50 100], 'bandwidth', 2);

% Bad channels and bursts
originalEEG = EEG;
EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, ...
    'LineNoiseCriterion', 4, 'Highpass', 'off', 'BurstCriterion', 20, ...
    'WindowCriterion', 0.25, 'BurstRejection', 'on', 'Distance', 'Euclidian');
EEG = pop_interp(EEG, originalEEG.chanlocs, 'spherical');
EEG = pop_reref(EEG, []);

% ICA + ICLabel
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
EEG = pop_iclabel(EEG, 'default');
EEG = pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);
EEG = pop_subcomp(EEG, [], 0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'gui', 'off');

% Epoch, baseline, reject
EEG = pop_epoch(EEG, {'S  1' 'S  2'}, [-0.2 0.8], 'newname', 'sub01 epochs', 'epochinfo', 'yes');
EEG = pop_rmbase(EEG, [-200 0], []);
EEG = pop_eegthresh(EEG, 1, 1:30, -100, 100, -0.2, 0.796, 0, 1);
EEG = eeg_checkset(EEG);
EEG = pop_saveset(EEG, 'filename', 'sub01_epochs.set', 'filepath', 'data/');
